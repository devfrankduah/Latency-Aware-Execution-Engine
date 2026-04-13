#!/usr/bin/env python3
"""
Multi-Asset Training - FIXED.

The bug in v1: mixing BTC ($50K), ETH ($2K), SOL ($50) into one DataFrame
broke all normalization. vs TWAP showed +2858 bps (impossible).

The fix: SEPARATE environment per asset. ONE shared agent rotates between them.
Each env normalizes against its own price/volume scale.

Training: Agent cycles BTC→ETH→SOL→BTC→... (2020-2023, 4 years each)
Validation: Each asset separately (2024 Jan-Jun)
Test: Each asset separately (2024 Jul-Dec)

Usage:
    python scripts/train_multi.py --train --episodes 50000 --qty 50
    python scripts/train_multi.py --train --episodes 3000 --qty 50 --synthetic
"""

import argparse, logging, math, sys, time
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ACTIONS = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0])
N_ACT = len(ACTIONS); VWAP_IDX = 3; STATE_DIM = 14

class Env:
    def __init__(self, df, qty=50.0, horizon=60, impact=0.3, max_part=0.15, name=''):
        self.qty, self.horizon, self.impact, self.max_part, self.name = qty, horizon, impact, max_part, name
        self.rng = np.random.default_rng()
        self._c = df['close'].values.astype(np.float64); self._v = df['volume'].values.astype(np.float64)
        raw = ((df['high']-df['low'])/df['close']).values.astype(np.float64)
        med = np.nanmedian(raw[raw>0])
        self._spread = np.clip(np.nan_to_num(raw/med*0.0002, nan=0.0002), 0.00005, 0.002)
        def _s(n): return np.nan_to_num(df[n].values.astype(np.float64),nan=0.0) if n in df.columns else np.zeros(len(df))
        self._rvol=_s('rolling_volatility'); self._vimb=_s('volume_imbalance'); self._sprd_raw=_s('spread_proxy')
        self._r5=_s('return_5bar'); self._r20=_s('return_20bar'); self._hsin=_s('hour_sin'); self._hcos=_s('hour_cos')
        vm=pd.Series(self._v).rolling(60,min_periods=10).mean().values
        self._vr=np.clip(np.nan_to_num(self._v/np.where(vm>0,vm,1.0),nan=1.0),0,5)
        sm=pd.Series(self._spread).rolling(60,min_periods=10).mean().values
        ss=pd.Series(self._spread).rolling(60,min_periods=10).std().values
        self._sprd_z=np.clip(np.nan_to_num((self._spread-sm)/np.where(ss>0,ss,1e-6),nan=0.0),-3,3)
        v=self._rvol[self._rvol>0]; self._vm,self._vs=(float(v.mean()),float(v.std())) if len(v)>0 else (1.0,1.0)
        s=self._sprd_raw[self._sprd_raw>0]; self._sm,self._ss=(float(s.mean()),float(s.std())) if len(s)>0 else (1e-4,1e-4)
        self._starts=np.arange(70,len(df)-horizon-1); self._reset()
        mv=np.median(self._v[self._v>0]); mp=np.median(self._c[self._c>0])
        log.info(f'  {name}: {len(df):,} bars, ${mp:,.0f}, vol={mv:.1f}, part={qty/horizon/max(mv,1):.2%}')
    def _reset(self):
        self.idx=self.start=0; self.rem=self.arr=self.cost=self.exe=0.0; self.step_n=0; self.done=False; self._mc=self._vc=0.0
    def reset(self, si=None):
        self._reset(); self.start=si if si is not None else self.rng.choice(self._starts)
        self.idx=self.start; self.rem=self.qty; self.arr=self._c[self.start]; return self._obs()
    def _fill(self, q, i):
        v,p=self._v[i],self._c[i]
        if q<1e-10 or v<1e-10: return 0.0,0.0
        pr=min(q/v,self.max_part); act=min(pr*v,self.rem)
        return act,(p*self._spread[i]*0.5+self.impact*p*(pr**1.5))*act
    def _vwap_qty(self, rem, bl):
        if bl<=0: return rem
        return (rem/bl)*self._vr[min(self.idx,len(self._vr)-1)]
    def step(self, action):
        if self.done: return self._obs(),0.0,True,{}
        bl=self.horizon-self.step_n; vt=self._vwap_qty(self.rem,bl)
        if bl<=3 and self.rem>0.01*self.qty: tgt=self.rem/max(bl,1)
        else: tgt=vt*ACTIONS[action]
        act,c=self._fill(tgt,self.idx); _,vc=self._fill(vt,self.idx)
        self.rem-=act; self.exe+=act; self.cost+=c; self._mc+=c; self._vc+=vc
        self.step_n+=1; self.done=self.step_n>=self.horizon or self.rem<1e-8
        reward=0.0
        if self.done:
            if self.exe/self.qty<0.90: reward=-20.0
            else: vw=self._run_vwap(); n=self.arr*self.qty; reward=(vw-self.cost)/max(n,1)*10_000
        self.idx+=1; return self._obs(),float(reward),self.done,{}
    def _run_vwap(self):
        rem,tot=self.qty,0.0
        for s in range(self.horizon):
            i=self.start+s
            if i>=len(self._c): break
            q=(rem/max(self.horizon-s,1))*self._vr[min(i,len(self._vr)-1)]; v,p=self._v[i],self._c[i]
            if q<1e-10 or v<1e-10: continue
            pr=min(q/v,self.max_part); act=min(pr*v,rem)
            tot+=(p*self._spread[i]*0.5+self.impact*p*(pr**1.5))*act; rem-=act
        return tot
    def _run_twap(self):
        rem,tot=self.qty,0.0
        for s in range(self.horizon):
            i=self.start+s
            if i>=len(self._c): break
            q=rem/max(self.horizon-s,1); v,p=self._v[i],self._c[i]
            if q<1e-10 or v<1e-10: continue
            pr=min(q/v,self.max_part); act=min(pr*v,rem)
            tot+=(p*self._spread[i]*0.5+self.impact*p*(pr**1.5))*act; rem-=act
        return tot
    def _obs(self):
        i=min(self.idx,len(self._c)-1); h=self.horizon; fp,tp=self.exe/self.qty,self.step_n/max(h,1)
        ca=np.clip((self._vc-self._mc)/max(self._vc,1e-10),-1,1) if self._vc>1e-10 else 0.0
        pm=np.clip((self._c[i]-self.arr)/max(self.arr,1e-10)*100,-5,5) if self.arr>0 else 0.0
        return np.array([self.rem/self.qty,max(0,h-self.step_n)/h,
            (self._rvol[i]-self._vm)/max(self._vs,1e-8),np.clip(self._vimb[i],0,5)/5.0,
            (self._sprd_raw[i]-self._sm)/max(self._ss,1e-8),np.clip(self._r5[i]*100,-5,5),
            np.clip(self._r20[i]*100,-5,5),self._hsin[i],self._hcos[i],
            float(ca),float(fp-tp),float(pm),np.clip(self._vr[i],0,5)/5.0,
            np.clip(self._sprd_z[i]/3.0,-1,1)],dtype=np.float32)
    def metrics(self):
        vw=self._run_vwap(); tw=self._run_twap(); n=self.arr*self.qty
        return {'cost':self.cost,'fill':self.exe/self.qty,'vwap_cost':vw,'twap_cost':tw,
                'vs_vwap':(vw-self.cost)/n*10_000 if n>0 else 0,'vs_twap':(tw-self.cost)/n*10_000 if n>0 else 0}

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(STATE_DIM,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,256),nn.LayerNorm(256),nn.ReLU())
        self.val=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        self.adv=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,N_ACT))
    def forward(self,x): s=self.shared(x); v,a=self.val(s),self.adv(s); return v+a-a.mean(-1,keepdim=True)

class Agent:
    def __init__(self,lr=3e-4):
        self.online=QNet(); self.target=QNet(); self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt=optim.AdamW(self.online.parameters(),lr=lr,weight_decay=1e-5); self.base_lr=lr; cap=500_000
        self.buf_s=np.zeros((cap,STATE_DIM),np.float32); self.buf_a=np.zeros(cap,np.int64)
        self.buf_r=np.zeros(cap,np.float32); self.buf_ns=np.zeros((cap,STATE_DIM),np.float32)
        self.buf_d=np.zeros(cap,np.float32); self.buf_p=np.ones(cap,np.float64)
        self.buf_pos=self.buf_size=0; self.buf_cap=cap; self.eps=1.0; self.steps=0; self.gamma=0.97; self.tau=0.005; self.bs=128
    def push(self,s,a,r,ns,d):
        i=self.buf_pos; self.buf_s[i]=s; self.buf_a[i]=a; self.buf_r[i]=r; self.buf_ns[i]=ns; self.buf_d[i]=d
        self.buf_p[i]=self.buf_p[:self.buf_size].max() if self.buf_size>0 else 1.0
        self.buf_pos=(i+1)%self.buf_cap; self.buf_size=min(self.buf_size+1,self.buf_cap)
    def set_lr(self,ep,total):
        lr=self.base_lr*0.5*(1+math.cos(math.pi*ep/total))
        for pg in self.opt.param_groups: pg['lr']=max(lr,1e-5)
    def act(self,s,greedy=False):
        if not greedy and np.random.random()<self.eps: return np.random.randint(N_ACT)
        with torch.no_grad(): return self.online(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
    def learn(self):
        if self.buf_size<5000: return 0.0
        beta=min(1.0,0.4+self.steps*0.6/150_000); p=self.buf_p[:self.buf_size]**0.6; p/=p.sum()
        idx=np.random.choice(self.buf_size,self.bs,p=p,replace=False)
        w=(self.buf_size*p[idx])**(-beta); w/=w.max()
        s=torch.FloatTensor(self.buf_s[idx]); a=torch.LongTensor(self.buf_a[idx])
        r=torch.FloatTensor(self.buf_r[idx]); ns=torch.FloatTensor(self.buf_ns[idx])
        d=torch.FloatTensor(self.buf_d[idx]); wt=torch.FloatTensor(w)
        q=self.online(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            na=self.online(ns).argmax(1); nq=self.target(ns).gather(1,na.unsqueeze(1)).squeeze(1)
            tgt=r+self.gamma*nq*(1-d)
        td=(q-tgt).detach().numpy(); loss=(wt*nn.SmoothL1Loss(reduction='none')(q,tgt)).mean()
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(),0.5); self.opt.step()
        for ii,t in zip(idx,td): self.buf_p[ii]=abs(t)+1e-6
        with torch.no_grad():
            for po,pt in zip(self.online.parameters(),self.target.parameters()):
                pt.data.copy_(self.tau*po.data+(1-self.tau)*pt.data)
        self.steps+=1; return loss.item()
    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        torch.save({'on':self.online.state_dict(),'tg':self.target.state_dict(),'opt':self.opt.state_dict(),'steps':self.steps,'eps':self.eps},path)
    def load(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        self.online.load_state_dict(ck['on']); self.target.load_state_dict(ck['tg']); self.opt.load_state_dict(ck['opt']); self.steps=ck['steps']; self.eps=ck['eps']

def evaluate(agent,env,n=500):
    old=agent.eps; agent.eps=0.0; vv,vt=[],[]
    for _ in range(n):
        s=env.reset()
        while not env.done: s,_,_,_=env.step(agent.act(s,True))
        m=env.metrics(); vv.append(m['vs_vwap']); vt.append(m['vs_twap'])
    agent.eps=old
    return {'vs_vwap':np.mean(vv),'vs_vwap_std':np.std(vv),'beat_vwap':np.mean([s>0 for s in vv])*100,
            'vs_twap':np.mean(vt),'vs_twap_std':np.std(vt),'beat_twap':np.mean([s>0 for s in vt])*100}

def load_asset(data_dir, symbol):
    from src.features.engine import compute_all_features
    path=Path(data_dir)/f'{symbol}_klines_1m.parquet'
    if not path.exists(): log.warning(f'{symbol}: not found'); return None,None,None
    df=pd.read_parquet(path); log.info(f'  {symbol}: {len(df):,} bars')
    df['year']=df['timestamp'].dt.year; df['month']=df['timestamp'].dt.month
    train=df[df['year']<=2023].copy().reset_index(drop=True)
    val=df[(df['year']==2024)&(df['month']<=6)].copy().reset_index(drop=True)
    test=df[(df['year']==2024)&(df['month']>6)].copy().reset_index(drop=True)
    for d in [train,val,test]: d.drop(columns=['year','month'],inplace=True,errors='ignore')
    train=compute_all_features(train); val=compute_all_features(val); test=compute_all_features(test)
    log.info(f'    Train={len(train):,} Val={len(val):,} Test={len(test):,}')
    return train,val,test

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',action='store_true'); parser.add_argument('--data-dir',default='data/processed')
    parser.add_argument('--synthetic',action='store_true'); parser.add_argument('--episodes',type=int,default=50000)
    parser.add_argument('--qty',type=float,default=50.0); parser.add_argument('--save-dir',default='models/multi')
    args=parser.parse_args()
    from src.features.engine import compute_all_features

    if args.synthetic:
        log.info('Generating synthetic...'); rng=np.random.default_rng(42); asset_data={}
        for sym,bp,bv in [('BTCUSDT',40000,30),('ETHUSDT',2000,200),('SOLUSDT',50,5000)]:
            n=100_000; c=np.zeros(n); c[0]=bp
            for i in range(1,n): c[i]=c[i-1]*np.exp(rng.normal(0,0.0008))
            noise=rng.uniform(0.0001,0.002,n); bvol=rng.exponential(bv,n)
            for i in range(n):
                h=(i%1440)//60; bvol[i]*=[0.3,0.2,0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0,2.5,2.0,1.8,2.2,2.5,2.8,3.0,2.5,2.0,1.5,1.0,0.7,0.5,0.4][h]
            df=pd.DataFrame({'timestamp':pd.date_range('2023-01-01',periods=n,freq='1min',tz='UTC'),
                'open':np.roll(c,1),'high':c*(1+noise),'low':c*(1-noise),'close':c,'volume':bvol,
                'quote_volume':bvol*c,'num_trades':rng.integers(100,2000,n),'symbol':sym})
            df.iloc[0,df.columns.get_loc('open')]=bp
            df['high']=df[['open','high','close']].max(axis=1); df['low']=df[['open','low','close']].min(axis=1)
            nt=len(df)
            asset_data[sym]={'train':compute_all_features(df.iloc[:int(nt*0.7)].copy().reset_index(drop=True)),
                'val':compute_all_features(df.iloc[int(nt*0.7):int(nt*0.85)].copy().reset_index(drop=True)),
                'test':compute_all_features(df.iloc[int(nt*0.85):].copy().reset_index(drop=True))}
    else:
        log.info('Loading assets separately...'); asset_data={}
        for sym in ['BTCUSDT','ETHUSDT','SOLUSDT']:
            tr,va,te=load_asset(args.data_dir,sym)
            if tr is not None: asset_data[sym]={'train':tr,'val':va,'test':te}

    available=list(asset_data.keys()); log.info(f'Assets: {available}')

    train_envs={sym:Env(d['train'],qty=args.qty,name=f'{sym}-train') for sym,d in asset_data.items()}
    val_envs={sym:Env(d['val'],qty=args.qty,name=f'{sym}-val') for sym,d in asset_data.items()}
    test_envs={sym:Env(d['test'],qty=args.qty,name=f'{sym}-test') for sym,d in asset_data.items()}
    env_list=list(train_envs.values())

    agent=Agent(lr=3e-4); sd=Path(args.save_dir); sd.mkdir(parents=True,exist_ok=True)
    n_ep=args.episodes; explore_until=int(n_ep*0.40); eps_floor=0.06; eval_every=max(500,n_ep//25)

    log.info('Warmstarting (3000 episodes cycling assets)...')
    for i in range(3000):
        env=env_list[i%len(env_list)]; s=env.reset()
        while not env.done:
            r=np.random.random()
            if r<0.45: a=VWAP_IDX
            elif r<0.60: a=np.random.choice([4,5,6])
            elif r<0.75: a=np.random.choice([0,1,2])
            else: a=np.random.randint(N_ACT)
            a=max(0,min(a,N_ACT-1)); ns,rw,_,_=env.step(a); agent.push(s,a,rw,ns,float(env.done)); s=ns
    log.info(f'Buffer: {agent.buf_size:,}')

    for sym,env in val_envs.items():
        bl=evaluate(agent,env,n=200)
        log.info(f'  {sym} baseline: vsVWAP={bl["vs_vwap"]:+.3f} vsTWAP={bl["vs_twap"]:+.3f}')

    log.info(f'\n{"="*70}')
    log.info(f'  MULTI-ASSET: {n_ep:,} ep | {args.qty} BTC | {len(available)} assets (separate envs)')
    log.info(f'{"="*70}\n')

    best_vwap=-999; t0=time.time()
    for ep in range(1,n_ep+1):
        agent.eps=max(eps_floor,1.0-(1.0-eps_floor)*ep/explore_until); agent.set_lr(ep,n_ep)
        env=env_list[ep%len(env_list)]; s=env.reset()
        while not env.done:
            a=agent.act(s); ns,r,_,_=env.step(a); agent.push(s,a,r,ns,float(env.done)); agent.learn(); s=ns
        if ep%eval_every==0:
            all_vw,all_tw=[],[]
            parts=[]
            for sym,venv in val_envs.items():
                v=evaluate(agent,venv,n=300); all_vw.append(v['vs_vwap']); all_tw.append(v['vs_twap'])
                parts.append(f'{sym[:3]}:{v["vs_vwap"]:+.1f}/{v["vs_twap"]:+.1f}')
            avg_vw,avg_tw=np.mean(all_vw),np.mean(all_tw); el=time.time()-t0; mk=''
            if avg_vw>best_vwap: best_vwap=avg_vw; agent.save(sd/'best.pt'); mk=' '
            eta=(n_ep-ep)/max(ep/el,0.1)/3600
            log.info(f'  Ep {ep:>6,}/{n_ep:,} │ ε={agent.eps:.3f} │ '
                     f'Avg vw:{avg_vw:>+7.2f} tw:{avg_tw:>+7.2f} │ {" ".join(parts)} │ '
                     f'{ep/el:.1f}/s ETA:{eta:.1f}h{mk}')
        if ep%5000==0: agent.save(sd/f'checkpoint_{ep}.pt')

    agent.save(sd/'final.pt'); log.info(f'\nDone: {n_ep:,} ep in {(time.time()-t0)/3600:.1f}h')

    bp=sd/'best.pt'
    if bp.exists(): agent.load(bp)
    agent.eps=0.0

    print(f'\n{"="*80}')
    print(f'  OUT-OF-SAMPLE - {args.qty} BTC | Multi-Asset (2024 H2)')
    print(f'{"="*80}')
    print(f'  {"Asset":<12s} {"vs VWAP":>12s} {"Beat%":>7s} {"vs TWAP":>12s} {"Beat%":>7s}')
    print(f'  {"─"*52}')
    all_vw,all_tw=[],[]
    for sym,tenv in test_envs.items():
        r=evaluate(agent,tenv,n=500)
        mk='[+]' if r['vs_vwap']>0 else '[~]' if abs(r['vs_vwap'])<1.0 else '[-]'
        print(f'  {sym:<12s} {r["vs_vwap"]:>+12.3f} {r["beat_vwap"]:>6.0f}%  {r["vs_twap"]:>+12.3f} {r["beat_twap"]:>6.0f}%  {mk}')
        all_vw.append(r['vs_vwap']); all_tw.append(r['vs_twap'])
    print(f'  {"─"*52}')
    print(f'  {"AVERAGE":<12s} {np.mean(all_vw):>+12.3f} {"":>7s} {np.mean(all_tw):>+12.3f}')
    print(f'{"="*80}')
    if np.mean(all_tw)>0: print(f'\n  Beats TWAP across all assets (avg {np.mean(all_tw):+.1f} bps)')
    rows=[]
    for sym,tenv in test_envs.items():
        r=evaluate(agent,tenv,n=500); rows.append({'asset':sym,**r})
    pd.DataFrame(rows).to_csv(sd/'results.csv',index=False)

if __name__=='__main__': main()