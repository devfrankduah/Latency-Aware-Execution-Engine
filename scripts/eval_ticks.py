#!/usr/bin/env python3
"""
Tick-Level Evaluation - FIXED.

v1 bug: created a new Env per episode (200x feature compute), agent got 0 valid.
Fix: aggregate ALL ticks into klines ONCE, create ONE Env, run agent normally.

Usage:
    python scripts/eval_ticks.py --tick-dir data/raw/trades/BTCUSDT --model models/multi/best.pt --qty 50
"""
import argparse, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def load_ticks(tick_dir, max_rows=10_000_000):
    files = sorted(Path(tick_dir).glob('*.csv'))
    if not files: log.error(f'No CSVs in {tick_dir}'); return None
    log.info(f'Loading {len(files)} tick files...')
    chunks = []; total = 0
    for f in files:
        try:
            df = pd.read_csv(f, header=None, names=['trade_id','price','qty','quote_qty','time','is_buyer_maker','is_best_match'],
                             dtype={'price':np.float64,'qty':np.float64})
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            chunks.append(df[['timestamp','price','qty','is_buyer_maker']]); total += len(df)
            log.info(f'  {f.name}: {len(df):,} trades')
            if total >= max_rows: break
        except Exception as e: log.warning(f'  Skip {f.name}: {e}')
    ticks = pd.concat(chunks, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    log.info(f'Total: {len(ticks):,} trades ({ticks["timestamp"].min()} -> {ticks["timestamp"].max()})')
    return ticks

def ticks_to_klines(ticks):
    log.info('Aggregating ticks into 1-min klines...')
    ticks = ticks.copy()
    ticks['minute'] = ticks['timestamp'].dt.floor('1min')
    k = ticks.groupby('minute').agg(open=('price','first'), high=('price','max'), low=('price','min'),
        close=('price','last'), volume=('qty','sum'), num_trades=('price','count')).reset_index()
    k.rename(columns={'minute':'timestamp'}, inplace=True)
    k['quote_volume'] = k['volume'] * k['close']; k['symbol'] = 'BTCUSDT'
    log.info(f'Klines: {len(k):,} bars')
    return k

def tick_execute(ticks, start_time, qty, horizon=60, policy='twap', max_part=0.15):
    end = start_time + pd.Timedelta(minutes=horizon)
    w = ticks[(ticks['timestamp']>=start_time)&(ticks['timestamp']<end)]
    if len(w) < 100: return None
    w = w.copy(); w['minute'] = w['timestamp'].dt.floor('1min')
    bars = w.groupby('minute').agg(volume=('qty','sum'),
        vwap=('price', lambda x: np.average(x, weights=w.loc[x.index,'qty'])),
        spread=('price', lambda x: x.quantile(0.75)-x.quantile(0.25) if len(x)>3 else x.max()-x.min())).reset_index()
    n = len(bars)
    if n < 5: return None
    if policy=='twap': sc = np.ones(n)/n
    elif policy=='vwap': v=bars['volume'].values; sc=v/v.sum() if v.sum()>0 else np.ones(n)/n
    elif policy=='immediate': sc=np.zeros(n); sc[0]=1.0
    else: sc=np.ones(n)/n
    rem=qty; cost=0.0; filled=0.0; arrival=w.iloc[0]['price']
    for i,(_,bar) in enumerate(bars.iterrows()):
        if rem<1e-10: break
        tgt=qty*sc[i]; act=min(tgt, bar['volume']*max_part, rem)
        if act>0: fp=bar['vwap']+bar['spread']*0.5; cost+=act*fp; filled+=act; rem-=act
    if filled<1e-10: return None
    avg=cost/filled; is_bps=(avg-arrival)/arrival*10_000
    return {'cost':cost,'fill':filled/qty,'is_bps':is_bps}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tick-dir', required=True)
    parser.add_argument('--model', default='models/multi/best.pt')
    parser.add_argument('--qty', type=float, default=50.0)
    parser.add_argument('--n-episodes', type=int, default=200)
    parser.add_argument('--max-rows', type=int, default=10_000_000)
    args = parser.parse_args()

    ticks = load_ticks(args.tick_dir, args.max_rows)
    if ticks is None: return
    klines = ticks_to_klines(ticks)
    from src.features.engine import compute_all_features
    klines = compute_all_features(klines)
    log.info(f'Features: {len(klines)} bars')

    rng = np.random.default_rng(42)
    min_t = ticks['timestamp'].min() + pd.Timedelta(hours=1)
    max_t = ticks['timestamp'].max() - pd.Timedelta(hours=2)
    offsets = sorted(rng.uniform(0, (max_t-min_t).total_seconds(), size=args.n_episodes))
    start_times = [min_t + pd.Timedelta(seconds=s) for s in offsets]

    # Baselines on tick data
    log.info(f'Running baselines ({args.n_episodes} episodes)...')
    results = {'Immediate':[], 'TWAP':[], 'VWAP':[]}
    for i, st in enumerate(start_times):
        for name, pol in [('Immediate','immediate'),('TWAP','twap'),('VWAP','vwap')]:
            r = tick_execute(ticks, st, args.qty, policy=pol)
            if r and r['fill'] > 0.5: results[name].append(r)
        if (i+1)%50==0: log.info(f'  {i+1}/{args.n_episodes}')

    # Agent on kline-aggregated data
    log.info('Running ML agent...')
    agent_results = []
    mp = Path(args.model)
    if mp.exists():
        from scripts.train_large import Env, Agent
        env = Env(klines, qty=args.qty)
        agent = Agent(); agent.load(str(mp)); agent.eps = 0.0
        kt = klines['timestamp'].values

        for i, st in enumerate(start_times):
            idx = np.searchsorted(kt, np.datetime64(st))
            if idx < 70 or idx > len(klines) - 61: continue
            s = env.reset(si=idx)
            while not env.done: s,_,_,_ = env.step(agent.act(s, greedy=True))
            agent_results.append(env.metrics())
        log.info(f'  Agent: {len(agent_results)} valid episodes')
    else:
        log.warning(f'  Model not found: {mp}')

    # Print
    print(f'\n{"="*80}')
    print(f'  TICK-LEVEL EVALUATION - {args.qty} BTC')
    print(f'  {len(ticks):,} trades -> {len(klines):,} kline bars')
    print(f'{"="*80}')
    print(f'\n  Baselines (real tick fills):')
    print(f'  {"Strategy":<14s} {"IS(bps)":>10s} {"Std":>8s} {"Cost($)":>14s} {"Fill":>7s} {"N":>5s}')
    print(f'  {"─"*60}')
    for name in ['Immediate','TWAP','VWAP']:
        r = results[name]
        if r: print(f'  {name:<14s} {np.mean([x["is_bps"] for x in r]):>+10.2f} '
                     f'{np.std([x["is_bps"] for x in r]):>8.2f} '
                     f'${np.mean([x["cost"] for x in r]):>13.2f} '
                     f'{np.mean([x["fill"] for x in r]):>6.1%} {len(r):>5d}')
    tc = np.mean([r['cost'] for r in results['TWAP']]) if results['TWAP'] else 0
    vc = np.mean([r['cost'] for r in results['VWAP']]) if results['VWAP'] else 0
    if tc>0 and vc>0: print(f'\n  VWAP vs TWAP (tick): {(1-vc/tc)*100:+.2f}%')

    if agent_results:
        print(f'\n  ML Agent (simulated impact on tick-derived klines):')
        vv=[r['vs_vwap'] for r in agent_results]; vt=[r['vs_twap'] for r in agent_results]
        ac=[r['cost'] for r in agent_results]; vc2=[r['vwap_cost'] for r in agent_results]; tc2=[r['twap_cost'] for r in agent_results]
        print(f'  {"Strategy":<14s} {"Cost($)":>14s} {"vs TWAP":>10s} {"vs VWAP":>10s}')
        print(f'  {"─"*50}')
        print(f'  {"TWAP (sim)":<14s} ${np.mean(tc2):>13.2f}')
        print(f'  {"VWAP (sim)":<14s} ${np.mean(vc2):>13.2f} {(1-np.mean(vc2)/np.mean(tc2))*100:>+9.1f}%')
        print(f'  {"ML Agent":<14s} ${np.mean(ac):>13.2f} {(1-np.mean(ac)/np.mean(tc2))*100:>+9.1f}% '
              f'{(1-np.mean(ac)/np.mean(vc2))*100:>+9.1f}%')
        print(f'\n  vs VWAP: {np.mean(vv):>+.3f} bps | Beat: {np.mean([v>0 for v in vv])*100:.0f}%')
        print(f'  vs TWAP: {np.mean(vt):>+.3f} bps | Beat: {np.mean([v>0 for v in vt])*100:.0f}%')
    print(f'{"="*80}')

if __name__ == '__main__':
    main()