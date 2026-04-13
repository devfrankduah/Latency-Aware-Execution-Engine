"""
Data schemas and contracts for the execution engine.

PRODUCTION PATTERN: Schema-first design.
Why? Because:
  1. Everyone on the team knows exactly what columns exist and their types
  2. Data validation catches issues BEFORE they corrupt your model
  3. Changing the schema is explicit, visible, and reviewed in Git
  4. Downstream code can rely on these guarantees
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KlineSchema:
    """Schema for OHLCV kline/candlestick data.

    This is the standard format after ingestion and cleaning.
    Every module downstream depends on these column names.

    frozen=True means instances are immutable - prevents accidental modification.
    """

    # Column names (the single source of truth)
    TIMESTAMP = "timestamp"         # pd.Timestamp, UTC, bar open time
    OPEN = "open"                   # float64, opening price
    HIGH = "high"                   # float64, highest price in bar
    LOW = "low"                     # float64, lowest price in bar
    CLOSE = "close"                 # float64, closing price
    VOLUME = "volume"               # float64, base asset volume (e.g., BTC)
    QUOTE_VOLUME = "quote_volume"   # float64, quote asset volume (e.g., USDT)
    TRADES = "num_trades"           # int64, number of trades in bar
    SYMBOL = "symbol"               # str, trading pair (e.g., "BTCUSDT")

    @classmethod
    def required_columns(cls) -> list[str]:
        """Columns that MUST be present after loading."""
        return [
            cls.TIMESTAMP,
            cls.OPEN,
            cls.HIGH,
            cls.LOW,
            cls.CLOSE,
            cls.VOLUME,
        ]

    @classmethod
    def all_columns(cls) -> list[str]:
        """All columns including optional ones."""
        return [
            cls.TIMESTAMP,
            cls.OPEN,
            cls.HIGH,
            cls.LOW,
            cls.CLOSE,
            cls.VOLUME,
            cls.QUOTE_VOLUME,
            cls.TRADES,
            cls.SYMBOL,
        ]


@dataclass(frozen=True)
class TradeSchema:
    """Schema for individual tick/trade data.

    Each row = one executed trade on the exchange.
    This is the richest data - enables realistic execution simulation.
    """

    TRADE_ID = "trade_id"           # int64, exchange-assigned ID
    TIMESTAMP = "timestamp"         # pd.Timestamp, UTC, trade time
    PRICE = "price"                 # float64, execution price
    QUANTITY = "quantity"           # float64, trade size in base asset
    QUOTE_QTY = "quote_quantity"    # float64, trade size in quote asset
    IS_BUYER_MAKER = "is_buyer_maker"  # bool, True if buyer was maker
    SYMBOL = "symbol"               # str, trading pair

    @classmethod
    def required_columns(cls) -> list[str]:
        return [
            cls.TRADE_ID,
            cls.TIMESTAMP,
            cls.PRICE,
            cls.QUANTITY,
            cls.IS_BUYER_MAKER,
        ]
