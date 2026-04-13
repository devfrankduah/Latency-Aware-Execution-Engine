"""
Data ingestion, validation, and storage.

Usage:
    from src.data.loader import load_klines_from_csv, load_klines_directory, save_processed
    from src.data.validator import validate_klines, ValidationReport
    from src.data.schemas import KlineSchema, TradeSchema
"""

from src.data.schemas import KlineSchema, TradeSchema

