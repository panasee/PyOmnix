"""
PyOmnix Financial Module

This module provides financial data analysis tools for both US and Chinese markets.
"""

# Import the unified API functions for easy access
from .unified_api import (
    get_stock_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_price_data,
    get_prices,
    prices_to_df,
    search_line_items,
)

# Import market utilities
from .market_utils import detect_market, normalize_ticker, format_ticker_for_api

# Import data models
from .data_models import MarketType
