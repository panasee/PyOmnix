"""
Market utilities for working with different stock markets.

This module provides utility functions for detecting market types,
formatting ticker symbols, and other market-specific operations.
"""

import re
from typing import Tuple

from .data_models import MarketType


def detect_market(ticker: str) -> MarketType:
    """
    Detect the market type based on the ticker symbol.
    
    Args:
        ticker: The ticker symbol to check
        
    Returns:
        MarketType: The detected market type (US or CHINA)
    """
    # Chinese stock codes are typically 6 digits
    if re.match(r'^\d{6}$', ticker):
        return MarketType.CHINA
    # US stock codes are typically 1-5 alphanumeric characters
    else:
        return MarketType.US


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol to a standard format.
    
    Args:
        ticker: The ticker symbol to normalize
        
    Returns:
        str: The normalized ticker symbol
    """
    # Remove any whitespace
    ticker = ticker.strip()
    
    # Handle Chinese tickers
    if detect_market(ticker) == MarketType.CHINA:
        # Ensure it's just the 6 digits without market prefix
        if re.match(r'^\d{6}$', ticker):
            return ticker
        # Extract the 6 digits if it has a market prefix
        match = re.search(r'(\d{6})', ticker)
        if match:
            return match.group(1)
    
    # For US tickers, just return uppercase
    return ticker.upper()


def get_exchange_prefix(ticker: str) -> Tuple[str, str]:
    """
    Get the exchange prefix for a ticker symbol.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Tuple[str, str]: A tuple containing (exchange_prefix, ticker_without_prefix)
    """
    market = detect_market(ticker)
    
    if market == MarketType.CHINA:
        # Determine if it's Shanghai (sh) or Shenzhen (sz) based on first digit
        if ticker.startswith(('0', '3')):
            return 'sz', ticker
        elif ticker.startswith(('6', '9')):
            return 'sh', ticker
        else:
            # Default to Shanghai for unknown patterns
            return 'sh', ticker
    else:
        # US market doesn't use exchange prefixes in the same way
        return '', ticker


def format_ticker_for_api(ticker: str) -> str:
    """
    Format ticker for API calls based on the market.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        str: The formatted ticker symbol
    """
    normalized = normalize_ticker(ticker)
    market = detect_market(normalized)
    
    if market == MarketType.CHINA:
        # Some Chinese APIs require the exchange prefix
        prefix, _ = get_exchange_prefix(normalized)
        if prefix:
            return f"{prefix}{normalized}"
        return normalized
    else:
        # US tickers are typically used as-is
        return normalized
