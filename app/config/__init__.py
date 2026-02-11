"""Configuration module for The Earnings Hunter app."""

from app.config.theme import (
    COLORS,
    PAGE_CSS,
    CARD_CSS,
    METRIC_CSS,
    SENTIMENT_CSS,
    MARKET_STATUS_CSS,
    HEADER_CSS,
    SIDEBAR_NAV_CSS,
    get_plotly_layout,
    get_sparkline_layout,
    format_number,
    format_percent,
    get_change_color,
    get_sentiment_class,
    inject_css,
    get_all_css,
)

__all__ = [
    "COLORS",
    "PAGE_CSS",
    "CARD_CSS",
    "METRIC_CSS",
    "SENTIMENT_CSS",
    "MARKET_STATUS_CSS",
    "HEADER_CSS",
    "SIDEBAR_NAV_CSS",
    "get_plotly_layout",
    "get_sparkline_layout",
    "format_number",
    "format_percent",
    "get_change_color",
    "get_sentiment_class",
    "inject_css",
    "get_all_css",
]
