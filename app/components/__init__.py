"""Dashboard components for The Earnings Hunter."""

from app.components.disclaimer import (
    show_disclaimer_modal,
    show_disclaimer_banner,
    show_disclaimer_footer,
    show_prediction_with_warning,
    show_inline_warning,
)

from app.components.market_status import (
    is_market_open,
    get_next_market_event,
    get_time_until_event,
    display_market_status,
    display_market_status_compact,
    get_market_session_info,
)

from app.components.index_cards import (
    fetch_market_indices,
    fetch_index_historical,
    create_sparkline,
    display_index_card,
    display_all_indices,
    display_indices_compact,
)

from app.components.stock_chart import (
    fetch_stock_data,
    fetch_stock_quote,
    fetch_company_profile,
    create_price_chart,
    display_stock_card,
    display_mini_chart,
)

from app.components.sentiment_gauge import (
    create_sentiment_gauge,
    display_sentiment_panel,
    display_mini_sentiment,
    create_dual_gauge,
    display_golden_triangle_sentiment,
)

__all__ = [
    # Disclaimer
    "show_disclaimer_modal",
    "show_disclaimer_banner",
    "show_disclaimer_footer",
    "show_prediction_with_warning",
    "show_inline_warning",
    # Market Status
    "is_market_open",
    "get_next_market_event",
    "get_time_until_event",
    "display_market_status",
    "display_market_status_compact",
    "get_market_session_info",
    # Index Cards
    "fetch_market_indices",
    "fetch_index_historical",
    "create_sparkline",
    "display_index_card",
    "display_all_indices",
    "display_indices_compact",
    # Stock Chart
    "fetch_stock_data",
    "fetch_stock_quote",
    "fetch_company_profile",
    "create_price_chart",
    "display_stock_card",
    "display_mini_chart",
    # Sentiment Gauge
    "create_sentiment_gauge",
    "display_sentiment_panel",
    "display_mini_sentiment",
    "create_dual_gauge",
    "display_golden_triangle_sentiment",
]
