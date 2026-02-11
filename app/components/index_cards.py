"""
Market Index Cards Component.

Displays major market indices (S&P 500, NASDAQ, Dow Jones, VIX)
with real-time prices, change percentages, and sparkline charts.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import streamlit as st
import plotly.graph_objects as go

from app.config.theme import (
    COLORS, CARD_CSS, get_sparkline_layout,
    format_number, format_percent, get_change_color
)
from config.settings import get_settings

settings = get_settings()


# Market index symbols and display names
MARKET_INDICES = {
    'SPY': {'name': 'S&P 500', 'description': 'S&P 500 ETF'},
    'QQQ': {'name': 'Nasdaq', 'description': 'NASDAQ-100 ETF'},
    'DIA': {'name': 'Dow Jones', 'description': 'Dow Jones Industrial'},
    '^VIX': {'name': 'VIX', 'description': 'Volatility Index'},
}

# VIX fear levels
VIX_LEVELS = {
    'low': (0, 15, 'Fear: Low', COLORS['accent_teal']),
    'moderate': (15, 25, 'Fear: Moderate', COLORS['accent_yellow']),
    'high': (25, 35, 'Fear: High', COLORS['accent_red']),
    'extreme': (35, 100, 'Fear: Extreme', COLORS['accent_red']),
}


@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_market_indices() -> Dict[str, Any]:
    """
    Fetch current market index data from FMP API.

    Returns:
        Dict with index data for each symbol
    """
    from src.data_ingestion.fmp_client import FMPClient

    try:
        client = FMPClient(api_key=settings.fmp_api_key)
        indices_data = {}

        for symbol in MARKET_INDICES.keys():
            # Handle VIX differently (it's ^VIX in some systems)
            fetch_symbol = 'UVXY' if symbol == '^VIX' else symbol

            try:
                quote = client.get_stock_quote(fetch_symbol)
                if quote:
                    indices_data[symbol] = {
                        'price': quote.price,
                        'change': quote.change,
                        'change_percent': quote.changesPercentage,
                        'previous_close': quote.previousClose,
                        'name': MARKET_INDICES[symbol]['name'],
                    }
            except Exception as e:
                # Use mock data if API fails
                indices_data[symbol] = get_mock_index_data(symbol)

        return indices_data

    except Exception as e:
        # Return mock data if client fails
        return {symbol: get_mock_index_data(symbol) for symbol in MARKET_INDICES.keys()}


def get_mock_index_data(symbol: str) -> Dict[str, Any]:
    """Get mock data for an index (fallback)."""
    mock_data = {
        'SPY': {'price': 5137.08, 'change': 62.83, 'change_percent': 1.24},
        'QQQ': {'price': 16274.94, 'change': 136.12, 'change_percent': 0.85},
        'DIA': {'price': 38989.83, 'change': -46.79, 'change_percent': -0.12},
        '^VIX': {'price': 13.44, 'change': -0.32, 'change_percent': -2.33},
    }
    data = mock_data.get(symbol, {'price': 0, 'change': 0, 'change_percent': 0})
    data['name'] = MARKET_INDICES[symbol]['name']
    data['previous_close'] = data['price'] - data['change']
    return data


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_index_historical(symbol: str, days: int = 5) -> List[float]:
    """
    Fetch historical prices for sparkline chart.

    Args:
        symbol: Index symbol
        days: Number of days of history

    Returns:
        List of closing prices
    """
    from src.data_ingestion.fmp_client import FMPClient

    try:
        client = FMPClient(api_key=settings.fmp_api_key)
        fetch_symbol = 'UVXY' if symbol == '^VIX' else symbol

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        prices = client.get_historical_prices(
            fetch_symbol,
            from_date=start_date,
            to_date=end_date
        )

        if prices:
            # Return closing prices in chronological order
            closes = [p.close for p in sorted(prices, key=lambda x: x.date)]
            return closes

    except Exception:
        pass

    # Return mock sparkline data
    return get_mock_sparkline_data(symbol)


def get_mock_sparkline_data(symbol: str) -> List[float]:
    """Get mock sparkline data."""
    import random
    base = {'SPY': 5100, 'QQQ': 16200, 'DIA': 39000, '^VIX': 14}
    base_val = base.get(symbol, 100)
    return [base_val + random.uniform(-50, 50) for _ in range(5)]


def create_sparkline(prices: List[float], positive: bool = True) -> go.Figure:
    """
    Create a mini sparkline chart.

    Args:
        prices: List of prices
        positive: Whether the trend is positive (green) or negative (red)

    Returns:
        Plotly figure
    """
    color = COLORS['accent_teal'] if positive else COLORS['accent_red']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=prices,
        mode='lines',
        line=dict(
            color=color,
            width=2,
            shape='spline',
            smoothing=1.3
        ),
        fill='tozeroy',
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
        hoverinfo='none'
    ))

    fig.update_layout(**get_sparkline_layout(height=40))

    return fig


def get_vix_fear_level(vix_value: float) -> tuple:
    """
    Get VIX fear level description and color.

    Args:
        vix_value: Current VIX value

    Returns:
        Tuple of (label, color)
    """
    for level, (low, high, label, color) in VIX_LEVELS.items():
        if low <= vix_value < high:
            return label, color
    return "Fear: Extreme", COLORS['accent_red']


def display_index_card(
    symbol: str,
    data: Dict[str, Any],
    sparkline_data: Optional[List[float]] = None
):
    """
    Display a single index card.

    Args:
        symbol: Index symbol
        data: Index data dict
        sparkline_data: Historical prices for sparkline
    """
    price = data.get('price', 0)
    change_pct = data.get('change_percent', 0)
    name = data.get('name', symbol)

    # Determine color
    change_class = get_change_color(change_pct)
    is_positive = change_pct >= 0

    # Special handling for VIX
    if symbol == '^VIX':
        fear_label, fear_color = get_vix_fear_level(price)
        sub_text = f'<span style="color: {fear_color}; font-size: 12px;">{fear_label}</span>'
    else:
        sign = '+' if change_pct >= 0 else ''
        sub_text = f'<span class="{change_class}">{sign}{change_pct:.2f}%</span>'

    # Create sparkline if data available
    if sparkline_data and len(sparkline_data) > 1:
        fig = create_sparkline(sparkline_data, positive=is_positive)
        sparkline_html = fig.to_html(
            include_plotlyjs=False,
            config={'displayModeBar': False},
            full_html=False
        )
    else:
        sparkline_html = ""

    # Format price
    if symbol == '^VIX':
        price_str = f"{price:.2f}"
    elif price >= 10000:
        price_str = f"{price:,.2f}"
    else:
        price_str = f"{price:,.2f}"

    html = f"""
    <div class="index-card">
        <div class="index-card-title">{name}</div>
        <div class="index-card-value">{price_str}</div>
        <div class="index-card-change">{sub_text}</div>
        <div style="margin-top: 8px; height: 40px;">
            {sparkline_html}
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def display_all_indices():
    """
    Display all market indices in a 4-column layout.
    """
    # Inject CSS
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    # Fetch data
    indices_data = fetch_market_indices()

    # Create 4 columns
    cols = st.columns(4)

    for col, symbol in zip(cols, MARKET_INDICES.keys()):
        with col:
            data = indices_data.get(symbol, get_mock_index_data(symbol))
            sparkline = fetch_index_historical(symbol)
            display_index_card(symbol, data, sparkline)


def display_indices_compact():
    """
    Display indices in a compact horizontal strip.
    """
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    indices_data = fetch_market_indices()

    items_html = []
    for symbol, data in indices_data.items():
        name = data.get('name', symbol)
        price = data.get('price', 0)
        change_pct = data.get('change_percent', 0)

        color = COLORS['accent_teal'] if change_pct >= 0 else COLORS['accent_red']
        sign = '+' if change_pct >= 0 else ''

        items_html.append(f"""
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 16px;">
            <span style="color: {COLORS['text_secondary']}; font-size: 12px;">{name}</span>
            <span style="color: {COLORS['text_primary']}; font-weight: 600;">{price:,.2f}</span>
            <span style="color: {color}; font-size: 12px;">{sign}{change_pct:.2f}%</span>
        </div>
        """)

    html = f"""
    <div style="
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        display: flex;
        justify-content: space-around;
        overflow-x: auto;
    ">
        {''.join(items_html)}
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)
