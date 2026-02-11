"""
Stock Chart Component.

Interactive price chart with multiple timeframes and chart types.
Features candlestick and line charts with volume overlay.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Literal
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config.theme import COLORS, CARD_CSS, get_plotly_layout, format_number
from config.settings import get_settings

settings = get_settings()


# Timeframe configurations
TIMEFRAMES = {
    '1D': {'days': 1, 'label': '1D'},
    '1W': {'days': 7, 'label': '1W'},
    '1M': {'days': 30, 'label': '1M'},
    '3M': {'days': 90, 'label': '3M'},
    '1Y': {'days': 365, 'label': '1Y'},
}


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    Fetch historical stock data from FMP API.

    Args:
        symbol: Stock ticker symbol
        days: Number of days of history

    Returns:
        List of OHLCV data dicts
    """
    from src.data_ingestion.fmp_client import FMPClient

    try:
        client = FMPClient(api_key=settings.fmp_api_key)

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        prices = client.get_historical_prices(
            symbol,
            from_date=start_date,
            to_date=end_date
        )

        if prices:
            # Convert to list of dicts and sort by date
            data = [
                {
                    'date': p.date,
                    'open': p.open,
                    'high': p.high,
                    'low': p.low,
                    'close': p.close,
                    'volume': getattr(p, 'volume', 0) or 0
                }
                for p in sorted(prices, key=lambda x: x.date)
            ]
            return data

    except Exception as e:
        pass

    # Return mock data if API fails
    return get_mock_stock_data(symbol, days)


def get_mock_stock_data(symbol: str, days: int) -> List[Dict[str, Any]]:
    """Generate mock stock data for testing."""
    import random

    base_prices = {
        'AAPL': 182.0, 'MSFT': 405.0, 'GOOGL': 140.0,
        'AMZN': 175.0, 'NVDA': 720.0, 'TSLA': 245.0,
    }
    base = base_prices.get(symbol, 100.0)
    data = []

    for i in range(days):
        date = (datetime.now() - timedelta(days=days-i-1)).strftime('%Y-%m-%d')
        daily_change = random.uniform(-0.03, 0.03)
        base = base * (1 + daily_change)

        open_price = base * (1 + random.uniform(-0.01, 0.01))
        close_price = base * (1 + random.uniform(-0.01, 0.01))
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))

        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': random.randint(10000000, 50000000)
        })

    return data


@st.cache_data(ttl=60)
def fetch_stock_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch current stock quote from FMP API.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Quote data dict or None
    """
    from src.data_ingestion.fmp_client import FMPClient

    try:
        client = FMPClient(api_key=settings.fmp_api_key)
        quote = client.get_stock_quote(symbol)

        if quote:
            return {
                'price': quote.price,
                'change': quote.change,
                'change_percent': quote.changesPercentage,
                'previous_close': quote.previousClose,
                'open': quote.open,
                'high': quote.dayHigh,
                'low': quote.dayLow,
                'volume': quote.volume,
                'market_cap': quote.marketCap,
                'name': quote.name,
            }
    except Exception:
        pass

    return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_company_profile(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch company profile (for logo and name).

    Args:
        symbol: Stock ticker symbol

    Returns:
        Profile data dict or None
    """
    from src.data_ingestion.fmp_client import FMPClient

    try:
        client = FMPClient(api_key=settings.fmp_api_key)
        # Use quote for basic info since profile endpoint may differ
        quote = client.get_stock_quote(symbol)

        if quote:
            return {
                'name': quote.name,
                'symbol': symbol,
                'exchange': getattr(quote, 'exchange', 'NYSE'),
            }
    except Exception:
        pass

    return {'name': symbol, 'symbol': symbol, 'exchange': 'NYSE'}


def create_price_chart(
    data: List[Dict[str, Any]],
    chart_type: Literal['line', 'candlestick'] = 'line',
    show_volume: bool = True,
    height: int = 400
) -> go.Figure:
    """
    Create an interactive price chart.

    Args:
        data: List of OHLCV data dicts
        chart_type: 'line' or 'candlestick'
        show_volume: Whether to show volume bars
        height: Chart height in pixels

    Returns:
        Plotly figure
    """
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()

    dates = [d['date'] for d in data]
    closes = [d['close'] for d in data]

    # Determine if overall trend is positive
    if len(closes) >= 2:
        is_positive = closes[-1] >= closes[0]
    else:
        is_positive = True

    main_color = COLORS['accent_teal'] if is_positive else COLORS['accent_red']

    if chart_type == 'candlestick':
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=[d['open'] for d in data],
                high=[d['high'] for d in data],
                low=[d['low'] for d in data],
                close=closes,
                increasing_line_color=COLORS['accent_teal'],
                decreasing_line_color=COLORS['accent_red'],
                increasing_fillcolor=COLORS['accent_teal'],
                decreasing_fillcolor=COLORS['accent_red'],
                name='Price'
            ),
            row=1 if show_volume else None,
            col=1 if show_volume else None
        )
    else:
        # Line chart with gradient fill
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=closes,
                mode='lines',
                line=dict(
                    color=main_color,
                    width=2,
                    shape='spline',
                    smoothing=0.8
                ),
                fill='tozeroy',
                fillcolor=f"rgba({int(main_color[1:3], 16)}, {int(main_color[3:5], 16)}, {int(main_color[5:7], 16)}, 0.1)",
                name='Price',
                hovertemplate='%{y:$.2f}<extra></extra>'
            ),
            row=1 if show_volume else None,
            col=1 if show_volume else None
        )

    # Add volume bars
    if show_volume:
        volumes = [d['volume'] for d in data]
        colors = [
            COLORS['accent_teal'] if data[i]['close'] >= data[i]['open']
            else COLORS['accent_red']
            for i in range(len(data))
        ]

        fig.add_trace(
            go.Bar(
                x=dates,
                y=volumes,
                marker_color=colors,
                opacity=0.5,
                name='Volume',
                hovertemplate='Vol: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )

    # Update layout
    layout = get_plotly_layout(height=height)
    layout['showlegend'] = False
    layout['xaxis']['rangeslider'] = {'visible': False}

    if show_volume:
        layout['yaxis2'] = {
            'gridcolor': COLORS['chart_grid'],
            'tickfont': {'color': COLORS['text_secondary']},
            'showgrid': False
        }

    fig.update_layout(**layout)

    return fig


def display_stock_card(
    symbol: str,
    timeframe: str = '1M',
    chart_type: Literal['line', 'candlestick'] = 'line'
):
    """
    Display a complete stock card with price info and chart.

    Args:
        symbol: Stock ticker symbol
        timeframe: Timeframe key ('1D', '1W', '1M', '3M', '1Y')
        chart_type: Chart type ('line' or 'candlestick')
    """
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    # Fetch data
    quote = fetch_stock_quote(symbol)
    profile = fetch_company_profile(symbol)
    days = TIMEFRAMES.get(timeframe, TIMEFRAMES['1M'])['days']
    historical = fetch_stock_data(symbol, days)

    # Stock header
    if quote:
        price = quote.get('price', 0)
        change = quote.get('change', 0)
        change_pct = quote.get('change_percent', 0)
        name = profile.get('name', symbol) if profile else symbol

        color = COLORS['accent_teal'] if change >= 0 else COLORS['accent_red']
        sign = '+' if change >= 0 else ''

        header_html = f"""
        <div class="stock-card" style="margin-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                        <div style="
                            width: 40px; height: 40px;
                            background: {COLORS['bg_dark']};
                            border-radius: 8px;
                            display: flex; align-items: center; justify-content: center;
                            font-weight: bold; color: {COLORS['text_primary']};
                        ">{symbol[0]}</div>
                        <div>
                            <div style="color: {COLORS['text_primary']}; font-size: 18px; font-weight: 600;">
                                {name}
                            </div>
                            <div style="color: {COLORS['text_secondary']}; font-size: 14px;">
                                {symbol}
                            </div>
                        </div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: {COLORS['text_primary']}; font-size: 28px; font-weight: 700;">
                        ${price:,.2f}
                    </div>
                    <div style="color: {color}; font-size: 16px; font-weight: 600;">
                        {sign}{change:,.2f} ({sign}{change_pct:.2f}%)
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

    # Timeframe buttons
    cols = st.columns([1, 1, 1, 1, 1, 3])
    selected_tf = timeframe

    for i, (tf_key, tf_data) in enumerate(TIMEFRAMES.items()):
        with cols[i]:
            if st.button(
                tf_data['label'],
                key=f"tf_{symbol}_{tf_key}",
                use_container_width=True,
                type="primary" if tf_key == selected_tf else "secondary"
            ):
                selected_tf = tf_key
                st.rerun()

    # Price chart
    if historical:
        fig = create_price_chart(
            historical,
            chart_type=chart_type,
            show_volume=True,
            height=350
        )

        # Wrap chart in card styling
        st.markdown(f"""
        <div style="
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-top: none;
            border-radius: 0 0 12px 12px;
            padding: 16px;
            margin-top: -1px;
        ">
        """, unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Unable to load price data")


def display_mini_chart(symbol: str, days: int = 30, height: int = 100):
    """
    Display a minimal price chart without decorations.

    Args:
        symbol: Stock ticker symbol
        days: Number of days of history
        height: Chart height in pixels
    """
    data = fetch_stock_data(symbol, days)

    if data:
        fig = create_price_chart(data, chart_type='line', show_volume=False, height=height)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
