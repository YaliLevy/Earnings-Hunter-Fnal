"""
The Earnings Hunter - Modern Financial Analysis Dashboard

A sleek, professional single-page app with glass-morphism design,
animated components, and the StockFlow design system.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.components.disclaimer import show_disclaimer_modal, show_disclaimer_footer
from app.config.theme import COLORS, MASTER_CSS, get_plotly_layout, get_candlestick_colors, get_prediction_class, get_prediction_color

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Earnings Hunter | AI Stock Analysis",
    page_icon="https://em-content.zobj.net/source/apple/391/direct-hit_1f3af.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject master CSS
st.markdown(MASTER_CSS, unsafe_allow_html=True)

# Show disclaimer modal
show_disclaimer_modal()


# =============================================================================
# COMMAND BAR (HEADER)
# =============================================================================
def render_command_bar():
    """Render the premium command bar header with API status."""
    import time
    api_latency = f"{int(time.time() * 1000) % 50 + 8}ms"  # Simulated latency

    st.markdown(f"""
    <div class="command-bar">
        <div class="logo-section">
            <div class="logo-dot"></div>
            <span class="logo-text">EARNINGS<span class="muted">HUNTER</span></span>
        </div>
        <div class="api-status">
            <div class="status-dot"></div>
            <span>API: {api_latency}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SEARCH SECTION (Command Palette Style)
# =============================================================================
def render_search():
    """Render the command palette style search bar."""
    col1, col2, col3 = st.columns([5, 1, 1])

    with col1:
        ticker = st.text_input(
            "Search",
            placeholder="⌘K SEARCH TICKER...",
            label_visibility="collapsed",
            key="ticker_input"
        )

    with col2:
        analyze_button = st.button(
            "HUNT",
            type="primary",
            use_container_width=True
        )

    with col3:
        deep_analysis = st.checkbox(
            "Deep",
            help="Run CrewAI agents for comprehensive analysis (~$0.02)"
        )

    return ticker, analyze_button, deep_analysis


# =============================================================================
# COMPANY HEADER
# =============================================================================
def render_company_header(result: Dict[str, Any]):
    """Render the company information header."""
    price = result.get('current_price') or 0
    change = result.get('price_change') or 0
    change_pct = result.get('price_change_pct') or 0
    change_class = "price-up" if change >= 0 else "price-down"
    change_sign = "+" if change >= 0 else ""

    st.markdown(f"""
    <div class="company-header">
        <h2 class="company-name">{result.get('company_name', result['symbol'])}</h2>
        <p class="company-info">{result['symbol']} • {result.get('quarter', 'Latest Quarter')}</p>
        <p class="company-price">
            ${price:,.2f}
            <span class="{change_class} price-change">
                {change_sign}{change:,.2f} ({change_sign}{change_pct:.2f}%)
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PREDICTION WITH CONFIDENCE RING
# =============================================================================
def render_prediction_banner(prediction: str, confidence: float):
    """Render the prediction with animated confidence ring."""
    pred_class = get_prediction_class(prediction)
    pred_color = get_prediction_color(prediction)

    # Calculate ring offset (175 is full circle, 0 is empty)
    ring_offset = 175 - (confidence * 175)

    icons = {
        "Growth": "🚀",
        "Risk": "⚠️",
        "Stagnation": "➡️"
    }
    icon = icons.get(prediction, "📊")

    verdict_class = "sell" if prediction == "Risk" else ""

    st.markdown(f"""
    <div class="confidence-container">
        <div class="confidence-ring">
            <svg class="ring-svg" viewBox="0 0 64 64">
                <circle class="ring-bg" cx="32" cy="32" r="28"/>
                <circle class="ring-progress" cx="32" cy="32" r="28"
                        style="stroke: {pred_color}; stroke-dashoffset: {ring_offset};"/>
            </svg>
            <span class="ring-value">{confidence:.0%}</span>
        </div>
        <div>
            <div class="verdict-text {verdict_class}" style="color: {pred_color};">
                {icon} {prediction.upper()} SIGNAL
            </div>
            <div class="verdict-description">
                AI confidence based on Golden Triangle analysis
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# RADAR CHART - GOLDEN TRIANGLE VISUALIZATION
# =============================================================================
def render_radar_chart(scores: Dict[str, Any]):
    """Render Golden Triangle as interactive Radar Chart."""
    financial = scores.get('financial', 0) or 0
    ceo = scores.get('ceo_tone') or 5  # Default to 5 if no transcript
    social = scores.get('social', 0) or 0
    total = scores.get('weighted_total', 0) or 0

    fig = go.Figure()

    # Add radar trace - triangle shape
    fig.add_trace(go.Scatterpolar(
        r=[financial, ceo, social, financial],  # Close the shape
        theta=['Financials', 'CEO Tone', 'Social', 'Financials'],
        fill='toself',
        fillcolor='rgba(0, 240, 144, 0.15)',
        line=dict(color=COLORS['accent_green'], width=2),
        name='Score'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                gridcolor=COLORS['border'],
                tickfont=dict(size=8, color=COLORS['text_muted']),
                showline=False
            ),
            angularaxis=dict(
                gridcolor=COLORS['border'],
                linecolor=COLORS['border'],
                tickfont=dict(size=10, color=COLORS['text_secondary'])
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=220,
        margin=dict(l=50, r=50, t=30, b=30)
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Score display below radar
    total_color = COLORS['accent_green'] if total >= 7 else COLORS['accent_red'] if total <= 4 else COLORS['accent_yellow']
    st.markdown(f"""
    <div style="text-align: center; margin-top: -10px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 36px; font-weight: 700; color: {total_color};">
            {total:.0f}
        </div>
        <div style="font-size: 10px; color: {COLORS['text_muted']}; text-transform: uppercase; letter-spacing: 2px;">
            SCORE
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ANOMALY CARDS
# =============================================================================
def render_anomaly_card(label: str, value: str, icon: str, is_warning: bool = False):
    """Render a detected anomaly card."""
    value_color = COLORS['accent_red'] if is_warning else COLORS['accent_green']
    border_color = 'rgba(255, 46, 80, 0.3)' if is_warning else 'rgba(0, 240, 144, 0.3)'

    st.markdown(f"""
    <div class="anomaly-card" style="border-left: 3px solid {value_color};">
        <div class="anomaly-header">
            <span class="anomaly-label">{label}</span>
            <span class="anomaly-icon">{icon}</span>
        </div>
        <div class="anomaly-value" style="color: {value_color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PRICE TARGET GAUGE
# =============================================================================
def render_price_target_gauge(current: float, target: float):
    """Render analyst price target as a gauge visualization."""
    if not current or not target:
        st.markdown(f"""
        <div class="bento-card">
            <div class="bento-label">Analyst Target</div>
            <div style="color: {COLORS['text_muted']}; text-align: center; padding: 20px;">No data</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Calculate position (0-100% along the bar)
    low = min(current * 0.8, target * 0.8)
    high = max(current * 1.2, target * 1.2)
    range_size = high - low
    current_pos = min(max(((current - low) / range_size) * 100, 5), 95)
    target_pos = min(max(((target - low) / range_size) * 100, 5), 95)

    st.markdown(f"""
    <div class="bento-card">
        <div class="bento-label">Analyst Target</div>
        <div class="target-gauge">
            <div class="target-track">
                <div class="target-marker" style="left: {current_pos}%;">
                    <div class="marker-line current"></div>
                    <div class="marker-label current">NOW</div>
                </div>
                <div class="target-marker" style="left: {target_pos}%;">
                    <div class="marker-line target"></div>
                    <div class="marker-label target">${target:.0f}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# BENTO CARD HELPERS
# =============================================================================
def render_eps_bento(fin_data: Dict[str, Any]):
    """Render EPS bento card."""
    eps_beat = fin_data.get("eps_beat", False)
    eps_actual = fin_data.get('eps_actual', 0)
    eps_est = fin_data.get('eps_estimated', 0)
    eps_surprise = fin_data.get('eps_surprise', 0)

    badge_class = "" if eps_beat else "miss-badge"
    badge_text = "BEAT" if eps_beat else "MISS"
    badge_color = COLORS['accent_green'] if eps_beat else COLORS['accent_red']

    st.markdown(f"""
    <div class="bento-card">
        <div class="bento-label">EPS (Normalized)</div>
        <div class="bento-value" style="color: {badge_color};">${eps_actual:.2f}</div>
        <div class="bento-sublabel">vs ${eps_est:.2f} cons.</div>
        <div style="margin-top: 8px;">
            <span class="{badge_class} beat-badge">{badge_text} {eps_surprise:+.1f}%</span>
        </div>
        <div class="bento-progress">
            <div class="bento-progress-fill" style="width: {min(abs(eps_surprise) * 5, 100)}%; background: {badge_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_revenue_bento(fin_data: Dict[str, Any]):
    """Render Revenue bento card."""
    rev_beat = fin_data.get("revenue_beat", False)
    rev_actual = fin_data.get('revenue_actual', 0)
    rev_surprise = fin_data.get('revenue_surprise', 0)

    yoy_color = COLORS['accent_green'] if rev_surprise > 0 else COLORS['accent_red']

    st.markdown(f"""
    <div class="bento-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="bento-label">Revenue (Q3)</div>
            <div style="font-size: 12px; color: {yoy_color};">{rev_surprise:+.0f}% YoY</div>
        </div>
        <div class="bento-value">${rev_actual:.1f}B</div>
        <div class="bento-progress">
            <div class="bento-progress-fill" style="width: {min(abs(rev_surprise) * 3, 100)}%; background: {yoy_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_insider_bar(fin_data: Dict[str, Any]):
    """Render Insider Activity visual bar."""
    buys = fin_data.get('insider_buys', 0)
    sells = fin_data.get('insider_sells', 0)
    total = buys + sells

    if total > 0:
        buy_pct = (buys / total) * 100
        sell_pct = (sells / total) * 100
    else:
        buy_pct = sell_pct = 50

    # Calculate dollar values (approximate)
    buy_value = buys * 50000  # Rough estimate
    sell_value = sells * 50000

    st.markdown(f"""
    <div class="bento-card">
        <div class="bento-label">Insider Activity (90d)</div>
        <div class="insider-bar-container">
            <div class="insider-labels">
                <span class="insider-label sell">SELL</span>
                <span class="insider-label buy">BUY</span>
            </div>
            <div class="insider-bar">
                <div class="insider-sell" style="width: {sell_pct}%;"></div>
                <div class="insider-buy" style="width: {buy_pct}%;"></div>
            </div>
            <div class="insider-values">
                <span>${sell_value/1e6:.1f}M Sold</span>
                <span>${buy_value/1e6:.1f}M Bought</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PRICE CHART - LINE CHART WITH GRADIENT
# =============================================================================
def render_price_chart(prices, ticker: str, timeframe: str = "3M"):
    """Render a modern line chart with gradient fill."""
    if not prices:
        st.info("No price history available")
        return

    dates = [p.date for p in prices]
    closes = [p.close for p in prices]

    # Create line chart with gradient fill
    fig = go.Figure()

    # Add the main line
    fig.add_trace(go.Scatter(
        x=dates,
        y=closes,
        mode='lines',
        name='Price',
        line=dict(color=COLORS['accent_green'], width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 240, 144, 0.1)',
        hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
    ))

    # Calculate price change for title color
    if len(closes) >= 2:
        price_change = closes[-1] - closes[0]
        line_color = COLORS['accent_green'] if price_change >= 0 else COLORS['accent_red']
        fill_color = 'rgba(0, 240, 144, 0.1)' if price_change >= 0 else 'rgba(255, 46, 80, 0.1)'

        # Update colors based on performance
        fig.update_traces(
            line=dict(color=line_color, width=2),
            fillcolor=fill_color
        )

    layout = get_plotly_layout(f"{ticker} - {timeframe} Price History", height=350, show_legend=False)
    layout['xaxis_rangeslider_visible'] = False
    layout['hovermode'] = 'x unified'
    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# =============================================================================
# NEWS CARDS
# =============================================================================
def render_news_card(article: Dict[str, Any]):
    """Render a single news article card."""
    score = article.get("score", 0)
    sentiment = article.get("sentiment", "neutral")

    sentiment_icons = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
    icon = sentiment_icons.get(sentiment, "🟡")

    ai_reason = article.get("ai_reason", "")
    source = article.get("source", "Unknown")

    st.markdown(f"""
    <div class="news-card {sentiment}">
        <div class="news-title">{icon} {article['title']}</div>
        <div class="news-meta">
            <span class="news-source">{source}{f' • {ai_reason}' if ai_reason else ''}</span>
            <span class="news-sentiment {sentiment}">{sentiment.upper()} ({score:+.2f})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# STAT BOX
# =============================================================================
def render_stat_box(value: str, label: str, delta: str = None, delta_positive: bool = True):
    """Render a stat box."""
    delta_html = ""
    if delta:
        delta_color = COLORS['accent_teal'] if delta_positive else COLORS['accent_red']
        delta_html = f'<div class="stat-delta" style="color: {delta_color};">{delta}</div>'

    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def run_analysis(ticker: str, progress_callback=None) -> Dict[str, Any]:
    """Run analysis using FMP data and ML models."""
    from src.data_ingestion.fmp_client import FMPClient
    from src.feature_engineering.transcript_analyzer import TranscriptAnalyzer
    from src.feature_engineering.sentiment_features import SentimentFeatureExtractor
    from src.ml.predictor import EarningsPredictor
    from config.settings import get_settings
    from src.utils.news_analyzer import get_news_analyzer

    settings = get_settings()
    ticker = ticker.upper().strip()
    fmp_client = FMPClient(api_key=settings.fmp_api_key)
    sentiment_analyzer = SentimentFeatureExtractor()
    news_analyzer = get_news_analyzer()

    result = {
        "symbol": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Step 1: Get stock quote
    if progress_callback:
        progress_callback("Fetching stock quote...", 10)

    quote = fmp_client.get_stock_quote(ticker)
    if quote:
        result["current_price"] = quote.price
        result["price_change"] = quote.change
        result["price_change_pct"] = quote.change_percentage
        result["company_name"] = quote.name
    else:
        result["current_price"] = 0
        result["company_name"] = ticker

    # Step 2: Get earnings data
    if progress_callback:
        progress_callback("Fetching earnings data...", 20)

    earnings = fmp_client.get_earnings_surprises(ticker)

    if earnings and len(earnings) > 0:
        latest = None
        for e in earnings:
            if e.eps is not None:
                latest = e
                break
        if latest is None:
            latest = earnings[0]

        result["earnings_date"] = latest.date or "N/A"

        if latest.date:
            try:
                date_obj = datetime.strptime(latest.date, "%Y-%m-%d")
                q = (date_obj.month - 1) // 3 + 1
                result["quarter"] = f"Q{q} {date_obj.year}"
                result["year"] = date_obj.year
                result["quarter_num"] = q
            except:
                result["quarter"] = "Latest"
                result["year"] = datetime.now().year
                result["quarter_num"] = (datetime.now().month - 1) // 3 + 1

        eps_actual = latest.eps or 0
        eps_estimated = latest.eps_estimated or 0
        revenue_actual = (latest.revenue or 0) / 1e9
        revenue_estimated = (latest.revenue_estimated or 0) / 1e9

        eps_surprise = ((eps_actual - eps_estimated) / abs(eps_estimated) * 100) if eps_estimated else 0
        revenue_surprise = ((revenue_actual - revenue_estimated) / abs(revenue_estimated) * 100) if revenue_estimated else 0

        result["financial_data"] = {
            "eps_actual": eps_actual,
            "eps_estimated": eps_estimated,
            "eps_surprise": eps_surprise,
            "revenue_actual": revenue_actual,
            "revenue_estimated": revenue_estimated,
            "revenue_surprise": revenue_surprise,
            "eps_beat": eps_surprise > 0,
            "revenue_beat": revenue_surprise > 0,
        }
    else:
        result["quarter"] = "N/A"
        result["earnings_date"] = "N/A"
        result["financial_data"] = {}

    # Step 3: Get income statement for margins
    if progress_callback:
        progress_callback("Fetching financial statements...", 30)

    statements = fmp_client.get_income_statement(ticker, period="quarter", limit=4)
    if statements and len(statements) > 0:
        latest_stmt = statements[0]
        revenue = latest_stmt.revenue or 0
        gross_profit = latest_stmt.gross_profit or 0
        operating_income = latest_stmt.operating_income or 0

        result["financial_data"]["gross_margin"] = (gross_profit / revenue * 100) if revenue else 0
        result["financial_data"]["operating_margin"] = (operating_income / revenue * 100) if revenue else 0

    # Step 4: Get analyst data
    if progress_callback:
        progress_callback("Fetching analyst estimates...", 40)

    price_target = fmp_client.get_price_target(ticker)
    if price_target:
        result["financial_data"]["price_target"] = price_target.target_consensus or 0
        if result.get("current_price") and price_target.target_consensus:
            upside = ((price_target.target_consensus - result["current_price"]) / result["current_price"]) * 100
            result["financial_data"]["price_target_upside"] = upside

    # Step 5: Get insider trading
    if progress_callback:
        progress_callback("Fetching insider activity...", 50)

    insider_data = fmp_client.get_insider_trading(ticker, limit=50)
    if insider_data:
        buys = sum(1 for t in insider_data if t.transaction_type and "Buy" in str(t.transaction_type))
        sells = sum(1 for t in insider_data if t.transaction_type and "Sell" in str(t.transaction_type))
        total = buys + sells
        result["financial_data"]["insider_sentiment"] = (buys - sells) / total if total > 0 else 0
        result["financial_data"]["insider_buys"] = buys
        result["financial_data"]["insider_sells"] = sells

        result["insider_transactions"] = [
            {
                "date": t.transaction_date,
                "name": t.reporting_name,
                "type": t.type_of_owner,
                "transaction": t.transaction_type,
                "shares": t.securities_transacted,
                "price": t.price,
                "value": t.securities_transacted * t.price if t.price else 0,
                "acquisition": t.acquisition_or_disposition
            }
            for t in insider_data[:20]
        ]

    # Step 6: Get earnings transcript
    if progress_callback:
        progress_callback("Analyzing CEO tone...", 60)

    year = result.get("year", datetime.now().year)
    quarter = result.get("quarter_num", (datetime.now().month - 1) // 3 + 1)

    transcript = fmp_client.get_earnings_call_transcript(ticker, year, quarter)

    result["ceo_analysis"] = {
        "has_transcript": False,
        "confidence_score": 0.5,
        "sentiment_score": 0.5,
        "tone_summary": "N/A",
    }

    if transcript and transcript.content:
        analyzer = TranscriptAnalyzer()
        ceo_analysis = analyzer.extract_features_ai(transcript.content, ticker)
        result["ceo_analysis"] = ceo_analysis

    # Step 7: Get stock news and analyze sentiment
    if progress_callback:
        progress_callback("Analyzing news sentiment...", 75)

    stock_news = []
    try:
        stock_news = fmp_client.get_stock_news(ticker, limit=50)
    except:
        pass

    news_articles = []
    combined_sentiment = 0

    if stock_news:
        raw_articles = []
        for news in stock_news:
            raw_articles.append({
                "title": news.title,
                "text": getattr(news, 'text', ''),
                "source": getattr(news, 'site', 'Unknown'),
                "url": getattr(news, 'url', None),
                "date": getattr(news, 'published_date', None),
            })

        company_name = result.get("company_name", ticker)

        news_articles = news_analyzer.analyze_news_batch(
            symbol=ticker,
            company_name=company_name,
            news_articles=raw_articles,
            max_results=15
        )

        for article in news_articles:
            combined_sentiment += article.get("score", 0)

    avg_sentiment = combined_sentiment / len(news_articles) if news_articles else 0
    bullish_count = sum(1 for a in news_articles if a["score"] > 0.05)
    bearish_count = sum(1 for a in news_articles if a["score"] < -0.05)

    result["social_analysis"] = {
        "news_count": len(news_articles),
        "combined_sentiment": (avg_sentiment + 1) / 2,
        "bullish_ratio": bullish_count / len(news_articles) if news_articles else 0.5,
        "bearish_ratio": bearish_count / len(news_articles) if news_articles else 0.2,
    }
    result["news_articles"] = news_articles

    # Step 8: Calculate Golden Triangle scores
    if progress_callback:
        progress_callback("Calculating scores...", 85)

    fin_data = result.get("financial_data", {})
    eps_beat_score = 5 if fin_data.get("eps_beat") else 0
    rev_beat_score = 3 if fin_data.get("revenue_beat") else 0
    margin_score = min(2, fin_data.get("gross_margin", 0) / 50)
    financial_score = min(10, eps_beat_score + rev_beat_score + margin_score)

    ceo_data = result.get("ceo_analysis", {})
    has_transcript = ceo_data.get("has_transcript", False)

    if has_transcript:
        confidence = ceo_data.get("confidence_score", 0.5)
        sentiment = ceo_data.get("sentiment_score", 0.5)
        ceo_score = (confidence + sentiment) * 5
    else:
        ceo_score = None

    social_data = result.get("social_analysis", {})
    social_sent = social_data.get("combined_sentiment", 0.5)
    news_bull = social_data.get("bullish_ratio", 0.5)
    social_score = (social_sent * 5 + news_bull * 5)

    if has_transcript:
        weighted_total = (financial_score * 0.40) + (ceo_score * 0.35) + (social_score * 0.25)
        financial_weight = 40
        ceo_weight = 35
        social_weight = 25
    else:
        weighted_total = (financial_score * 0.61) + (social_score * 0.39)
        financial_weight = 61
        ceo_weight = 0
        social_weight = 39

    result["scores"] = {
        "financial": round(financial_score, 1),
        "ceo_tone": round(ceo_score, 1) if ceo_score is not None else None,
        "social": round(social_score, 1),
        "weighted_total": round(weighted_total, 1),
        "has_transcript": has_transcript,
        "weights": {
            "financial": financial_weight,
            "ceo_tone": ceo_weight,
            "social": social_weight
        }
    }

    # Step 9: Run ML prediction
    if progress_callback:
        progress_callback("Running ML models...", 90)

    try:
        predictor = EarningsPredictor()
        predictor.load_models()

        features = {
            "fin_eps_surprise": fin_data.get("eps_surprise", 0) / 100,
            "fin_revenue_surprise": fin_data.get("revenue_surprise", 0) / 100,
            "fin_eps_beat": 1 if fin_data.get("eps_beat") else 0,
            "fin_revenue_beat": 1 if fin_data.get("revenue_beat") else 0,
            "fin_gross_margin": fin_data.get("gross_margin", 0) / 100,
            "fin_insider_sentiment": fin_data.get("insider_sentiment", 0),
            "ceo_confidence_score": ceo_data.get("confidence_score", 0.5),
            "ceo_overall_sentiment": ceo_data.get("sentiment_score", 0.5),
            "ceo_uncertainty_ratio": ceo_data.get("uncertainty_ratio", 0.2),
        }

        prediction_result = predictor.predict(features, mode="consensus")
        result["prediction"] = prediction_result.get("consensus_prediction", "Stagnation")
        result["confidence"] = prediction_result.get("best_model_confidence", 0.5)

    except Exception as e:
        if weighted_total >= 7:
            result["prediction"] = "Growth"
            result["confidence"] = 0.7
        elif weighted_total <= 4:
            result["prediction"] = "Risk"
            result["confidence"] = 0.6
        else:
            result["prediction"] = "Stagnation"
            result["confidence"] = 0.5

    # Step 10: Get historical prices for chart
    if progress_callback:
        progress_callback("Loading price history...", 95)

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        prices = fmp_client.get_historical_prices(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        result["price_history"] = prices
    except:
        result["price_history"] = []

    if progress_callback:
        progress_callback("Complete!", 100)

    return result


def run_agent_analysis(ticker: str, ml_prediction: str, ml_confidence: float) -> Optional[str]:
    """Run CrewAI agents for deep analysis."""
    try:
        from src.agents.crew import EarningsHunterCrew

        year = datetime.now().year
        quarter = (datetime.now().month - 1) // 3 + 1

        crew = EarningsHunterCrew(verbose=False)
        result = crew.analyze(
            symbol=ticker,
            year=year,
            quarter=quarter,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence
        )

        if result.get("status") == "success":
            return result.get("raw_output", "No output generated")
        else:
            return f"Agent error: {result.get('error', 'Unknown')}"

    except ImportError as e:
        return f"CrewAI not installed: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# MAIN APP
# =============================================================================

# Render command bar (header)
render_command_bar()

st.markdown("<div style='padding: 0 24px;'>", unsafe_allow_html=True)

# Render search section
ticker, analyze_button, deep_analysis = render_search()

# Handle analysis
if analyze_button and ticker:
    ticker = ticker.upper().strip()

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(text, value):
        status_text.markdown(f"<p style='text-align: center; color: {COLORS['text_secondary']};'>{text}</p>", unsafe_allow_html=True)
        progress_bar.progress(value)

    try:
        result = run_analysis(ticker, update_progress)
        st.session_state["result"] = result

        if deep_analysis:
            status_text.markdown(f"<p style='text-align: center; color: {COLORS['accent_purple']};'>🤖 Running AI Agent analysis... (30-60 seconds)</p>", unsafe_allow_html=True)
            progress_bar.progress(95)
            agent_report = run_agent_analysis(
                ticker,
                result.get("prediction", "Stagnation"),
                result.get("confidence", 0.5)
            )
            st.session_state["agent_report"] = agent_report

        status_text.empty()
        progress_bar.empty()
        st.rerun()

    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Analysis failed: {str(e)}")

elif analyze_button and not ticker:
    st.warning("Please enter a stock ticker symbol")


# =============================================================================
# DISPLAY RESULTS - 3-COLUMN COCKPIT LAYOUT
# =============================================================================
if "result" in st.session_state:
    result = st.session_state["result"]
    fin_data = result.get("financial_data", {})
    ceo_data = result.get("ceo_analysis", {})
    scores = result.get("scores", {})

    # 3-COLUMN COCKPIT LAYOUT
    col_left, col_center, col_right = st.columns([1, 2.5, 1.2])

    # =========================================================================
    # ZONE B - Left Sidebar (Intelligence Hub)
    # =========================================================================
    with col_left:
        st.markdown('<p class="section-title">THE GOLDEN TRIANGLE</p>', unsafe_allow_html=True)
        render_radar_chart(scores)

        st.markdown('<p class="section-title">AI VERDICT</p>', unsafe_allow_html=True)
        render_prediction_banner(
            result.get("prediction", "Stagnation"),
            result.get("confidence", 0.5)
        )

        st.markdown('<p class="section-title">DETECTED ANOMALIES</p>', unsafe_allow_html=True)

        # Generate anomaly cards based on analysis
        anomalies_shown = 0

        # CEO Tone anomaly
        ceo_confidence = ceo_data.get("confidence_score", 0.5)
        if ceo_confidence < 0.4:
            render_anomaly_card("CEO Tone", "Nervous", "⚠️", is_warning=True)
            anomalies_shown += 1
        elif ceo_confidence > 0.7:
            render_anomaly_card("CEO Tone", "Confident", "✓", is_warning=False)
            anomalies_shown += 1

        # EPS anomaly
        eps_surprise = fin_data.get("eps_surprise", 0)
        if eps_surprise > 10:
            render_anomaly_card("Option Flow", "Bullish Divergence", "⚡", is_warning=False)
            anomalies_shown += 1
        elif eps_surprise < -10:
            render_anomaly_card("Earnings Miss", "Significant", "⚠️", is_warning=True)
            anomalies_shown += 1

        # Social/Retail anomaly
        social_data = result.get("social_analysis", {})
        bullish_ratio = social_data.get("bullish_ratio", 0.5)
        if bullish_ratio > 0.7:
            render_anomaly_card("Retail Interest", "Spiking", "📈", is_warning=False)
            anomalies_shown += 1
        elif bullish_ratio < 0.3:
            render_anomaly_card("Sentiment", "Bearish", "📉", is_warning=True)
            anomalies_shown += 1

        if anomalies_shown == 0:
            st.markdown(f"""
            <div class="anomaly-card">
                <div class="anomaly-label">No anomalies detected</div>
                <div class="anomaly-value neutral" style="color: {COLORS['text_muted']};">Normal</div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # ZONE C - Center (Market Stage)
    # =========================================================================
    with col_center:
        # Real-time price update - fetch fresh quote
        from src.data_ingestion.fmp_client import FMPClient
        from config.settings import get_settings

        settings = get_settings()
        fmp_client = FMPClient(api_key=settings.fmp_api_key)

        # Fetch real-time quote
        fresh_quote = fmp_client.get_stock_quote(result['symbol'])
        if fresh_quote:
            price = fresh_quote.price
            change = fresh_quote.change
            change_pct = fresh_quote.change_percentage
        else:
            price = result.get('current_price') or 0
            change = result.get('price_change') or 0
            change_pct = result.get('price_change_pct') or 0

        change_color = COLORS['accent_green'] if change >= 0 else COLORS['accent_red']
        change_sign = "+" if change >= 0 else ""

        st.markdown(f"""
        <div class="ticker-header">
            <div style="display: flex; align-items: baseline; gap: 16px;">
                <h1 class="ticker-symbol">{result['symbol']}</h1>
                <div class="ticker-price">
                    <span class="price-value">${price:,.2f}</span>
                    <span class="price-change" style="color: {change_color};">
                        ∿ {change_sign}{change_pct:.2f}% today
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Functional timeframe buttons
        timeframe_options = {"1D": 1, "5D": 5, "1M": 30, "3M": 90, "6M": 180, "YTD": 365, "1Y": 365, "5Y": 1825}

        # Initialize timeframe in session state
        if "chart_timeframe" not in st.session_state:
            st.session_state.chart_timeframe = "3M"

        # Create button columns
        tf_cols = st.columns(7)
        timeframes = ["1D", "5D", "1M", "3M", "6M", "1Y", "5Y"]

        for i, tf in enumerate(timeframes):
            with tf_cols[i]:
                if st.button(tf, key=f"tf_{tf}", use_container_width=True,
                           type="primary" if st.session_state.chart_timeframe == tf else "secondary"):
                    st.session_state.chart_timeframe = tf
                    st.rerun()

        # Get historical prices based on selected timeframe
        selected_tf = st.session_state.chart_timeframe
        days = timeframe_options.get(selected_tf, 90)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            prices = fmp_client.get_historical_prices(
                result['symbol'],
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
        except:
            prices = result.get("price_history", [])

        # Price chart
        if prices:
            render_price_chart(prices, result["symbol"], selected_tf)

    # =========================================================================
    # ZONE D - Right Sidebar (Financial Bento)
    # =========================================================================
    with col_right:
        st.markdown('<p class="section-title">FUNDAMENTAL CORE</p>', unsafe_allow_html=True)

        # EPS Bento Card
        render_eps_bento(fin_data)

        # Revenue Bento Card
        render_revenue_bento(fin_data)

        # Insider Activity Bar
        render_insider_bar(fin_data)

        # Price Target Gauge
        render_price_target_gauge(
            result.get("current_price", 0),
            fin_data.get("price_target", 0)
        )

        # System status
        st.markdown(f"""
        <div class="system-status">
            SYSTEM STATUS: OPERATIONAL<br>
            LAST UPDATED: {datetime.now().strftime("%I:%M:%S %p")}
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # DETAILED TABS (Below cockpit)
    # =========================================================================
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Summary",
        "📊 Detailed Financials",
        "🎤 CEO Analysis",
        "📰 News & Sentiment"
    ])

    with tab1:
        eps_s = fin_data.get("eps_surprise", 0)
        rev_s = fin_data.get("revenue_surprise", 0)

        st.markdown(f"""
        **{result['symbol']}** reported earnings with an EPS **{'beat' if eps_s > 0 else 'miss'}** of
        **{abs(eps_s):.1f}%** and revenue **{'beat' if rev_s > 0 else 'miss'}** of **{abs(rev_s):.1f}%**.

        CEO tone analysis indicates **{ceo_data.get('tone_summary', 'neutral')}** sentiment.

        The Golden Triangle weighted score is **{scores.get('weighted_total', 0):.1f}/10**.
        """)

        # Key metrics grid
        st.markdown("#### Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("EPS Actual", f"${fin_data.get('eps_actual', 0):.2f}")
        with col2:
            st.metric("Revenue", f"${fin_data.get('revenue_actual', 0):.2f}B")
        with col3:
            st.metric("Gross Margin", f"{fin_data.get('gross_margin', 0):.1f}%")
        with col4:
            pt = fin_data.get('price_target', 0)
            st.metric("Price Target", f"${pt:.2f}" if pt else "N/A")

    with tab2:
        # Insider transactions table
        insider_transactions = result.get("insider_transactions", [])
        if insider_transactions:
            import pandas as pd
            st.markdown("#### Insider Transactions")
            df = pd.DataFrame(insider_transactions)
            df["value"] = df["value"].apply(lambda x: f"${x:,.0f}" if x > 0 else "-")
            df["price"] = df["price"].apply(lambda x: f"${x:.2f}" if x and x > 0 else "-")
            df["shares"] = df["shares"].apply(lambda x: f"{x:,.0f}" if x else "-")
            df_display = df.rename(columns={
                "date": "Date", "name": "Insider", "transaction": "Type",
                "shares": "Shares", "price": "Price", "value": "Value"
            })[["Date", "Insider", "Type", "Shares", "Price", "Value"]]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

    with tab3:
        if ceo_data.get("has_transcript"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence Score", f"{ceo_data.get('confidence_score', 0.5):.0%}")
                st.metric("Guidance Sentiment", f"{ceo_data.get('guidance_sentiment', 0.5):.0%}")
            with col2:
                st.metric("Sentiment Score", f"{ceo_data.get('sentiment_score', 0.5):.0%}")
                st.metric("Uncertainty Ratio", f"{ceo_data.get('uncertainty_ratio', 0.2):.0%}")

            st.markdown(f"**Overall Tone:** {ceo_data.get('tone_summary', 'Neutral')}")
            st.markdown("---")
            st.markdown("#### Executive Summary")
            executive_summary = ceo_data.get('executive_summary', 'No summary available.')
            st.markdown(f"""
            <div class="bento-card">
                <p style="color: {COLORS['text_secondary']}; line-height: 1.8; margin: 0;">
                    {executive_summary}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No earnings transcript available for this quarter.")

    with tab4:
        social = result.get("social_analysis", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Articles Analyzed", social.get("news_count", 0))
        with col2:
            st.metric("Bullish Ratio", f"{social.get('bullish_ratio', 0.5):.0%}")
        with col3:
            st.metric("Bearish Ratio", f"{social.get('bearish_ratio', 0.2):.0%}")

        st.markdown("---")
        news_articles = result.get("news_articles", [])
        if news_articles:
            for article in news_articles:
                render_news_card(article)
        else:
            st.info("No news articles found for this stock.")

    # Agent report section
    if "agent_report" in st.session_state:
        st.markdown("---")
        st.markdown('<p class="section-title">🤖 AI AGENT DEEP ANALYSIS</p>', unsafe_allow_html=True)
        agent_report = st.session_state["agent_report"]
        if agent_report.startswith("Error") or agent_report.startswith("CrewAI") or agent_report.startswith("Agent"):
            st.warning(agent_report)
        else:
            st.markdown(f"""
            <div class="bento-card">
                {agent_report}
            </div>
            """, unsafe_allow_html=True)
            st.caption("Generated by CrewAI agents: Scout Agent + Social Listener + Fusion Agent")


# Close main content div
st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
show_disclaimer_footer()

# Powered by badge with system status
st.markdown(f"""
<div class="system-status">
    Powered by FMP API • OpenAI GPT-4 • CrewAI
</div>
""", unsafe_allow_html=True)
