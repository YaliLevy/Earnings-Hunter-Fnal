"""
The Earnings Hunter - Institutional Cybernetics Dashboard
Bloomberg Terminal density meets Linear's refined minimalism.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.components.disclaimer import show_disclaimer_modal, show_disclaimer_footer
from app.config.theme import COLORS, MASTER_CSS, get_plotly_layout, get_prediction_class, get_prediction_color

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

# Additional CSS for exact design match
st.markdown(f"""
<style>
    /* Full viewport cockpit - no scrolling on main frame */
    .main .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
        min-height: 100vh;
    }}

    /* Command Bar - Zone A */
    .command-bar-v2 {{
        height: 56px;
        background: rgba(5, 5, 5, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid {COLORS['border']};
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        position: sticky;
        top: 0;
        z-index: 1000;
    }}

    .logo-v2 {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .logo-icon {{
        font-size: 20px;
    }}

    .logo-text-v2 {{
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1px;
        color: {COLORS['text_primary']};
    }}

    .search-container {{
        flex: 1;
        max-width: 600px;
        margin: 0 32px;
    }}

    .search-input {{
        width: 100%;
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 10px 16px;
        color: {COLORS['text_primary']};
        font-size: 14px;
        text-align: center;
    }}

    .search-input::placeholder {{
        color: {COLORS['text_muted']};
    }}

    .right-controls {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}

    .deep-reasoning-toggle {{
        display: flex;
        align-items: center;
        gap: 8px;
        color: {COLORS['text_muted']};
        font-size: 12px;
        font-weight: 500;
    }}

    .live-indicator {{
        display: flex;
        align-items: center;
        gap: 6px;
        color: {COLORS['accent_green']};
        font-size: 12px;
        font-weight: 600;
    }}

    .live-dot {{
        width: 8px;
        height: 8px;
        background: {COLORS['accent_green']};
        border-radius: 50%;
        animation: pulse 2s infinite;
    }}

    /* Ticker Info Strip */
    .ticker-strip {{
        background: {COLORS['bg_dark']};
        border-bottom: 1px solid {COLORS['border']};
        padding: 12px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .ticker-left {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}

    .company-name {{
        font-size: 14px;
        font-weight: 500;
        color: {COLORS['text_primary']};
    }}

    .ticker-price-lg {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}

    .ticker-change {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 500;
    }}

    .ticker-right {{
        display: flex;
        gap: 24px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: {COLORS['text_muted']};
    }}

    /* Main Cockpit Layout */
    .cockpit-layout {{
        display: flex;
        min-height: calc(100vh - 110px);
    }}

    /* Zone B - Intelligence Hub */
    .zone-b {{
        width: 260px;
        min-width: 260px;
        background: {COLORS['bg_dark']};
        border-right: 1px solid {COLORS['border']};
        padding: 20px;
        display: flex;
        flex-direction: column;
    }}

    .zone-title {{
        font-size: 10px;
        font-weight: 700;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 16px;
    }}

    /* SVG Confidence Ring */
    .confidence-ring-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 20px 0;
    }}

    .confidence-ring-svg {{
        width: 140px;
        height: 140px;
        transform: rotate(-90deg);
    }}

    .ring-track {{
        fill: none;
        stroke: {COLORS['border']};
        stroke-width: 8;
    }}

    .ring-fill {{
        fill: none;
        stroke: {COLORS['accent_green']};
        stroke-width: 8;
        stroke-linecap: round;
        stroke-dasharray: 377;
        transition: stroke-dashoffset 1s ease-out;
    }}

    .ring-center {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
    }}

    .ring-percentage {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px;
        font-weight: 700;
        color: {COLORS['accent_green']};
    }}

    .ring-label {{
        font-size: 10px;
        font-weight: 600;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }}

    .verdict-badge {{
        text-align: center;
        margin-top: 12px;
    }}

    .verdict-text {{
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 1px;
    }}

    .verdict-bullish {{ color: {COLORS['accent_green']}; }}
    .verdict-bearish {{ color: {COLORS['accent_red']}; }}
    .verdict-neutral {{ color: {COLORS['accent_yellow']}; }}

    /* AI Analysis Block */
    .ai-analysis-section {{
        margin-top: 24px;
    }}

    .ai-analysis-title {{
        font-size: 10px;
        font-weight: 700;
        color: {COLORS['accent_red']};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }}

    .ai-analysis-text {{
        font-size: 13px;
        color: {COLORS['text_secondary']};
        line-height: 1.6;
    }}

    /* Zone C - Market Stage */
    .zone-c {{
        flex: 1;
        background: {COLORS['bg_dark']};
        padding: 20px 24px;
        display: flex;
        flex-direction: column;
    }}

    .chart-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }}

    .chart-title {{
        font-size: 14px;
        font-weight: 500;
        color: {COLORS['text_primary']};
    }}

    .chart-subtitle {{
        font-size: 11px;
        color: {COLORS['text_muted']};
        margin-left: 12px;
    }}

    .timeframe-pills {{
        display: flex;
        gap: 4px;
        background: {COLORS['bg_card']};
        padding: 4px;
        border-radius: 6px;
    }}

    .tf-pill {{
        padding: 6px 12px;
        font-size: 11px;
        font-weight: 600;
        color: {COLORS['text_muted']};
        background: transparent;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .tf-pill:hover {{
        color: {COLORS['text_primary']};
    }}

    .tf-pill.active {{
        background: {COLORS['accent_green']};
        color: {COLORS['bg_dark']};
    }}

    .chart-container {{
        flex: 1;
        min-height: 300px;
    }}

    .empty-chart {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: {COLORS['text_muted']};
    }}

    .empty-chart-text {{
        font-size: 16px;
        margin-bottom: 8px;
    }}

    .empty-chart-hint {{
        font-size: 12px;
        color: {COLORS['text_dark']};
    }}

    /* Zone D - Financial Intel */
    .zone-d {{
        width: 280px;
        min-width: 280px;
        background: {COLORS['bg_dark']};
        border-left: 1px solid {COLORS['border']};
        padding: 20px;
        overflow-y: auto;
    }}

    /* Bento Cards V2 */
    .bento-v2 {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
        cursor: default;
    }}

    .bento-v2:hover {{
        border-color: {COLORS['text_muted']};
    }}

    .bento-v2.clickable {{
        cursor: pointer;
    }}

    .bento-v2.clickable:hover {{
        background: {COLORS['bg_card_hover']};
    }}

    .bento-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }}

    .bento-title {{
        font-size: 10px;
        font-weight: 700;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .bento-chevron {{
        color: {COLORS['text_muted']};
        font-size: 14px;
    }}

    /* Financials Card */
    .fin-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }}

    .fin-label {{
        font-size: 11px;
        color: {COLORS['text_muted']};
    }}

    .fin-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}

    .fin-value.positive {{ color: {COLORS['accent_green']}; }}
    .fin-value.negative {{ color: {COLORS['accent_red']}; }}

    /* Earnings Call Card */
    .earnings-quarter {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: {COLORS['text_muted']};
        margin-bottom: 8px;
    }}

    .earnings-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        margin-bottom: 8px;
    }}

    .earnings-badge.bullish {{
        background: rgba(0, 240, 144, 0.15);
        color: {COLORS['accent_green']};
    }}

    .earnings-badge.bearish {{
        background: rgba(255, 46, 80, 0.15);
        color: {COLORS['accent_red']};
    }}

    .earnings-badge.neutral {{
        background: rgba(107, 114, 128, 0.15);
        color: {COLORS['text_muted']};
    }}

    .earnings-summary {{
        font-size: 12px;
        color: {COLORS['text_secondary']};
        line-height: 1.5;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}

    /* News Card Items */
    .news-item {{
        display: flex;
        align-items: flex-start;
        gap: 8px;
        padding: 8px 0;
        border-bottom: 1px solid {COLORS['border']};
    }}

    .news-item:last-child {{
        border-bottom: none;
    }}

    .news-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-top: 6px;
        flex-shrink: 0;
    }}

    .news-dot.bullish {{ background: {COLORS['accent_green']}; }}
    .news-dot.bearish {{ background: {COLORS['accent_red']}; }}
    .news-dot.neutral {{ background: {COLORS['text_muted']}; }}

    .news-title-text {{
        font-size: 11px;
        color: {COLORS['text_secondary']};
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}

    /* Insider Activity Card */
    .insider-bar-v2 {{
        display: flex;
        height: 24px;
        background: {COLORS['border']};
        border-radius: 4px;
        overflow: hidden;
        margin: 12px 0;
    }}

    .insider-buy-bar {{
        background: {COLORS['accent_green']};
        transition: width 0.5s ease;
    }}

    .insider-sell-bar {{
        background: {COLORS['accent_red']};
        transition: width 0.5s ease;
    }}

    .insider-counts {{
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
    }}

    .insider-count {{
        display: flex;
        align-items: center;
        gap: 4px;
    }}

    .insider-count.buy {{ color: {COLORS['accent_green']}; }}
    .insider-count.sell {{ color: {COLORS['accent_red']}; }}

    /* Dialog Styles */
    .dialog-overlay {{
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(4px);
        z-index: 2000;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .dialog-content {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 16px;
        max-width: 700px;
        width: 90%;
        max-height: 80vh;
        overflow: hidden;
    }}

    .dialog-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 24px;
        border-bottom: 1px solid {COLORS['border']};
    }}

    .dialog-title {{
        font-size: 16px;
        font-weight: 600;
        color: {COLORS['text_primary']};
    }}

    .dialog-close {{
        background: none;
        border: none;
        color: {COLORS['text_muted']};
        font-size: 24px;
        cursor: pointer;
    }}

    .dialog-body {{
        padding: 24px;
        max-height: 60vh;
        overflow-y: auto;
    }}

    /* Footer Disclaimer */
    .footer-disclaimer {{
        background: {COLORS['bg_dark']};
        border-top: 1px solid {COLORS['border']};
        padding: 8px 24px;
        font-size: 9px;
        color: {COLORS['text_muted']};
        text-align: center;
    }}

    /* Hide default Streamlit elements */
    .stButton > button {{
        background: {COLORS['accent_green']} !important;
        color: {COLORS['bg_dark']} !important;
        border: none !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }}

    /* Streamlit form submit */
    .stTextInput input {{
        background: {COLORS['bg_card']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['text_primary']} !important;
        border-radius: 8px !important;
    }}

    .stTextInput input:focus {{
        border-color: {COLORS['accent_green']} !important;
        box-shadow: 0 0 0 2px rgba(0, 240, 144, 0.1) !important;
    }}
</style>
""", unsafe_allow_html=True)

# Show disclaimer modal
show_disclaimer_modal()


# =============================================================================
# ANALYSIS FUNCTION (Keep existing logic)
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
        result["volume"] = getattr(quote, 'volume', 0)
        result["market_cap"] = getattr(quote, 'market_cap', 0)
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
    result["transcript_content"] = None

    if transcript and transcript.content:
        analyzer = TranscriptAnalyzer()
        ceo_analysis = analyzer.extract_features_ai(transcript.content, ticker)
        result["ceo_analysis"] = ceo_analysis
        result["transcript_content"] = transcript.content

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
    else:
        weighted_total = (financial_score * 0.61) + (social_score * 0.39)

    result["scores"] = {
        "financial": round(financial_score, 1),
        "ceo_tone": round(ceo_score, 1) if ceo_score is not None else None,
        "social": round(social_score, 1),
        "weighted_total": round(weighted_total, 1),
        "has_transcript": has_transcript,
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
# RENDER FUNCTIONS
# =============================================================================

def render_command_bar():
    """Render Zone A - Command Bar."""
    deep_reasoning = st.session_state.get("deep_reasoning", False)
    dr_color = COLORS['accent_purple'] if deep_reasoning else COLORS['text_muted']

    st.markdown(f"""
    <div class="command-bar-v2">
        <div class="logo-v2">
            <span class="logo-icon">⚡</span>
            <span class="logo-text-v2">EARNINGS HUNTER</span>
        </div>
        <div class="right-controls">
            <div class="deep-reasoning-toggle" style="color: {dr_color};">
                DEEP REASONING
            </div>
            <div class="live-indicator">
                <div class="live-dot"></div>
                LIVE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ticker_strip(result: Dict[str, Any]):
    """Render the ticker info strip."""
    price = result.get('current_price', 0)
    change = result.get('price_change', 0)
    change_pct = result.get('price_change_pct', 0)
    change_color = COLORS['accent_green'] if change >= 0 else COLORS['accent_red']
    change_sign = "+" if change >= 0 else ""

    volume = result.get('volume', 0)
    market_cap = result.get('market_cap', 0)

    # Format volume
    if volume >= 1e6:
        vol_str = f"{volume/1e6:.1f}M"
    elif volume >= 1e3:
        vol_str = f"{volume/1e3:.1f}K"
    else:
        vol_str = str(int(volume))

    # Format market cap
    if market_cap >= 1e12:
        cap_str = f"${market_cap/1e12:.1f}T"
    elif market_cap >= 1e9:
        cap_str = f"${market_cap/1e9:.1f}B"
    else:
        cap_str = f"${market_cap/1e6:.1f}M"

    st.markdown(f"""
    <div class="ticker-strip">
        <div class="ticker-left">
            <span class="company-name">{result.get('company_name', result['symbol'])}</span>
            <span class="ticker-price-lg">${price:,.2f}</span>
            <span class="ticker-change" style="color: {change_color};">
                {change_sign}{change:.2f} ({change_sign}{change_pct:.2f}%)
            </span>
        </div>
        <div class="ticker-right">
            <span>Vol: {vol_str}</span>
            <span>MCap: {cap_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_ring(confidence: float, prediction: str):
    """Render SVG confidence ring."""
    # Calculate stroke-dashoffset (377 is circumference, 0 is full)
    offset = 377 * (1 - confidence)

    # Get color based on prediction
    if prediction == "Growth":
        ring_color = COLORS['accent_green']
        verdict_class = "verdict-bullish"
        verdict_text = "BULLISH"
    elif prediction == "Risk":
        ring_color = COLORS['accent_red']
        verdict_class = "verdict-bearish"
        verdict_text = "BEARISH"
    else:
        ring_color = COLORS['accent_yellow']
        verdict_class = "verdict-neutral"
        verdict_text = "NEUTRAL"

    st.markdown(f"""
    <div class="confidence-ring-container">
        <div style="position: relative; width: 140px; height: 140px;">
            <svg class="confidence-ring-svg" viewBox="0 0 140 140">
                <circle class="ring-track" cx="70" cy="70" r="60"/>
                <circle class="ring-fill" cx="70" cy="70" r="60"
                        style="stroke: {ring_color}; stroke-dashoffset: {offset};"/>
            </svg>
            <div class="ring-center">
                <div class="ring-percentage" style="color: {ring_color};">{confidence:.0%}</div>
                <div class="ring-label">CONFIDENCE</div>
            </div>
        </div>
        <div class="verdict-badge">
            <div class="verdict-text {verdict_class}">{verdict_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ai_analysis(result: Dict[str, Any]):
    """Render AI Analysis text block."""
    ceo_data = result.get("ceo_analysis", {})
    fin_data = result.get("financial_data", {})
    prediction = result.get("prediction", "Stagnation")

    # Generate analysis text
    symbol = result.get("symbol", "")
    company = result.get("company_name", symbol)

    eps_beat = fin_data.get("eps_beat", False)
    eps_surprise = fin_data.get("eps_surprise", 0)

    if prediction == "Growth":
        sentiment = "exceptional"
        outlook = "strong upside potential"
    elif prediction == "Risk":
        sentiment = "concerning"
        outlook = "elevated risk"
    else:
        sentiment = "mixed"
        outlook = "sideways movement"

    analysis_text = f"{company} demonstrates {sentiment} fundamental strength "

    if eps_beat:
        analysis_text += f"with a significant {abs(eps_surprise):.0f}% EPS beat, "
    else:
        analysis_text += f"despite a {abs(eps_surprise):.0f}% EPS miss, "

    analysis_text += f"indicating {outlook}. "

    # Add CEO tone if available
    if ceo_data.get("has_transcript"):
        tone = ceo_data.get("tone_summary", "neutral")
        analysis_text += f"CEO tone analysis shows {tone.lower()} sentiment."

    st.markdown(f"""
    <div class="ai-analysis-section">
        <div class="ai-analysis-title">AI ANALYSIS</div>
        <div class="ai-analysis-text">{analysis_text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_price_chart(prices: List, ticker: str, timeframe: str = "1M"):
    """Render price chart with SMA and projection."""
    if not prices:
        return

    dates = [p.date for p in prices]
    closes = [p.close for p in prices]

    # Calculate SMA(20)
    sma_window = min(20, len(closes))
    sma = []
    for i in range(len(closes)):
        if i < sma_window - 1:
            sma.append(None)
        else:
            sma.append(sum(closes[i-sma_window+1:i+1]) / sma_window)

    # Create projection (simple linear extrapolation)
    if len(closes) >= 5:
        recent = closes[-5:]
        slope = (recent[-1] - recent[0]) / 4
        last_date = datetime.strptime(dates[-1], "%Y-%m-%d") if isinstance(dates[-1], str) else dates[-1]
        proj_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]
        proj_values = [closes[-1] + slope * i for i in range(1, 8)]
    else:
        proj_dates = []
        proj_values = []

    # Determine colors based on performance
    if len(closes) >= 2:
        price_change = closes[-1] - closes[0]
        line_color = COLORS['accent_green'] if price_change >= 0 else COLORS['accent_red']
        fill_color = 'rgba(0, 240, 144, 0.15)' if price_change >= 0 else 'rgba(255, 46, 80, 0.15)'
    else:
        line_color = COLORS['accent_green']
        fill_color = 'rgba(0, 240, 144, 0.15)'

    fig = go.Figure()

    # Area chart for price
    fig.add_trace(go.Scatter(
        x=dates,
        y=closes,
        mode='lines',
        name='Price',
        line=dict(color=line_color, width=2),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'
    ))

    # SMA line
    fig.add_trace(go.Scatter(
        x=dates,
        y=sma,
        mode='lines',
        name='SMA(20)',
        line=dict(color=COLORS['text_muted'], width=1, dash='solid'),
        hovertemplate='SMA: $%{y:.2f}<extra></extra>'
    ))

    # Projection line
    if proj_dates:
        fig.add_trace(go.Scatter(
            x=[dates[-1]] + proj_dates,
            y=[closes[-1]] + proj_values,
            mode='lines',
            name='Projection',
            line=dict(color=COLORS['text_muted'], width=1, dash='dot'),
            hovertemplate='Proj: $%{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text_primary']),
        height=320,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(
            gridcolor=COLORS['border'],
            showgrid=True,
            gridwidth=1,
            tickfont=dict(size=10, color=COLORS['text_muted']),
        ),
        yaxis=dict(
            gridcolor=COLORS['border'],
            showgrid=True,
            gridwidth=1,
            tickfont=dict(size=10, color=COLORS['text_muted']),
            side='right',
        ),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_financials_card(fin_data: Dict[str, Any]):
    """Render Financials bento card."""
    eps_actual = fin_data.get('eps_actual', 0)
    eps_est = fin_data.get('eps_estimated', 0)
    eps_surprise = fin_data.get('eps_surprise', 0)
    rev_actual = fin_data.get('revenue_actual', 0)
    rev_est = fin_data.get('revenue_estimated', 0)

    surprise_class = "positive" if eps_surprise >= 0 else "negative"

    st.markdown(f"""
    <div class="bento-v2">
        <div class="bento-header">
            <span class="bento-title">FINANCIALS</span>
        </div>
        <div class="fin-row">
            <span class="fin-label">ACTUAL</span>
            <span class="fin-value {surprise_class}">${eps_actual:.2f}</span>
        </div>
        <div class="fin-row">
            <span class="fin-label">EST.</span>
            <span class="fin-value">${eps_est:.2f}</span>
        </div>
        <div class="fin-row">
            <span class="fin-label">SURPRISE</span>
            <span class="fin-value {surprise_class}">{'+' if eps_surprise >= 0 else ''}{eps_surprise:.2f}</span>
        </div>
        <div class="fin-row">
            <span class="fin-label">REVENUE</span>
            <span class="fin-value">${rev_actual:.2f}B</span>
        </div>
        <div class="fin-row">
            <span class="fin-label">EST. REV</span>
            <span class="fin-value">${rev_est:.2f}B</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_earnings_call_card(result: Dict[str, Any]):
    """Render Earnings Call bento card."""
    quarter = result.get("quarter", "N/A")
    ceo_data = result.get("ceo_analysis", {})
    has_transcript = ceo_data.get("has_transcript", False)

    if has_transcript:
        confidence = ceo_data.get("confidence_score", 0.5)
        sentiment = ceo_data.get("sentiment_score", 0.5)

        if sentiment > 0.6:
            badge_class = "bullish"
            badge_text = "BULLISH"
        elif sentiment < 0.4:
            badge_class = "bearish"
            badge_text = "BEARISH"
        else:
            badge_class = "neutral"
            badge_text = "NEUTRAL"

        summary = ceo_data.get("executive_summary", "Click to view transcript...")[:100]
        if len(summary) >= 100:
            summary += "..."
    else:
        badge_class = "neutral"
        badge_text = "NO DATA"
        confidence = 0
        summary = "No transcript available"

    st.markdown(f"""
    <div class="bento-v2 clickable" onclick="document.getElementById('transcript-dialog').showModal()">
        <div class="bento-header">
            <span class="bento-title">EARNINGS CALL</span>
            <span class="bento-chevron">›</span>
        </div>
        <div class="earnings-quarter">{quarter}</div>
        <div>
            <span class="earnings-badge {badge_class}">{badge_text}</span>
            <span style="font-size: 11px; color: {COLORS['text_muted']}; margin-left: 8px;">{confidence:.0%}</span>
        </div>
        <div class="earnings-summary">{summary}</div>
    </div>
    """, unsafe_allow_html=True)

    # Expander for transcript dialog (Streamlit-native)
    if has_transcript:
        with st.expander("📄 View Earnings Call Transcript", expanded=False):
            st.markdown(f"**{result.get('company_name', result['symbol'])} - {quarter}**")
            st.markdown(f"**Sentiment:** {badge_text} ({confidence:.0%} confidence)")
            st.markdown("---")

            executive_summary = ceo_data.get("executive_summary", "No summary available.")
            st.markdown(f"**Summary:** {executive_summary}")
            st.markdown("---")

            transcript = result.get("transcript_content", "")
            if transcript:
                st.text_area("Full Transcript", transcript, height=300, disabled=True)


def render_news_card(news_articles: List[Dict[str, Any]]):
    """Render Latest News bento card."""
    items_html = ""
    for article in news_articles[:6]:
        score = article.get("score", 0)
        if score > 0.05:
            dot_class = "bullish"
        elif score < -0.05:
            dot_class = "bearish"
        else:
            dot_class = "neutral"

        title = article.get("title", "")[:80]
        if len(article.get("title", "")) > 80:
            title += "..."

        items_html += f"""
        <div class="news-item">
            <div class="news-dot {dot_class}"></div>
            <div class="news-title-text">{title}</div>
        </div>
        """

    if not items_html:
        items_html = '<div style="color: ' + COLORS['text_muted'] + '; font-size: 12px; padding: 16px; text-align: center;">No news</div>'

    st.markdown(f"""
    <div class="bento-v2">
        <div class="bento-header">
            <span class="bento-title">LATEST NEWS</span>
        </div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)


def render_insider_activity_card(result: Dict[str, Any]):
    """Render Insider Activity bento card."""
    fin_data = result.get("financial_data", {})
    buys = fin_data.get("insider_buys", 0)
    sells = fin_data.get("insider_sells", 0)
    total = buys + sells

    if total > 0:
        buy_pct = (buys / total) * 100
        sell_pct = (sells / total) * 100
    else:
        buy_pct = 50
        sell_pct = 50

    st.markdown(f"""
    <div class="bento-v2 clickable">
        <div class="bento-header">
            <span class="bento-title">INSIDER ACTIVITY</span>
            <span class="bento-chevron">›</span>
        </div>
        <div class="insider-bar-v2">
            <div class="insider-buy-bar" style="width: {buy_pct}%;"></div>
            <div class="insider-sell-bar" style="width: {sell_pct}%;"></div>
        </div>
        <div class="insider-counts">
            <div class="insider-count buy">BUY: {buys}</div>
            <div class="insider-count sell">SELL: {sells}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Expander for insider details
    insider_transactions = result.get("insider_transactions", [])
    if insider_transactions:
        with st.expander("📊 View Insider Activity Details", expanded=False):
            import pandas as pd

            # Group by quarter
            df = pd.DataFrame(insider_transactions)
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['quarter'] = df['date'].dt.to_period('Q').astype(str)

                for quarter in df['quarter'].unique()[:6]:
                    q_df = df[df['quarter'] == quarter]
                    q_buys = len(q_df[q_df['transaction'].str.contains('Buy', na=False, case=False)])
                    q_sells = len(q_df[q_df['transaction'].str.contains('Sell', na=False, case=False)])
                    total_shares = q_df['shares'].sum()

                    badge = "SELL" if q_sells > q_buys else "BUY" if q_buys > q_sells else "MIXED"
                    badge_color = COLORS['accent_red'] if q_sells > q_buys else COLORS['accent_green']

                    st.markdown(f"""
                    <div style="background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']};
                                border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: {COLORS['text_primary']};">
                                {quarter} | Buy: {q_buys} Sell: {q_sells}
                            </span>
                            <span style="background: rgba({badge_color}, 0.15); color: {badge_color};
                                        padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700;">
                                {badge}
                            </span>
                        </div>
                        <div style="font-size: 11px; color: {COLORS['text_muted']}; margin-top: 4px;">
                            {total_shares:,.0f} shares
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

# Render command bar (header)
render_command_bar()

# Search Section with better layout
col_search_left, col_search_center, col_search_right = st.columns([1, 3, 1])

with col_search_center:
    search_col1, search_col2, search_col3 = st.columns([5, 1, 1])

    with search_col1:
        ticker = st.text_input(
            "Search",
            placeholder="Search ticker... ⌘K",
            label_visibility="collapsed",
            key="ticker_input"
        )

    with search_col2:
        analyze_button = st.button("🎯", type="primary", use_container_width=True, help="Analyze stock")

    with search_col3:
        deep_analysis = st.checkbox("Deep", help="Run AI agents ($0.02)")
        st.session_state["deep_reasoning"] = deep_analysis

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
            status_text.markdown(f"<p style='text-align: center; color: {COLORS['accent_purple']};'>🤖 Running AI Agent analysis...</p>", unsafe_allow_html=True)
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
# DISPLAY RESULTS
# =============================================================================
if "result" in st.session_state:
    result = st.session_state["result"]
    fin_data = result.get("financial_data", {})

    # Ticker Info Strip
    render_ticker_strip(result)

    # 3-Column Cockpit Layout
    col_b, col_c, col_d = st.columns([1, 2.2, 1.1])

    # ZONE B - Intelligence Hub (Left)
    with col_b:
        st.markdown('<div class="zone-title">INTELLIGENCE HUB</div>', unsafe_allow_html=True)

        # SVG Confidence Ring
        render_confidence_ring(
            result.get("confidence", 0.5),
            result.get("prediction", "Stagnation")
        )

        # AI Analysis Text
        render_ai_analysis(result)

    # ZONE C - Market Stage (Center)
    with col_c:
        # Chart Header
        selected_tf = st.session_state.get("chart_timeframe", "1M")

        st.markdown(f"""
        <div class="chart-header">
            <div>
                <span class="chart-title">{result['symbol']} — {selected_tf}</span>
                <span class="chart-subtitle">SMA(20) + Projection</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Timeframe Buttons
        timeframe_options = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}

        if "chart_timeframe" not in st.session_state:
            st.session_state.chart_timeframe = "1M"

        tf_cols = st.columns(5)
        for i, tf in enumerate(timeframe_options.keys()):
            with tf_cols[i]:
                btn_type = "primary" if st.session_state.chart_timeframe == tf else "secondary"
                if st.button(tf, key=f"tf_{tf}", use_container_width=True, type=btn_type):
                    st.session_state.chart_timeframe = tf
                    st.rerun()

        # Fetch prices based on timeframe
        from src.data_ingestion.fmp_client import FMPClient
        from config.settings import get_settings

        settings = get_settings()
        fmp_client = FMPClient(api_key=settings.fmp_api_key)

        days = timeframe_options.get(st.session_state.chart_timeframe, 30)
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

        # Render chart
        if prices:
            render_price_chart(prices, result["symbol"], st.session_state.chart_timeframe)
        else:
            st.markdown("""
            <div class="empty-chart">
                <div class="empty-chart-text">Search a ticker to load chart</div>
                <div class="empty-chart-hint">⌘K to open search</div>
            </div>
            """, unsafe_allow_html=True)

    # ZONE D - Financial Intel (Right)
    with col_d:
        st.markdown('<div class="zone-title">FINANCIAL INTEL</div>', unsafe_allow_html=True)

        # Financials Card
        render_financials_card(fin_data)

        # Earnings Call Card
        render_earnings_call_card(result)

        # Latest News Card
        render_news_card(result.get("news_articles", []))

        # Insider Activity Card
        render_insider_activity_card(result)

else:
    # Empty state - show placeholder
    col_b, col_c, col_d = st.columns([1, 2.2, 1.1])

    with col_b:
        st.markdown('<div class="zone-title">INTELLIGENCE HUB</div>', unsafe_allow_html=True)

        # Empty confidence ring
        st.markdown(f"""
        <div class="confidence-ring-container">
            <div style="position: relative; width: 140px; height: 140px;">
                <svg class="confidence-ring-svg" viewBox="0 0 140 140">
                    <circle class="ring-track" cx="70" cy="70" r="60"/>
                </svg>
                <div class="ring-center">
                    <div class="ring-percentage" style="color: {COLORS['accent_red']};">0%</div>
                    <div class="ring-label">CONFIDENCE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div class="empty-chart" style="height: 400px;">
            <div class="empty-chart-text">Search a ticker to load chart</div>
            <div class="empty-chart-hint">⌘K to open search</div>
        </div>
        """, unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="zone-title">FINANCIAL INTEL</div>', unsafe_allow_html=True)

        for title in ["EPS", "EARNINGS CALL", "LATEST NEWS", "INSIDER ACTIVITY"]:
            st.markdown(f"""
            <div class="bento-v2">
                <div class="bento-header">
                    <span class="bento-title">{title}</span>
                </div>
                <div style="color: {COLORS['text_muted']}; font-size: 12px; text-align: center; padding: 20px;">
                    No data
                </div>
            </div>
            """, unsafe_allow_html=True)


# Footer Disclaimer
st.markdown("---")
show_disclaimer_footer()
