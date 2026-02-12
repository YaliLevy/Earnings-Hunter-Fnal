"""
Analysis router - Main endpoint for stock analysis.

Wraps the existing EarningsOrchestrator to expose analysis via REST API.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import traceback

from src.agents.orchestrator import EarningsOrchestrator, AnalysisResult
from src.agents.crew import EarningsHunterCrew
from config.settings import get_settings

router = APIRouter()

# Initialize orchestrator (singleton pattern)
_orchestrator: Optional[EarningsOrchestrator] = None


def get_orchestrator() -> EarningsOrchestrator:
    """Get or create the orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = EarningsOrchestrator()
    return _orchestrator


# Response models
class GoldenTriangleScore(BaseModel):
    """Individual Golden Triangle component score."""
    score: float
    weight: float
    weighted_score: float
    label: str


class GoldenTriangle(BaseModel):
    """Golden Triangle analysis result."""
    financial: GoldenTriangleScore
    ceo_tone: GoldenTriangleScore
    social: GoldenTriangleScore
    composite: Dict[str, Any]


class FinancialSummary(BaseModel):
    """Financial data summary."""
    eps_actual: Optional[float] = None
    eps_estimated: Optional[float] = None
    eps_surprise: Optional[float] = None
    eps_beat: Optional[bool] = None
    revenue_actual: Optional[float] = None
    revenue_estimated: Optional[float] = None
    revenue_surprise: Optional[float] = None
    revenue_beat: Optional[bool] = None
    double_beat: Optional[bool] = None
    current_price: Optional[float] = None
    price_change: Optional[float] = None
    price_change_pct: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[float] = None
    insider_buys: Optional[int] = None
    insider_sells: Optional[int] = None


class CEOToneSummary(BaseModel):
    """CEO tone analysis summary."""
    has_transcript: bool = False
    confidence_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    tone_summary: Optional[str] = None
    executive_summary: Optional[str] = None
    key_positive_phrases: Optional[List[str]] = None
    key_negative_phrases: Optional[List[str]] = None


class NewsArticle(BaseModel):
    """News article with sentiment."""
    title: str
    url: Optional[str] = None
    published_date: Optional[str] = None
    source: Optional[str] = None
    score: float = 0.0
    sentiment: str = "neutral"  # bullish, bearish, neutral


class SocialSummary(BaseModel):
    """Social sentiment summary."""
    available: bool = False
    news_count: int = 0
    combined_sentiment: float = 0.0
    bullish_percentage: float = 50.0
    sentiment_classification: str = "Neutral"


class InsiderTrade(BaseModel):
    """Individual insider trade."""
    date: str
    reporter: str
    transaction: str
    shares: float
    price: Optional[float] = None
    value: Optional[float] = None
    type: str  # BUY or SELL


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    symbol: str
    company_name: Optional[str] = None
    earnings_date: str
    year: int
    quarter: int
    quarter_label: str
    prediction: str
    confidence: float
    golden_triangle: Dict[str, Any]
    financial_summary: Dict[str, Any]
    ceo_tone_summary: Dict[str, Any]
    social_summary: Dict[str, Any]
    news_articles: List[Dict[str, Any]]
    insider_transactions: List[Dict[str, Any]]
    transcript_content: Optional[str] = None
    research_insight: Optional[str] = None
    disclaimer: str


@router.get("/analyze/{symbol}", response_model=AnalysisResponse)
async def analyze_stock(
    symbol: str,
    force_refresh: bool = Query(False, description="Force refresh, ignore cache")
):
    """
    Analyze a stock's latest earnings.

    Returns complete Golden Triangle analysis including:
    - Financial metrics (40% weight)
    - CEO tone analysis (35% weight)
    - Social sentiment (25% weight)
    - ML prediction with confidence
    """
    symbol = symbol.upper().strip()

    try:
        orchestrator = get_orchestrator()
        result = orchestrator.analyze_latest_earnings(symbol, force_refresh=force_refresh)

        # Convert AnalysisResult to response format
        return _convert_to_response(result)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/deep-analysis/{symbol}")
async def deep_analysis(
    symbol: str,
    prediction: str = Query("Stagnation", description="Current prediction"),
    confidence: float = Query(0.5, description="Current confidence")
):
    """
    Run CrewAI agent deep analysis.

    This is more expensive ($0.01-0.02) but provides deeper research insight.
    """
    symbol = symbol.upper().strip()

    try:
        settings = get_settings()
        crew = EarningsHunterCrew()

        # Run crew analysis
        report = crew.kickoff(
            symbol=symbol,
            prediction=prediction,
            confidence=confidence
        )

        return {
            "symbol": symbol,
            "report": report,
            "status": "completed"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Deep analysis failed: {str(e)}")


def _convert_to_response(result: AnalysisResult) -> dict:
    """Convert AnalysisResult to API response format."""

    # Extract financial data
    fin_data = result.financial_summary or {}
    ceo_data = result.ceo_tone_summary or {}
    social_data = result.social_summary or {}

    # Get news articles directly from result (now stored in AnalysisResult)
    # Map 'site' to 'source' for frontend compatibility
    raw_news = result.news_articles if hasattr(result, 'news_articles') and result.news_articles else []
    news_articles = []
    for article in raw_news:
        news_articles.append({
            "title": article.get("title", ""),
            "url": article.get("url"),
            "published_date": article.get("published_date"),
            "source": article.get("site"),  # Map site -> source
            "score": article.get("score", 0),
            "sentiment": article.get("sentiment", "neutral")
        })

    # Get insider transactions directly from result (now stored in AnalysisResult)
    # Ensure all required fields are present
    raw_insiders = result.insider_transactions if hasattr(result, 'insider_transactions') and result.insider_transactions else []
    insider_transactions = []
    for trade in raw_insiders:
        insider_transactions.append({
            "date": trade.get("date", ""),
            "reporter": trade.get("reporter", "Unknown"),
            "transaction": trade.get("transaction", ""),
            "shares": trade.get("shares", 0),
            "price": trade.get("price"),
            "value": trade.get("value"),
            "type": trade.get("type", "OTHER")
        })

    # Quarter label
    quarter_label = f"Q{result.quarter} {result.year}"

    # Build executive summary from CEO analysis
    executive_summary = None
    if ceo_data.get('available'):
        tone = ceo_data.get('tone_summary', 'neutral')
        positive = ceo_data.get('key_positive_phrases', [])
        negative = ceo_data.get('key_negative_phrases', [])

        summary_parts = [f"Management tone is {tone}."]
        if positive:
            summary_parts.append(f"Positive highlights: {', '.join(positive[:2])}.")
        if negative:
            summary_parts.append(f"Concerns mentioned: {', '.join(negative[:2])}.")
        executive_summary = " ".join(summary_parts)

    # Get transcript content
    transcript_content = result.transcript_content if hasattr(result, 'transcript_content') else None

    return {
        "symbol": result.symbol,
        "company_name": fin_data.get('company_name', result.symbol),
        "earnings_date": result.earnings_date,
        "year": result.year,
        "quarter": result.quarter,
        "quarter_label": quarter_label,
        "prediction": result.prediction,
        "confidence": result.confidence or 0.5,
        "golden_triangle": result.golden_triangle,
        "financial_summary": {
            "eps_actual": fin_data.get('eps_actual'),
            "eps_estimated": fin_data.get('eps_estimated'),
            "eps_surprise": fin_data.get('eps_surprise', fin_data.get('eps_surprise_pct')),
            "eps_beat": fin_data.get('eps_beat'),
            "revenue_actual": fin_data.get('revenue_actual'),
            "revenue_estimated": fin_data.get('revenue_estimated'),
            "revenue_surprise": fin_data.get('revenue_surprise', fin_data.get('revenue_surprise_pct')),
            "revenue_beat": fin_data.get('revenue_beat'),
            "double_beat": fin_data.get('double_beat'),
            "current_price": fin_data.get('current_price'),
            "price_change": fin_data.get('price_change'),
            "price_change_pct": fin_data.get('price_change_pct'),
            "market_cap": fin_data.get('market_cap'),
            "volume": fin_data.get('volume'),
            "insider_buys": fin_data.get('insider_buys', 0),
            "insider_sells": fin_data.get('insider_sells', 0),
        },
        "ceo_tone_summary": {
            "has_transcript": ceo_data.get('available', False),
            "confidence_score": ceo_data.get('confidence_score'),
            "sentiment_score": ceo_data.get('sentiment_score', ceo_data.get('ceo_score')),
            "tone_summary": ceo_data.get('tone_summary'),
            "executive_summary": executive_summary,
            "key_positive_phrases": ceo_data.get('key_positive_phrases', []),
            "key_negative_phrases": ceo_data.get('key_negative_phrases', []),
        },
        "social_summary": {
            "available": social_data.get('available', True),
            "news_count": social_data.get('news_count', 0),
            "combined_sentiment": social_data.get('combined_sentiment', 0),
            "bullish_percentage": social_data.get('bullish_percentage', 50),
            "sentiment_classification": social_data.get('sentiment_classification', 'Neutral'),
        },
        "news_articles": news_articles,
        "insider_transactions": insider_transactions,
        "transcript_content": transcript_content,
        "research_insight": result.research_insight.get('insight') if result.research_insight else None,
        "disclaimer": result.disclaimer
    }
