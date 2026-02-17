"""
Quote router - Real-time stock quote endpoint.

Fast endpoint for getting current price data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.data_ingestion.fmp_client import FMPClient
from config.settings import get_settings

router = APIRouter()

# Initialize FMP client
_fmp_client: Optional[FMPClient] = None


def get_fmp_client() -> FMPClient:
    """Get or create FMP client instance."""
    global _fmp_client
    if _fmp_client is None:
        settings = get_settings()
        _fmp_client = FMPClient(settings.fmp_api_key)
    return _fmp_client


class QuoteResponse(BaseModel):
    """Stock quote response."""
    symbol: str
    name: Optional[str] = None
    price: float
    change: Optional[float] = None
    change_percent: Optional[float] = None
    day_low: Optional[float] = None
    day_high: Optional[float] = None
    year_low: Optional[float] = None
    year_high: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[float] = None
    avg_volume: Optional[float] = None
    open_price: Optional[float] = None
    previous_close: Optional[float] = None
    pe: Optional[float] = None
    eps: Optional[float] = None


@router.get("/quote/{symbol}", response_model=QuoteResponse)
async def get_quote(symbol: str):
    """
    Get real-time stock quote.

    Returns current price, change, volume, and market cap.
    This is a fast endpoint for the ticker strip.
    """
    symbol = symbol.upper().strip()

    try:
        fmp_client = get_fmp_client()
        quote = fmp_client.get_stock_quote(symbol)

        if not quote:
            raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")

        # Calculate change_percent if not provided
        change_percent = quote.change_percentage
        if change_percent is None and quote.previous_close and quote.previous_close != 0:
            change_percent = (quote.change / quote.previous_close) * 100

        return {
            "symbol": quote.symbol,
            "name": quote.name,
            "price": quote.price,
            "change": quote.change,
            "change_percent": change_percent,
            "day_low": quote.day_low,
            "day_high": quote.day_high,
            "year_low": quote.year_low,
            "year_high": quote.year_high,
            "market_cap": quote.market_cap,
            "volume": quote.volume,
            "avg_volume": quote.avg_volume,
            "open_price": quote.open_price,
            "previous_close": quote.previous_close,
            "pe": quote.pe,
            "eps": quote.eps
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quote: {str(e)}")


class NewsItem(BaseModel):
    """News item response."""
    title: str
    text: Optional[str] = None
    url: Optional[str] = None
    published_date: Optional[str] = None
    site: Optional[str] = None
    score: float = 0.0
    sentiment: str = "neutral"


@router.get("/news/{symbol}", response_model=List[NewsItem])
async def get_news(symbol: str, limit: int = 20):
    """
    Get latest news for a stock with sentiment analysis.

    Returns news articles with NLP sentiment scores.
    """
    symbol = symbol.upper().strip()

    try:
        fmp_client = get_fmp_client()
        news = fmp_client.get_stock_news(symbol, limit=limit)

        if not news:
            return []

        # Import sentiment analyzer
        from src.feature_engineering.sentiment_features import SentimentFeatureExtractor
        sentiment_extractor = SentimentFeatureExtractor()

        result = []
        for article in news:
            # Analyze sentiment
            text = f"{article.title} {article.text or ''}"
            score = sentiment_extractor.analyze_single_text(text)

            result.append({
                "title": article.title,
                "text": article.text,
                "url": article.url,
                "published_date": str(article.published_date) if article.published_date else None,
                "site": article.site,
                "score": score,
                "sentiment": "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get news: {str(e)}")


class InsiderTradeResponse(BaseModel):
    """Insider trade response."""
    date: str
    reporter: str
    transaction: str
    shares: float
    price: Optional[float] = None
    value: Optional[float] = None
    type: str  # BUY or SELL


@router.get("/insiders/{symbol}", response_model=List[InsiderTradeResponse])
async def get_insider_trades(symbol: str, limit: int = 50):
    """
    Get insider trading activity.

    Returns list of insider buy/sell transactions.
    """
    symbol = symbol.upper().strip()

    try:
        fmp_client = get_fmp_client()
        trades = fmp_client.get_insider_trading(symbol, limit=limit)

        if not trades:
            return []

        result = []
        for trade in trades:
            trade_type = "BUY" if trade.is_purchase else "SELL" if trade.is_sale else "OTHER"

            result.append({
                "date": str(trade.transaction_date),
                "reporter": trade.reporting_name or "Unknown",
                "transaction": trade.transaction_type,
                "shares": trade.securities_transacted,
                "price": trade.price,
                "value": trade.transaction_value,
                "type": trade_type
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insider trades: {str(e)}")


class TranscriptResponse(BaseModel):
    """Earnings call transcript response."""
    symbol: str
    year: int
    quarter: int
    date: Optional[str] = None
    content: str
    analysis: Dict[str, Any]


@router.get("/transcript/{symbol}")
async def get_transcript(
    symbol: str,
    year: Optional[int] = None,
    quarter: Optional[int] = None
):
    """
    Get earnings call transcript with AI analysis.

    If year/quarter not specified, returns latest available.
    """
    symbol = symbol.upper().strip()

    try:
        fmp_client = get_fmp_client()

        # If no year/quarter, get latest earnings info
        if not year or not quarter:
            earnings_info = fmp_client.get_latest_earnings_info(symbol)
            if earnings_info:
                year = earnings_info["year"]
                quarter = earnings_info["quarter"]
            else:
                raise HTTPException(status_code=404, detail="No earnings data found")

        # Get transcript
        transcript = fmp_client.get_earnings_call_transcript(symbol, year, quarter)

        if not transcript or not transcript.content:
            raise HTTPException(
                status_code=404,
                detail=f"No transcript available for {symbol} Q{quarter} {year}"
            )

        # Analyze transcript
        from src.feature_engineering.transcript_analyzer import TranscriptAnalyzer
        analyzer = TranscriptAnalyzer()
        analysis = analyzer.calculate_ceo_sentiment_score(transcript.content)

        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "date": str(transcript.date) if transcript.date else None,
            "content": transcript.content,
            "analysis": {
                "confidence_score": analysis.get("ceo_confidence_score"),
                "sentiment_score": analysis.get("ceo_score"),
                "tone_summary": analysis.get("tone_summary"),
                "key_positive_phrases": analysis.get("key_positive_phrases", []),
                "key_negative_phrases": analysis.get("key_negative_phrases", []),
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transcript: {str(e)}")
