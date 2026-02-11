"""
FMP API Tools for CrewAI agents.

Provides BaseTool classes for accessing FMP Ultimate API data.
Compatible with CrewAI 1.x+
"""

from typing import Optional, Type
from pydantic import BaseModel, Field
import json

from crewai.tools import BaseTool

from src.data_ingestion.fmp_client import FMPClient
from src.utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Initialize client singleton
_fmp_client: Optional[FMPClient] = None


def get_fmp_client() -> FMPClient:
    """Get or create FMP client singleton."""
    global _fmp_client
    if _fmp_client is None:
        _fmp_client = FMPClient(api_key=settings.fmp_api_key)
    return _fmp_client


# ===== Tool Input Schemas =====

class SymbolInput(BaseModel):
    """Input schema for tools that only need a symbol."""
    symbol: str = Field(description="Stock ticker symbol (e.g., NVDA, AAPL)")


class DateRangeInput(BaseModel):
    """Input schema for date range tools."""
    from_date: str = Field(description="Start date (YYYY-MM-DD)")
    to_date: str = Field(description="End date (YYYY-MM-DD)")


class SymbolDateRangeInput(BaseModel):
    """Input schema for symbol + date range tools."""
    symbol: str = Field(description="Stock ticker symbol")
    from_date: str = Field(description="Start date (YYYY-MM-DD)")
    to_date: str = Field(description="End date (YYYY-MM-DD)")


class TranscriptInput(BaseModel):
    """Input schema for transcript tool."""
    symbol: str = Field(description="Stock ticker symbol")
    year: int = Field(description="Year of earnings call (e.g., 2025)")
    quarter: int = Field(description="Quarter (1, 2, 3, or 4)")


class TrendingInput(BaseModel):
    """Input schema for trending tool."""
    sentiment_type: str = Field(default="bullish", description="'bullish' or 'bearish'")


# ===== Tool Classes =====

class FetchStockQuoteTool(BaseTool):
    name: str = "Fetch Stock Quote"
    description: str = "Fetch current stock quote including price, change, and market cap."
    args_schema: Type[BaseModel] = SymbolInput

    def _run(self, symbol: str) -> str:
        try:
            client = get_fmp_client()
            quote = client.get_stock_quote(symbol)
            if quote:
                return json.dumps(quote.model_dump(), indent=2)
            return json.dumps({"error": f"No quote found for {symbol}"})
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return json.dumps({"error": str(e)})


class FetchEarningsSurprisesTool(BaseTool):
    name: str = "Fetch Earnings Surprises"
    description: str = "Fetch historical earnings surprises (actual vs estimated EPS/revenue)."
    args_schema: Type[BaseModel] = SymbolInput

    def _run(self, symbol: str) -> str:
        try:
            client = get_fmp_client()
            surprises = client.get_earnings_surprises(symbol)
            return json.dumps([s.model_dump() for s in surprises], indent=2)
        except Exception as e:
            logger.error(f"Error fetching earnings surprises for {symbol}: {e}")
            return json.dumps({"error": str(e)})


class FetchIncomeStatementTool(BaseTool):
    name: str = "Fetch Income Statement"
    description: str = "Fetch quarterly income statements for financial analysis."
    args_schema: Type[BaseModel] = SymbolInput

    def _run(self, symbol: str) -> str:
        try:
            client = get_fmp_client()
            statements = client.get_income_statement(symbol, period="quarter", limit=8)
            return json.dumps([s.model_dump() for s in statements], indent=2)
        except Exception as e:
            logger.error(f"Error fetching income statement for {symbol}: {e}")
            return json.dumps({"error": str(e)})


class FetchEarningsTranscriptTool(BaseTool):
    name: str = "Fetch Earnings Transcript"
    description: str = "Fetch full earnings call transcript for CEO tone analysis. CRITICAL for the 35% CEO Tone component."
    args_schema: Type[BaseModel] = TranscriptInput

    def _run(self, symbol: str, year: int, quarter: int) -> str:
        try:
            client = get_fmp_client()
            transcript = client.get_earnings_call_transcript(symbol, year, quarter)
            if transcript:
                return json.dumps(transcript.model_dump(), indent=2)
            return json.dumps({"error": f"No transcript found for {symbol} Q{quarter} {year}"})
        except Exception as e:
            logger.error(f"Error fetching transcript for {symbol} Q{quarter} {year}: {e}")
            return json.dumps({"error": str(e)})


class FetchAnalystDataTool(BaseTool):
    name: str = "Fetch Analyst Data"
    description: str = "Fetch analyst estimates, price targets, and ratings."
    args_schema: Type[BaseModel] = SymbolInput

    def _run(self, symbol: str) -> str:
        try:
            client = get_fmp_client()
            estimates = client.get_analyst_estimates(symbol)
            price_target = client.get_price_target(symbol)

            result = {
                "analyst_estimates": [e.model_dump() for e in estimates] if estimates else [],
                "price_target_consensus": price_target.model_dump() if price_target else None
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error fetching analyst data for {symbol}: {e}")
            return json.dumps({"error": str(e)})


class FetchInsiderActivityTool(BaseTool):
    name: str = "Fetch Insider Activity"
    description: str = "Fetch recent insider trading transactions (buys/sells)."
    args_schema: Type[BaseModel] = SymbolInput

    def _run(self, symbol: str) -> str:
        try:
            client = get_fmp_client()
            transactions = client.get_insider_trading(symbol, limit=100)
            return json.dumps([t.model_dump() for t in transactions], indent=2)
        except Exception as e:
            logger.error(f"Error fetching insider activity for {symbol}: {e}")
            return json.dumps({"error": str(e)})


class FetchStockNewsTool(BaseTool):
    name: str = "Fetch Stock News"
    description: str = "Fetch recent news articles for sentiment analysis."
    args_schema: Type[BaseModel] = SymbolInput

    def _run(self, symbol: str) -> str:
        try:
            client = get_fmp_client()
            news = client.get_stock_news(symbol, limit=50)
            return json.dumps([n.model_dump() for n in news], indent=2)
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return json.dumps({"error": str(e)})


class FetchHistoricalPricesTool(BaseTool):
    name: str = "Fetch Historical Prices"
    description: str = "Fetch historical stock prices for a date range."
    args_schema: Type[BaseModel] = SymbolDateRangeInput

    def _run(self, symbol: str, from_date: str, to_date: str) -> str:
        try:
            client = get_fmp_client()
            prices = client.get_historical_prices(symbol, from_date, to_date)
            return json.dumps([p.model_dump() for p in prices], indent=2)
        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {e}")
            return json.dumps({"error": str(e)})


class FetchLatestEarningsInfoTool(BaseTool):
    name: str = "Fetch Latest Earnings Info"
    description: str = "Get the most recent earnings event for a symbol."
    args_schema: Type[BaseModel] = SymbolInput

    def _run(self, symbol: str) -> str:
        try:
            client = get_fmp_client()
            info = client.get_latest_earnings_info(symbol)
            if info:
                return json.dumps(info, indent=2)
            return json.dumps({"error": f"No earnings info found for {symbol}"})
        except Exception as e:
            logger.error(f"Error fetching latest earnings info for {symbol}: {e}")
            return json.dumps({"error": str(e)})


# ===== Social Sentiment Tools =====

class FetchSocialSentimentTool(BaseTool):
    name: str = "Fetch Social Sentiment"
    description: str = "Fetch social media sentiment (Twitter/StockTwits). This is the 25% Street Psychology component."
    args_schema: Type[BaseModel] = SymbolInput

    def _run(self, symbol: str) -> str:
        try:
            client = get_fmp_client()
            social = client.get_social_sentiment(symbol)
            news = client.get_stock_news(symbol, limit=20)

            result = {
                "symbol": symbol.upper(),
                "social_sentiment": social.model_dump() if social else None,
                "news_count": len(news) if news else 0,
                "source": "FMP Stock News (API social sentiment may be unavailable)"
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error fetching social sentiment for {symbol}: {e}")
            return json.dumps({"error": str(e)})


# ===== List of all tools =====

ALL_FMP_TOOLS = [
    FetchStockQuoteTool(),
    FetchEarningsSurprisesTool(),
    FetchIncomeStatementTool(),
    FetchEarningsTranscriptTool(),
    FetchAnalystDataTool(),
    FetchInsiderActivityTool(),
    FetchStockNewsTool(),
    FetchHistoricalPricesTool(),
    FetchLatestEarningsInfoTool(),
    FetchSocialSentimentTool(),
]
