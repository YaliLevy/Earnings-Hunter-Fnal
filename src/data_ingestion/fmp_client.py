"""
Financial Modeling Prep (FMP) API client.

FMP Ultimate subscription provides unlimited API access.
No rate limiting required.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.logger import get_logger
from src.data_ingestion.validators import (
    EarningsData,
    EarningsTranscript,
    FinancialStatement,
    AnalystEstimate,
    PriceTarget,
    InsiderTransaction,
    InstitutionalHolder,
    StockQuote,
    HistoricalPrice,
    StockNews,
    SocialSentiment,
    SentimentRSS,
)

logger = get_logger(__name__)


class FMPClientError(Exception):
    """Custom exception for FMP API errors."""
    pass


class FMPClient:
    """
    FMP API client with automatic retries.

    Updated for FMP's new /stable/ API (August 2025+).
    FMP Ultimate subscription - NO rate limiting needed.
    """

    BASE_URL = "https://financialmodelingprep.com/stable"
    BASE_URL_V3 = "https://financialmodelingprep.com/api/v3"
    BASE_URL_V4 = "https://financialmodelingprep.com/api/v4"

    def __init__(self, api_key: str):
        """
        Initialize FMP client.

        Args:
            api_key: FMP API key (Ultimate subscription)
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EarningsHunter/1.0"
        })

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, requests.Timeout))
    )
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        version: str = "stable"
    ) -> Any:
        """
        Make API request with automatic retries.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            version: API version ("stable", "v3", or "v4")

        Returns:
            JSON response data
        """
        params = params or {}
        params["apikey"] = self.api_key

        # Select base URL based on version
        if version == "v3":
            url = f"{self.BASE_URL_V3}/{endpoint}"
        elif version == "v4":
            url = f"{self.BASE_URL_V4}/{endpoint}"
        else:
            url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API error messages
            if isinstance(data, dict) and "Error Message" in data:
                raise FMPClientError(data["Error Message"])

            return data

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {endpoint}: {e}")
            raise FMPClientError(f"HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            raise

    # ===== Earnings Functions =====

    def get_earnings_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[EarningsData]:
        """
        Fetch earnings calendar for date range.

        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            List of EarningsData objects
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        # New stable API endpoint
        data = self._request("earnings-calendar", params)

        if not data:
            return []

        return [EarningsData.model_validate(item) for item in data]

    def get_earnings_surprises(self, symbol: str) -> List[EarningsData]:
        """
        Fetch historical earnings surprises for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of EarningsData objects with actual vs estimated
        """
        # New stable API: use query param instead of path
        params = {"symbol": symbol.upper()}
        data = self._request("earnings", params)

        if not data:
            return []

        return [EarningsData.model_validate(item) for item in data]

    def get_historical_earnings(self, symbol: str) -> List[EarningsData]:
        """
        Fetch historical earnings calendar for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of EarningsData objects
        """
        # New stable API: same as earnings with symbol param
        params = {"symbol": symbol.upper()}
        data = self._request("earnings", params)

        if not data:
            return []

        return [EarningsData.model_validate(item) for item in data]

    # ===== Transcript Functions (CRITICAL - 35% of model) =====

    def get_earnings_call_transcript(
        self,
        symbol: str,
        year: int,
        quarter: int
    ) -> Optional[EarningsTranscript]:
        """
        Fetch earnings call transcript for a specific quarter.

        This is CRITICAL for CEO tone analysis (35% of Golden Triangle).

        Args:
            symbol: Stock ticker symbol
            year: Fiscal year
            quarter: Fiscal quarter (1-4)

        Returns:
            EarningsTranscript object or None if not available
        """
        if quarter not in [1, 2, 3, 4]:
            raise ValueError(f"Quarter must be 1-4, got {quarter}")

        # New stable API endpoint
        params = {"symbol": symbol.upper(), "year": year, "quarter": quarter}
        data = self._request("earning-call-transcript", params)

        if not data or len(data) == 0:
            logger.warning(f"No transcript found for {symbol} Q{quarter} {year}")
            return None

        # FMP returns a list, get first item
        transcript_data = data[0] if isinstance(data, list) else data

        return EarningsTranscript(
            symbol=symbol.upper(),
            year=year,
            quarter=quarter,
            date=transcript_data.get("date", ""),
            content=transcript_data.get("content", "")
        )

    def get_available_transcripts(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get list of available transcripts for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of available transcript dates
        """
        # New stable API endpoint
        params = {"symbol": symbol.upper()}
        data = self._request("earning-call-transcript-dates", params)
        return data if data else []

    # ===== Financial Statement Functions =====

    def get_income_statement(
        self,
        symbol: str,
        period: str = "quarter",
        limit: int = 8
    ) -> List[FinancialStatement]:
        """
        Fetch income statements for a symbol.

        Args:
            symbol: Stock ticker symbol
            period: "quarter" or "annual"
            limit: Number of statements to fetch

        Returns:
            List of FinancialStatement objects
        """
        # New stable API endpoint
        params = {"symbol": symbol.upper(), "period": period, "limit": limit}
        data = self._request("income-statement", params)

        if not data:
            return []

        return [FinancialStatement.model_validate(item) for item in data]

    # ===== Analyst Functions =====

    def get_analyst_estimates(self, symbol: str, period: str = "quarterly") -> List[AnalystEstimate]:
        """
        Fetch analyst EPS and revenue estimates.

        Args:
            symbol: Stock ticker symbol
            period: "quarterly" or "annual"

        Returns:
            List of AnalystEstimate objects
        """
        # New stable API endpoint - requires period parameter
        params = {"symbol": symbol.upper(), "period": period}
        data = self._request("analyst-estimates", params)

        if not data:
            return []

        return [AnalystEstimate.model_validate(item) for item in data]

    def get_price_target(self, symbol: str) -> Optional[PriceTarget]:
        """
        Fetch analyst price target consensus.

        Args:
            symbol: Stock ticker symbol

        Returns:
            PriceTarget object or None
        """
        # New stable API endpoint
        params = {"symbol": symbol.upper()}
        data = self._request("price-target-consensus", params)

        if not data or len(data) == 0:
            return None

        # FMP returns a list, get first item
        target_data = data[0] if isinstance(data, list) else data
        return PriceTarget.model_validate(target_data)

    def get_analyst_recommendations(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch analyst recommendations (buy/sell ratings).

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of recommendation data
        """
        # v3 API: symbol in path
        data = self._request(f"analyst-stock-recommendations/{symbol.upper()}", version="v3")
        return data if data else []

    # ===== Insider Trading Functions =====

    def get_insider_trading(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[InsiderTransaction]:
        """
        Fetch recent insider trading transactions.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum transactions to fetch

        Returns:
            List of InsiderTransaction objects
        """
        # New stable API endpoint - use /search path
        params = {"symbol": symbol.upper(), "limit": limit}
        data = self._request("insider-trading/search", params)

        if not data:
            return []

        return [InsiderTransaction.model_validate(item) for item in data]

    # ===== Institutional Holdings Functions =====

    def get_institutional_holders(self, symbol: str) -> List[InstitutionalHolder]:
        """
        Fetch major institutional holders.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of InstitutionalHolder objects
        """
        # v3 API: symbol in path
        data = self._request(f"institutional-holder/{symbol.upper()}", version="v3")

        if not data:
            return []

        return [InstitutionalHolder.model_validate(item) for item in data]

    # ===== Stock Price Functions =====

    def get_stock_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Fetch current stock quote.

        Args:
            symbol: Stock ticker symbol

        Returns:
            StockQuote object or None
        """
        # New stable API endpoint
        params = {"symbol": symbol.upper()}
        data = self._request("quote", params)

        if not data or len(data) == 0:
            return None

        return StockQuote.model_validate(data[0])

    def get_historical_prices(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[HistoricalPrice]:
        """
        Fetch historical stock prices.

        Args:
            symbol: Stock ticker symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            List of HistoricalPrice objects
        """
        # New stable API endpoint
        params = {"symbol": symbol.upper()}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = self._request("historical-price-eod/full", params)

        if not data:
            return []

        # New API returns list directly, not nested under "historical"
        return [HistoricalPrice.model_validate(item) for item in data]

    def get_price_on_date(
        self,
        symbol: str,
        date: str
    ) -> Optional[HistoricalPrice]:
        """
        Get stock price for a specific date.

        Args:
            symbol: Stock ticker symbol
            date: Date string (YYYY-MM-DD)

        Returns:
            HistoricalPrice object or None
        """
        prices = self.get_historical_prices(symbol, from_date=date, to_date=date)
        return prices[0] if prices else None

    # ===== News Functions =====

    def get_stock_news(
        self,
        symbol: str,
        limit: int = 50
    ) -> List[StockNews]:
        """
        Fetch recent news articles for a stock.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum articles to fetch

        Returns:
            List of StockNews objects
        """
        # New stable API endpoint - news/stock with symbols parameter (plural!)
        params = {"symbols": symbol.upper(), "limit": limit}
        data = self._request("news/stock", params)

        if not data:
            return []

        return [StockNews.model_validate(item) for item in data]

    # ===== Helper Functions =====

    def get_latest_earnings_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent earnings event for a symbol.

        This is used by the orchestrator to automatically find
        the latest earnings report.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with earnings date, year, quarter info
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Get historical earnings surprises
        earnings = self.get_historical_earnings(symbol)

        if not earnings:
            logger.warning(f"No earnings history found for {symbol}")
            return None

        # Filter to past dates and sort by date descending
        past_earnings = [
            e for e in earnings
            if e.date and e.date <= today
        ]

        if not past_earnings:
            return None

        # Sort by date descending
        past_earnings.sort(key=lambda x: x.date, reverse=True)
        latest = past_earnings[0]

        # Determine the correct fiscal year and quarter.
        # Use the available transcripts list from FMP which contains the correct
        # fiscalYear. This handles companies with non-calendar fiscal years
        # (e.g., NVDA fiscal year ends in January).
        quarter = None
        year = None

        try:
            available_transcripts = self.get_available_transcripts(symbol)
            if available_transcripts:
                # Match transcript by earnings date
                for t in available_transcripts:
                    if t.get("date") == latest.date:
                        quarter = t["quarter"]
                        year = t["fiscalYear"]
                        logger.info(f"Matched transcript: Q{quarter} FY{year} (date: {latest.date})")
                        break

                # If no exact date match, take the most recent transcript
                if quarter is None and available_transcripts:
                    t = available_transcripts[0]
                    quarter = t["quarter"]
                    year = t["fiscalYear"]
                    logger.info(f"Using latest available transcript: Q{quarter} FY{year} (date: {t.get('date')})")
        except Exception as e:
            logger.warning(f"Failed to fetch available transcripts: {e}")

        # Fallback: use income statements if transcript lookup failed
        if quarter is None or year is None:
            statements = self.get_income_statement(symbol, period="quarter", limit=10)
            if statements:
                for stmt in statements:
                    stmt_date = datetime.strptime(stmt.date, "%Y-%m-%d")
                    earnings_date = datetime.strptime(latest.date, "%Y-%m-%d")
                    days_diff = (earnings_date - stmt_date).days

                    if 0 <= days_diff <= 120:
                        if stmt.period and stmt.period.startswith("Q"):
                            quarter = int(stmt.period[1])
                            if stmt.calendar_year:
                                year = int(stmt.calendar_year)
                            else:
                                year = stmt_date.year
                            logger.info(f"Fallback: matched to {stmt.period} FY{year}")
                            break

        # Last fallback: date-based approximation
        if quarter is None or year is None:
            logger.warning(f"Could not determine fiscal quarter, using date approximation")
            date_obj = datetime.strptime(latest.date, "%Y-%m-%d")
            month = date_obj.month
            if month in [1, 2, 3]:
                quarter = 4
                year = date_obj.year - 1
            elif month in [4, 5, 6]:
                quarter = 1
                year = date_obj.year
            elif month in [7, 8, 9]:
                quarter = 2
                year = date_obj.year
            else:
                quarter = 3
                year = date_obj.year

        return {
            "symbol": symbol.upper(),
            "earnings_date": latest.date,
            "year": year,
            "quarter": quarter,
            "eps": latest.eps,
            "eps_estimated": latest.eps_estimated,
            "revenue": latest.revenue,
            "revenue_estimated": latest.revenue_estimated,
        }

    def calculate_price_change(
        self,
        symbol: str,
        before_date: str,
        after_date: str
    ) -> Optional[float]:
        """
        Calculate percentage price change between two dates.

        Used for labeling training data.

        Args:
            symbol: Stock ticker symbol
            before_date: Date before earnings (YYYY-MM-DD)
            after_date: Date after earnings (YYYY-MM-DD)

        Returns:
            Percentage change or None
        """
        before_price = self.get_price_on_date(symbol, before_date)
        after_price = self.get_price_on_date(symbol, after_date)

        if not before_price or not after_price:
            return None

        if before_price.close == 0:
            return None

        return ((after_price.close - before_price.close) / before_price.close) * 100

    # ===== Social Sentiment Functions (replaces Reddit) =====

    def get_social_sentiment(self, symbol: str) -> Optional[SocialSentiment]:
        """
        Get social media sentiment from FMP.
        Aggregates sentiment from Twitter, StockTwits, and other platforms.

        This replaces Reddit data for the 25% social component of Golden Triangle.

        Args:
            symbol: Stock ticker symbol

        Returns:
            SocialSentiment object or None
        """
        # v4 API: historical-social-sentiment
        params = {"symbol": symbol.upper()}
        data = self._request("historical-social-sentiment", params, version="v4")

        if not data or len(data) == 0:
            logger.warning(f"No social sentiment found for {symbol}")
            return None

        # FMP returns a list, get first (most recent) item
        sentiment_data = data[0] if isinstance(data, list) else data
        return SocialSentiment.model_validate(sentiment_data)

    def get_social_sentiment_historical(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[SocialSentiment]:
        """
        Get historical social sentiment data.
        Useful for training data collection.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum records to fetch

        Returns:
            List of SocialSentiment objects
        """
        # v4 API: historical-social-sentiment
        params = {"symbol": symbol.upper(), "limit": limit}
        data = self._request("historical-social-sentiment", params, version="v4")

        if not data:
            return []

        return [SocialSentiment.model_validate(item) for item in data]

    def get_stock_sentiment_rss(
        self,
        symbol: str,
        limit: int = 50
    ) -> List[SentimentRSS]:
        """
        Get sentiment from news/RSS feeds.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum articles to fetch

        Returns:
            List of SentimentRSS objects
        """
        # v4 API: stock-news-sentiments/rss-feed
        params = {"page": 0, "tickers": symbol.upper()}
        data = self._request("stock-news-sentiments/rss-feed", params, version="v4")

        if not data:
            return []

        return [SentimentRSS.model_validate(item) for item in data]

    def get_trending_social(
        self,
        sentiment_type: str = "bullish",
        source: str = "stocktwits"
    ) -> List[Dict[str, Any]]:
        """
        Get trending stocks by social sentiment.

        Args:
            sentiment_type: "bullish" or "bearish"
            source: "stocktwits" or "twitter"

        Returns:
            List of trending stock data
        """
        # New stable API endpoint
        params = {"type": sentiment_type, "source": source}
        data = self._request("social-sentiments-trending", params)

        return data if data else []

    def get_social_sentiment_changes(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get social sentiment changes for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with sentiment change data
        """
        # New stable API endpoint
        params = {"symbol": symbol.upper()}
        data = self._request("social-sentiments-change", params)

        if not data or len(data) == 0:
            return None

        return data[0] if isinstance(data, list) else data
