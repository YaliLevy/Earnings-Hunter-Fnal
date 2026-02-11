"""
Pydantic validators for data models.

Defines data structures for all API responses and internal data types.
"""

from datetime import datetime
from typing import Optional, List, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class EarningsData(BaseModel):
    """Earnings calendar and surprise data from FMP."""

    symbol: str
    date: str
    # New stable API uses epsActual/revenueActual
    eps: Optional[float] = Field(None, alias="epsActual")
    eps_estimated: Optional[float] = Field(None, alias="epsEstimated")
    revenue: Optional[float] = Field(None, alias="revenueActual")
    revenue_estimated: Optional[float] = Field(None, alias="revenueEstimated")
    fiscal_date_ending: Optional[str] = Field(None, alias="fiscalDateEnding")
    time: Optional[str] = None  # "bmo" (before market open) or "amc" (after market close)
    last_updated: Optional[str] = Field(None, alias="lastUpdated")

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    @property
    def eps_surprise(self) -> Optional[float]:
        """Calculate EPS surprise percentage."""
        if self.eps is not None and self.eps_estimated is not None and self.eps_estimated != 0:
            return (self.eps - self.eps_estimated) / abs(self.eps_estimated)
        return None

    @property
    def revenue_surprise(self) -> Optional[float]:
        """Calculate revenue surprise percentage."""
        if self.revenue is not None and self.revenue_estimated is not None and self.revenue_estimated != 0:
            return (self.revenue - self.revenue_estimated) / self.revenue_estimated
        return None

    @property
    def beat_eps(self) -> Optional[bool]:
        """Check if company beat EPS estimates."""
        surprise = self.eps_surprise
        return surprise > 0 if surprise is not None else None

    @property
    def beat_revenue(self) -> Optional[bool]:
        """Check if company beat revenue estimates."""
        surprise = self.revenue_surprise
        return surprise > 0 if surprise is not None else None

    model_config = {"populate_by_name": True}


class EarningsTranscript(BaseModel):
    """Earnings call transcript from FMP."""

    symbol: str
    year: int
    quarter: int
    date: str
    content: str

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    @field_validator("quarter")
    @classmethod
    def validate_quarter(cls, v: int) -> int:
        """Validate quarter is 1-4."""
        if v not in [1, 2, 3, 4]:
            raise ValueError(f"Quarter must be 1-4, got {v}")
        return v

    @property
    def fiscal_period(self) -> str:
        """Return fiscal period string (e.g., 'Q3 2025')."""
        return f"Q{self.quarter} {self.year}"


class FinancialStatement(BaseModel):
    """Income statement data from FMP."""

    date: str
    symbol: str
    reported_currency: Optional[str] = Field(None, alias="reportedCurrency")
    cik: Optional[str] = None
    filling_date: Optional[str] = Field(None, alias="fillingDate")
    accepted_date: Optional[str] = Field(None, alias="acceptedDate")
    calendar_year: Optional[str] = Field(None, alias="calendarYear")
    period: Optional[str] = None

    # Revenue & Income
    revenue: float = 0
    cost_of_revenue: Optional[float] = Field(None, alias="costOfRevenue")
    gross_profit: Optional[float] = Field(None, alias="grossProfit")
    gross_profit_ratio: Optional[float] = Field(None, alias="grossProfitRatio")

    # Operating
    operating_expenses: Optional[float] = Field(None, alias="operatingExpenses")
    operating_income: Optional[float] = Field(None, alias="operatingIncome")
    operating_income_ratio: Optional[float] = Field(None, alias="operatingIncomeRatio")

    # Net Income
    net_income: float = Field(0, alias="netIncome")
    net_income_ratio: Optional[float] = Field(None, alias="netIncomeRatio")

    # EPS
    eps: float = 0
    eps_diluted: Optional[float] = Field(None, alias="epsdiluted")

    # Other
    ebitda: Optional[float] = None
    ebitda_ratio: Optional[float] = Field(None, alias="ebitdaratio")

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    @property
    def gross_margin(self) -> Optional[float]:
        """Calculate gross margin."""
        if self.gross_profit and self.revenue and self.revenue != 0:
            return self.gross_profit / self.revenue
        return self.gross_profit_ratio

    @property
    def operating_margin(self) -> Optional[float]:
        """Calculate operating margin."""
        if self.operating_income and self.revenue and self.revenue != 0:
            return self.operating_income / self.revenue
        return self.operating_income_ratio

    @property
    def net_margin(self) -> Optional[float]:
        """Calculate net margin."""
        if self.revenue and self.revenue != 0:
            return self.net_income / self.revenue
        return self.net_income_ratio

    model_config = {"populate_by_name": True}


class AnalystEstimate(BaseModel):
    """Analyst estimates from FMP (new stable API format)."""

    symbol: str
    date: str

    # EPS Estimates (new API uses epsAvg instead of estimatedEpsAvg)
    estimated_eps_avg: Optional[float] = Field(None, alias="epsAvg")
    estimated_eps_high: Optional[float] = Field(None, alias="epsHigh")
    estimated_eps_low: Optional[float] = Field(None, alias="epsLow")
    number_analyst_estimated_eps: Optional[int] = Field(None, alias="numberAnalystEstimatedEps")

    # Revenue Estimates (new API uses revenueAvg instead of estimatedRevenueAvg)
    estimated_revenue_avg: Optional[float] = Field(None, alias="revenueAvg")
    estimated_revenue_high: Optional[float] = Field(None, alias="revenueHigh")
    estimated_revenue_low: Optional[float] = Field(None, alias="revenueLow")
    number_analyst_estimated_revenue: Optional[int] = Field(None, alias="numberAnalystEstimatedRevenue")

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    model_config = {"populate_by_name": True}


class PriceTarget(BaseModel):
    """Analyst price target consensus from FMP."""

    symbol: str
    target_high: Optional[float] = Field(None, alias="targetHigh")
    target_low: Optional[float] = Field(None, alias="targetLow")
    target_consensus: Optional[float] = Field(None, alias="targetConsensus")
    target_median: Optional[float] = Field(None, alias="targetMedian")

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    model_config = {"populate_by_name": True}


class InsiderTransaction(BaseModel):
    """Insider trading transaction from FMP."""

    symbol: str
    transaction_date: str = Field(alias="transactionDate")
    transaction_type: str = Field(alias="transactionType")  # "P-Purchase" or "S-Sale"
    securities_transacted: float = Field(alias="securitiesTransacted")
    price: Optional[float] = None
    reporting_name: str = Field(alias="reportingName")
    type_of_owner: str = Field(alias="typeOfOwner")
    form_type: Optional[str] = Field(None, alias="formType")
    acquisition_or_disposition: Optional[str] = Field(None, alias="acquisitionOrDisposition")  # "A" or "D"

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    @property
    def is_purchase(self) -> bool:
        """Check if transaction is a purchase."""
        return "P" in self.transaction_type.upper() or "BUY" in self.transaction_type.upper()

    @property
    def is_sale(self) -> bool:
        """Check if transaction is a sale."""
        return "S" in self.transaction_type.upper() or "SELL" in self.transaction_type.upper()

    @property
    def transaction_value(self) -> Optional[float]:
        """Calculate transaction value."""
        if self.price:
            return self.securities_transacted * self.price
        return None

    model_config = {"populate_by_name": True}


class InstitutionalHolder(BaseModel):
    """Institutional holder data from FMP."""

    holder: str
    shares: int
    date_reported: str = Field(alias="dateReported")
    change: Optional[int] = None
    change_percentage: Optional[float] = Field(None, alias="changePercentage")

    model_config = {"populate_by_name": True}


class StockQuote(BaseModel):
    """Stock quote data from FMP."""

    symbol: str
    name: Optional[str] = None
    price: float
    change: Optional[float] = None
    change_percentage: Optional[float] = Field(None, alias="changesPercentage")
    day_low: Optional[float] = Field(None, alias="dayLow")
    day_high: Optional[float] = Field(None, alias="dayHigh")
    year_low: Optional[float] = Field(None, alias="yearLow")
    year_high: Optional[float] = Field(None, alias="yearHigh")
    market_cap: Optional[float] = Field(None, alias="marketCap")
    volume: Optional[int] = None
    avg_volume: Optional[int] = Field(None, alias="avgVolume")
    open_price: Optional[float] = Field(None, alias="open")
    previous_close: Optional[float] = Field(None, alias="previousClose")
    pe: Optional[float] = None
    eps: Optional[float] = None

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    model_config = {"populate_by_name": True}


class HistoricalPrice(BaseModel):
    """Historical price data from FMP."""

    date: str
    open_price: float = Field(alias="open")
    high: float
    low: float
    close: float
    adj_close: Optional[float] = Field(None, alias="adjClose")
    volume: int
    unadjusted_volume: Optional[int] = Field(None, alias="unadjustedVolume")
    change: Optional[float] = None
    change_percent: Optional[float] = Field(None, alias="changePercent")

    model_config = {"populate_by_name": True}


class SocialSentiment(BaseModel):
    """
    FMP Social Sentiment data.

    Aggregates sentiment from Twitter, StockTwits, and other platforms.
    This replaces Reddit data for the 25% social component of Golden Triangle.
    """

    symbol: str
    date: str

    # StockTwits metrics
    stocktwits_posts: int = Field(0, alias="stocktwitsPosts")
    stocktwits_comments: int = Field(0, alias="stocktwitsComments")
    stocktwits_likes: int = Field(0, alias="stocktwitsLikes")
    stocktwits_impressions: int = Field(0, alias="stocktwitsImpressions")
    stocktwits_sentiment: float = Field(0.5, alias="stocktwitsSentiment")

    # Twitter metrics
    twitter_posts: int = Field(0, alias="twitterPosts")
    twitter_comments: int = Field(0, alias="twitterComments")
    twitter_likes: int = Field(0, alias="twitterLikes")
    twitter_impressions: int = Field(0, alias="twitterImpressions")
    twitter_sentiment: float = Field(0.5, alias="twitterSentiment")

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    @property
    def total_posts(self) -> int:
        """Total posts across all platforms."""
        return self.stocktwits_posts + self.twitter_posts

    @property
    def total_engagement(self) -> int:
        """Total engagement (comments + likes)."""
        return (
            self.stocktwits_comments + self.stocktwits_likes +
            self.twitter_comments + self.twitter_likes
        )

    @property
    def total_impressions(self) -> int:
        """Total impressions across platforms."""
        return self.stocktwits_impressions + self.twitter_impressions

    @property
    def combined_sentiment(self) -> float:
        """
        Weighted average sentiment.
        Twitter weighted higher due to typically larger volume.
        """
        return 0.4 * self.stocktwits_sentiment + 0.6 * self.twitter_sentiment

    @property
    def engagement_rate(self) -> float:
        """Engagement per post."""
        if self.total_posts == 0:
            return 0.0
        return self.total_engagement / self.total_posts

    model_config = {"populate_by_name": True}


class SentimentRSS(BaseModel):
    """FMP News Sentiment RSS feed data."""

    symbol: str
    published_date: str = Field(alias="publishedDate")
    title: str
    text: str = ""
    sentiment: str = "Neutral"  # "Bullish", "Bearish", "Neutral"
    sentiment_score: float = Field(0.0, alias="sentimentScore")
    url: Optional[str] = None
    site: Optional[str] = None

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.sentiment.lower() == "bullish"

    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.sentiment.lower() == "bearish"

    model_config = {"populate_by_name": True}


class StockNews(BaseModel):
    """Stock news article from FMP."""

    symbol: str
    published_date: str = Field(alias="publishedDate")
    title: str
    text: str
    url: str
    site: Optional[str] = None

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        if v:
            return v.upper().strip()
        return v

    model_config = {"populate_by_name": True}
