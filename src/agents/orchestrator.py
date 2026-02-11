"""
Earnings analysis orchestrator.

Smart orchestrator that:
1. Identifies the most recent earnings report for a ticker
2. Fetches all relevant data automatically
3. Coordinates feature extraction and analysis
4. Generates research insights
"""

from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger
from src.utils.cache import AnalysisCache
from src.data_ingestion.fmp_client import FMPClient
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.feature_engineering.transcript_analyzer import TranscriptAnalyzer
from src.feature_engineering.sentiment_features import SentimentFeatureExtractor
from src.ml.predictor import EarningsPredictor
from config.settings import get_settings

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    symbol: str
    earnings_date: str
    year: int
    quarter: int
    prediction: str
    confidence: Optional[float]
    golden_triangle: Dict[str, Any]
    financial_summary: Dict[str, Any]
    ceo_tone_summary: Dict[str, Any]
    social_summary: Dict[str, Any]
    research_insight: Optional[Dict[str, Any]]
    features: Dict[str, float]
    analyzed_at: str
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EarningsOrchestrator:
    """
    Orchestrate the complete earnings analysis pipeline.

    This is the main entry point for analyzing a stock.
    User enters ticker -> Gets complete analysis with research insight.
    """

    def __init__(
        self,
        fmp_api_key: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize orchestrator.

        Args:
            fmp_api_key: FMP API key (defaults to settings)
            use_cache: Whether to use caching
        """
        settings = get_settings()

        # Initialize clients (FMP only - no Reddit)
        self.fmp_client = FMPClient(fmp_api_key or settings.fmp_api_key)

        # Initialize components
        self.feature_pipeline = FeaturePipeline()
        self.transcript_analyzer = TranscriptAnalyzer()
        self.predictor = EarningsPredictor()
        self.sentiment_extractor = SentimentFeatureExtractor()

        # Initialize cache
        self.use_cache = use_cache
        if use_cache:
            self.cache = AnalysisCache(
                cache_dir=settings.cache_directory,
                expiry_hours=settings.cache_expiry_hours
            )
        else:
            self.cache = None

        self.disclaimer = (
            "This analysis is for educational and informational purposes only. "
            "It is NOT financial advice. Always consult a licensed professional "
            "before making investment decisions."
        )

    def analyze_latest_earnings(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> AnalysisResult:
        """
        Analyze the latest earnings for a ticker.

        This is the main entry point. User provides ticker,
        system automatically finds and analyzes the latest earnings.

        Args:
            symbol: Stock ticker symbol (e.g., "NVDA")
            force_refresh: Ignore cache and run fresh analysis

        Returns:
            Complete AnalysisResult
        """
        symbol = symbol.upper().strip()
        logger.info(f"Starting analysis for {symbol}...")

        # Check cache
        if self.use_cache and not force_refresh:
            cached = self.cache.get(symbol)
            if cached:
                logger.info(f"Using cached analysis for {symbol}")
                return AnalysisResult(**cached)

        # Step 1: Find latest earnings
        logger.info("Finding latest earnings...")
        earnings_info = self.fmp_client.get_latest_earnings_info(symbol)

        if not earnings_info:
            raise ValueError(f"No earnings data found for {symbol}")

        year = earnings_info["year"]
        quarter = earnings_info["quarter"]
        earnings_date = earnings_info["earnings_date"]

        logger.info(f"Latest earnings: Q{quarter} {year} ({earnings_date})")

        # Step 2: Fetch all data (using FMP social instead of Reddit)
        logger.info("Fetching financial data...")
        data = self._fetch_all_data(symbol, year, quarter)

        # Step 3: Extract features
        logger.info("Extracting features...")
        features = self._extract_features(symbol, earnings_date, data)

        # Step 4: Analyze CEO tone
        logger.info("Analyzing CEO tone...")
        ceo_analysis = None
        if data.get("transcript"):
            ceo_analysis = self.transcript_analyzer.calculate_ceo_sentiment_score(
                data["transcript"]
            )

        # Step 5: Make prediction
        logger.info("Making prediction...")
        try:
            ml_features = self.feature_pipeline.get_ml_features(features)
            prediction_result = self.predictor.predict(ml_features, mode="consensus")
        except Exception as e:
            logger.warning(f"Prediction failed: {e}. Using default.")
            prediction_result = {
                "consensus_prediction": "Stagnation",
                "best_model_confidence": None
            }

        # Step 6: Calculate Golden Triangle scores
        golden_triangle = self.feature_pipeline.get_golden_triangle_scores(features)

        # Step 7: Build summaries
        financial_summary = self._build_financial_summary(data, earnings_info)
        ceo_summary = self._build_ceo_summary(ceo_analysis, data.get("transcript"))
        social_summary = self._build_social_summary(data.get("stock_news", []))

        # Step 8: Create result
        result = AnalysisResult(
            symbol=symbol,
            earnings_date=earnings_date,
            year=year,
            quarter=quarter,
            prediction=prediction_result.get("consensus_prediction", "Unknown"),
            confidence=prediction_result.get("best_model_confidence"),
            golden_triangle=golden_triangle,
            financial_summary=financial_summary,
            ceo_tone_summary=ceo_summary,
            social_summary=social_summary,
            research_insight=None,  # Will be filled by InsightGenerator
            features=features,
            analyzed_at=datetime.now().isoformat(),
            disclaimer=self.disclaimer
        )

        # Cache result
        if self.use_cache:
            self.cache.set(symbol, result.to_dict())

        logger.info(f"Analysis complete for {symbol}: {result.prediction}")
        return result

    def _fetch_all_data(
        self,
        symbol: str,
        year: int,
        quarter: int
    ) -> Dict[str, Any]:
        """Fetch all required data."""
        data = {}

        # Financial data
        try:
            data["earnings"] = self.fmp_client.get_earnings_surprises(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch earnings: {e}")
            data["earnings"] = []

        try:
            data["statements"] = self.fmp_client.get_income_statement(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch statements: {e}")
            data["statements"] = []

        try:
            data["estimates"] = self.fmp_client.get_analyst_estimates(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch estimates: {e}")
            data["estimates"] = []

        try:
            data["price_target"] = self.fmp_client.get_price_target(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch price target: {e}")
            data["price_target"] = None

        try:
            data["insiders"] = self.fmp_client.get_insider_trading(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch insider data: {e}")
            data["insiders"] = []

        # Institutional holders - endpoint not available (403)
        data["institutions"] = []

        try:
            data["quote"] = self.fmp_client.get_stock_quote(symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch quote: {e}")
            data["quote"] = None

        # Transcript (CRITICAL for CEO tone)
        try:
            transcript = self.fmp_client.get_earnings_call_transcript(symbol, year, quarter)
            data["transcript"] = transcript.content if transcript else None
        except Exception as e:
            logger.warning(f"Failed to fetch transcript: {e}")
            data["transcript"] = None

        # Stock news (replaces social sentiment API)
        try:
            data["stock_news"] = self.fmp_client.get_stock_news(symbol, limit=50)
        except Exception as e:
            logger.warning(f"Failed to fetch stock news: {e}")
            data["stock_news"] = []

        return data

    def _extract_features(
        self,
        symbol: str,
        earnings_date: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract features from fetched data."""
        current_price = data["quote"].price if data.get("quote") else None

        # Get first earnings entry
        earnings = data["earnings"][0] if data.get("earnings") else None

        return self.feature_pipeline.extract_all_features(
            earnings=earnings,
            statements=data.get("statements"),
            estimates=data.get("estimates"),
            price_target=data.get("price_target"),
            insiders=data.get("insiders"),
            institutions=data.get("institutions"),
            current_price=current_price,
            transcript=data.get("transcript"),
            stock_news=data.get("stock_news"),
            symbol=symbol,
            earnings_date=earnings_date
        )

    def _build_financial_summary(
        self,
        data: Dict[str, Any],
        earnings_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build financial summary."""
        eps = earnings_info.get("eps")
        eps_est = earnings_info.get("eps_estimated")
        revenue = earnings_info.get("revenue")
        revenue_est = earnings_info.get("revenue_estimated")

        eps_beat = eps > eps_est if eps and eps_est else None
        revenue_beat = revenue > revenue_est if revenue and revenue_est else None

        return {
            "eps_actual": eps,
            "eps_estimated": eps_est,
            "eps_surprise_pct": ((eps - eps_est) / abs(eps_est) * 100)
            if eps and eps_est and eps_est != 0 else None,
            "eps_beat": eps_beat,
            "revenue_actual": revenue,
            "revenue_estimated": revenue_est,
            "revenue_surprise_pct": ((revenue - revenue_est) / revenue_est * 100)
            if revenue and revenue_est and revenue_est != 0 else None,
            "revenue_beat": revenue_beat,
            "double_beat": eps_beat and revenue_beat if eps_beat is not None else None,
            "current_price": data["quote"].price if data.get("quote") else None,
            "market_cap": data["quote"].market_cap if data.get("quote") else None,
        }

    def _build_ceo_summary(
        self,
        ceo_analysis: Optional[Dict[str, Any]],
        transcript: Optional[str]
    ) -> Dict[str, Any]:
        """Build CEO tone summary."""
        if not ceo_analysis:
            return {
                "available": False,
                "reason": "No transcript available"
            }

        return {
            "available": True,
            "confidence_score": ceo_analysis.get("ceo_confidence_score"),
            "sentiment_score": ceo_analysis.get("ceo_overall_sentiment"),
            "guidance_sentiment": ceo_analysis.get("forward_guidance_sentiment"),
            "uncertainty_ratio": ceo_analysis.get("uncertainty_ratio"),
            "tone_summary": ceo_analysis.get("tone_summary"),
            "ceo_score": ceo_analysis.get("ceo_score"),
            "key_positive_phrases": ceo_analysis.get("key_positive_phrases", [])[:3],
            "key_negative_phrases": ceo_analysis.get("key_negative_phrases", [])[:3],
            "raised_guidance": ceo_analysis.get("guidance_analysis", {}).get("has_raised_guidance"),
            "lowered_guidance": ceo_analysis.get("guidance_analysis", {}).get("has_lowered_guidance"),
        }

    def _build_social_summary(
        self,
        stock_news: list
    ) -> Dict[str, Any]:
        """Build sentiment summary from stock news."""
        if not stock_news:
            return {
                "available": False,
                "reason": "No news data available"
            }

        # Analyze sentiment from news titles and text
        texts = []
        for news in stock_news:
            # Combine title and text for sentiment analysis
            text = f"{news.title}. {news.text}" if hasattr(news, 'text') else news.title
            texts.append(text)

        # Use sentiment extractor to analyze news
        sentiment_features = self.sentiment_extractor.analyze_texts(texts)
        sentiment_summary = self.sentiment_extractor.get_sentiment_summary(sentiment_features)

        # Convert combined sentiment (-1 to 1) to percentage (0 to 100)
        combined = sentiment_features.get("combined_sentiment", 0)
        bullish_pct = (combined + 1) / 2 * 100  # Convert -1..1 to 0..100

        summary = {
            "available": True,
            "source": "FMP Stock News (sentiment analyzed via NLP)",
            "news_count": len(stock_news),
            "combined_sentiment": combined,
            "bullish_percentage": bullish_pct,
            "positive_ratio": sentiment_features.get("positive_ratio", 0.5),
            "negative_ratio": sentiment_features.get("negative_ratio", 0.5),
            "sentiment_classification": sentiment_summary.get("classification", "Neutral"),
            "sentiment_description": sentiment_summary.get("description", ""),
            "confidence": sentiment_summary.get("confidence", 0.5)
        }

        return summary
