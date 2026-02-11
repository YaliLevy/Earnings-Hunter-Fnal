"""
Feature pipeline for The Earnings Hunter.

Orchestrates all feature extraction from the Golden Triangle sources:
- Financial (40%)
- CEO Tone (35%)
- Social (25%)
"""

from typing import Dict, List, Optional, Any
import pandas as pd

from src.utils.logger import get_logger
from src.feature_engineering.financial_features import FinancialFeatureExtractor
from src.feature_engineering.transcript_analyzer import TranscriptAnalyzer
from src.feature_engineering.social_features import SocialFeatureExtractor
from src.data_ingestion.validators import (
    EarningsData,
    FinancialStatement,
    AnalystEstimate,
    InsiderTransaction,
    InstitutionalHolder,
    PriceTarget,
    StockNews,
)

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Orchestrates feature extraction from all Golden Triangle sources.

    Weights:
    - Financial (Hard Data): 40%
    - CEO Tone (Soft Data): 35%
    - Social (Street Psychology): 25%
    """

    def __init__(self):
        """Initialize feature pipeline with all extractors."""
        self.financial_extractor = FinancialFeatureExtractor()
        self.transcript_analyzer = TranscriptAnalyzer()
        self.social_extractor = SocialFeatureExtractor()

        # Golden Triangle weights
        self.weights = {
            "financial": 0.40,
            "ceo_tone": 0.35,
            "social": 0.25
        }

    def extract_financial_features(
        self,
        earnings: Optional[EarningsData] = None,
        statements: Optional[List[FinancialStatement]] = None,
        estimates: Optional[List[AnalystEstimate]] = None,
        price_target: Optional[PriceTarget] = None,
        insiders: Optional[List[InsiderTransaction]] = None,
        institutions: Optional[List[InstitutionalHolder]] = None,
        current_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extract financial features (40% weight).

        Args:
            earnings: Earnings data
            statements: Financial statements
            estimates: Analyst estimates
            price_target: Price target consensus
            insiders: Insider transactions
            institutions: Institutional holders
            current_price: Current stock price

        Returns:
            Dict of financial features
        """
        return self.financial_extractor.extract_all(
            earnings=earnings,
            statements=statements,
            estimates=estimates,
            price_target=price_target,
            insiders=insiders,
            institutions=institutions,
            current_price=current_price
        )

    def extract_ceo_tone_features(
        self,
        transcript: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract CEO tone features from transcript (35% weight).

        Args:
            transcript: Full earnings call transcript

        Returns:
            Dict of CEO tone features
        """
        if not transcript:
            logger.warning("No transcript provided for CEO tone analysis")
            return {}

        return self.transcript_analyzer.extract_features(transcript)

    def extract_social_features(
        self,
        stock_news: Optional[List[StockNews]] = None
    ) -> Dict[str, float]:
        """
        Extract social features from FMP stock news (25% weight).

        Args:
            stock_news: List of StockNews articles

        Returns:
            Dict of social features
        """
        return self.social_extractor.extract_features(
            stock_news=stock_news
        )

    def extract_all_features(
        self,
        # Financial data
        earnings: Optional[EarningsData] = None,
        statements: Optional[List[FinancialStatement]] = None,
        estimates: Optional[List[AnalystEstimate]] = None,
        price_target: Optional[PriceTarget] = None,
        insiders: Optional[List[InsiderTransaction]] = None,
        institutions: Optional[List[InstitutionalHolder]] = None,
        current_price: Optional[float] = None,
        # Transcript
        transcript: Optional[str] = None,
        # Stock news (for sentiment)
        stock_news: Optional[List[StockNews]] = None,
        # Metadata
        symbol: Optional[str] = None,
        earnings_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract all features from all sources.

        This is the main entry point for feature extraction.

        Args:
            earnings: Earnings data
            statements: Financial statements
            estimates: Analyst estimates
            price_target: Price target consensus
            insiders: Insider transactions
            institutions: Institutional holders
            current_price: Current stock price
            transcript: Earnings call transcript
            stock_news: FMP StockNews articles (for sentiment analysis)
            symbol: Stock symbol (for metadata)
            earnings_date: Earnings date (for metadata)

        Returns:
            Dict containing all features and metadata
        """
        features = {}

        # Metadata
        if symbol:
            features["symbol"] = symbol
        if earnings_date:
            features["earnings_date"] = earnings_date

        # Extract financial features (40%)
        logger.info("Extracting financial features...")
        financial_features = self.extract_financial_features(
            earnings=earnings,
            statements=statements,
            estimates=estimates,
            price_target=price_target,
            insiders=insiders,
            institutions=institutions,
            current_price=current_price
        )
        features.update(financial_features)

        # Extract CEO tone features (35%)
        logger.info("Extracting CEO tone features...")
        ceo_features = self.extract_ceo_tone_features(transcript)
        features.update(ceo_features)

        # Extract social features (25%) - from FMP stock news
        logger.info("Extracting social features from FMP stock news...")
        social_features = self.extract_social_features(stock_news)
        features.update(social_features)

        # Calculate component scores
        financial_score = self.financial_extractor.calculate_financial_score(features)
        ceo_score = features.get("ceo_score", 5.0)
        social_score = self.social_extractor.calculate_social_score(features)

        # Add scores
        features["score_financial"] = financial_score
        features["score_ceo_tone"] = ceo_score
        features["score_social"] = social_score

        # Calculate weighted composite score
        composite_score = (
            financial_score * self.weights["financial"] +
            ceo_score * self.weights["ceo_tone"] +
            social_score * self.weights["social"]
        )
        features["score_composite"] = composite_score

        logger.info(f"Extracted {len(features)} total features")
        return features

    def get_ml_features(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get only numeric features suitable for ML model.

        Filters out metadata and non-numeric values.

        Args:
            features: Full features dict

        Returns:
            Dict of numeric features only
        """
        ml_features = {}

        for key, value in features.items():
            # Skip metadata
            if key in ["symbol", "earnings_date"]:
                continue

            # Only include numeric values
            if isinstance(value, (int, float)):
                ml_features[key] = float(value)

        return ml_features

    def features_to_dataframe(
        self,
        features: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Convert features to pandas DataFrame.

        Args:
            features: Features dict

        Returns:
            Single-row DataFrame
        """
        ml_features = self.get_ml_features(features)
        return pd.DataFrame([ml_features])

    def get_golden_triangle_scores(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get Golden Triangle component scores.

        Args:
            features: Full features dict

        Returns:
            Dict with component scores and weighted total
        """
        financial_score = features.get("score_financial", 5.0)
        ceo_score = features.get("score_ceo_tone", 5.0)
        social_score = features.get("score_social", 5.0)
        composite = features.get("score_composite", 5.0)

        return {
            "financial": {
                "score": financial_score,
                "weight": self.weights["financial"],
                "weighted_score": financial_score * self.weights["financial"],
                "label": "Hard Data (FMP)"
            },
            "ceo_tone": {
                "score": ceo_score,
                "weight": self.weights["ceo_tone"],
                "weighted_score": ceo_score * self.weights["ceo_tone"],
                "label": "CEO Tone (Transcript)"
            },
            "social": {
                "score": social_score,
                "weight": self.weights["social"],
                "weighted_score": social_score * self.weights["social"],
                "label": "Street Psychology (FMP Social)"
            },
            "composite": {
                "score": composite,
                "label": "Weighted Total"
            }
        }

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names for ML model.

        Returns:
            List of feature column names
        """
        # Create dummy data to get feature names
        dummy_features = self.extract_all_features()
        ml_features = self.get_ml_features(dummy_features)
        return sorted(ml_features.keys())

    def validate_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate features for ML model input.

        Args:
            features: Features dict

        Returns:
            Validation result with any issues
        """
        expected_features = self.get_feature_names()
        actual_features = set(features.keys())
        expected_set = set(expected_features)

        missing = expected_set - actual_features
        extra = actual_features - expected_set

        # Check for invalid values
        invalid_values = {}
        for key, value in features.items():
            if value is None:
                invalid_values[key] = "None value"
            elif not isinstance(value, (int, float)):
                invalid_values[key] = f"Invalid type: {type(value)}"
            elif pd.isna(value):
                invalid_values[key] = "NaN value"

        return {
            "valid": len(missing) == 0 and len(invalid_values) == 0,
            "missing_features": list(missing),
            "extra_features": list(extra),
            "invalid_values": invalid_values,
            "feature_count": len(actual_features)
        }
