"""
Social Sentiment Features from FMP Stock News.

Extracts the 25% "Street Psychology" component of the Golden Triangle
using FMP's stock news data with NLP sentiment analysis.

This replaces the broken social-sentiment API endpoint.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from src.utils.logger import get_logger
from src.feature_engineering.sentiment_features import SentimentFeatureExtractor
from src.data_ingestion.validators import StockNews

logger = get_logger(__name__)


class SocialFeatureExtractor:
    """
    Extract social sentiment features from FMP stock news.

    Features contribute 25% to the Golden Triangle weighted score.
    Uses NLP (VADER + TextBlob) to analyze news sentiment.
    """

    def __init__(self):
        """Initialize the social feature extractor."""
        self.sentiment_analyzer = SentimentFeatureExtractor()

    def extract_features(
        self,
        stock_news: Optional[List[StockNews]] = None
    ) -> Dict[str, float]:
        """
        Extract all social sentiment features from stock news.

        Args:
            stock_news: List of StockNews from FMP get_stock_news()

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Extract news sentiment features
        features.update(self._extract_news_sentiment_features(stock_news))

        # Calculate hype index
        features["social_hype_index"] = self._calculate_hype_index(features)

        return features

    def _extract_news_sentiment_features(
        self,
        stock_news: Optional[List[StockNews]]
    ) -> Dict[str, float]:
        """
        Extract sentiment features from stock news using NLP.

        Args:
            stock_news: List of StockNews objects

        Returns:
            Dictionary of news sentiment features
        """
        if not stock_news or len(stock_news) == 0:
            return {
                "social_total_posts": 0,
                "social_total_engagement": 0,
                "social_total_impressions": 0,
                "social_engagement_rate": 0,
                "social_virality": 0,
                "social_sentiment_stocktwits": 0.5,
                "social_sentiment_twitter": 0.5,
                "social_sentiment_combined": 0.5,
                "social_sentiment_agreement": 1.0,
                "social_stocktwits_ratio": 0.5,
                "news_sentiment_mean": 0.0,
                "news_sentiment_std": 0.0,
                "news_bullish_ratio": 0.5,
                "news_bearish_ratio": 0.5,
                "news_volume": 0,
            }

        # Analyze sentiment from news titles and text
        texts = []
        for news in stock_news:
            text = f"{news.title}. {news.text}" if news.text else news.title
            texts.append(text)

        # Use sentiment extractor to analyze news
        sentiment_features = self.sentiment_analyzer.analyze_texts(texts)

        # Get combined sentiment (-1 to 1)
        combined = sentiment_features.get("combined_sentiment", 0)

        # Convert to 0-1 scale for compatibility
        sentiment_normalized = (combined + 1) / 2

        features = {
            # Volume features (using news count as proxy)
            "social_total_posts": float(len(stock_news)),
            "social_total_engagement": float(len(stock_news) * 10),  # Estimate
            "social_total_impressions": float(len(stock_news) * 100),  # Estimate
            "social_engagement_rate": 10.0,  # Default engagement rate
            "social_virality": 100.0,  # Default virality

            # Sentiment features (using news sentiment as proxy)
            "social_sentiment_stocktwits": sentiment_normalized,
            "social_sentiment_twitter": sentiment_normalized,
            "social_sentiment_combined": sentiment_normalized,
            "social_sentiment_agreement": 1.0,  # Single source, perfect agreement
            "social_stocktwits_ratio": 0.5,  # Not applicable

            # News-specific features
            "news_sentiment_mean": combined,
            "news_sentiment_std": sentiment_features.get("vader_compound_std", 0.0),
            "news_bullish_ratio": sentiment_features.get("positive_ratio", 0.5),
            "news_bearish_ratio": sentiment_features.get("negative_ratio", 0.5),
            "news_volume": float(len(stock_news)),
        }

        return features

    def _calculate_hype_index(
        self,
        features: Dict[str, float]
    ) -> float:
        """
        Calculate composite hype index (0-100).

        Combines:
        - Volume score (30%)
        - Sentiment intensity (30%)
        - Sentiment positivity (40%)

        High hype (>70) may indicate speculative excess (risk).

        Args:
            features: Dictionary of extracted features

        Returns:
            Hype index from 0 to 100
        """
        # Volume score (0-30)
        # Normalize: assume 50 news articles is high activity
        volume = features.get("news_volume", 0)
        volume_normalized = min(volume / 50, 1.0)
        volume_score = volume_normalized * 30

        # Sentiment intensity score (0-30)
        # Based on variance/std - high variance = mixed signals
        std = features.get("news_sentiment_std", 0)
        intensity_normalized = min(std / 0.5, 1.0)  # 0.5 std is high variance
        intensity_score = (1 - intensity_normalized) * 30  # Lower variance = higher score

        # Sentiment positivity score (0-40)
        # Use combined sentiment (0-1 scale)
        sentiment = features.get("social_sentiment_combined", 0.5)
        sentiment_score = sentiment * 40

        # Combine
        hype_index = volume_score + intensity_score + sentiment_score

        return min(max(hype_index, 0), 100)

    def calculate_social_score(self, features: Dict[str, float]) -> float:
        """
        Calculate overall social score (0-10) for UI display.

        This is the 25% component of the Golden Triangle.

        Args:
            features: Dictionary of extracted features

        Returns:
            Score from 0 to 10
        """
        hype = features.get("social_hype_index", 50.0) / 100  # Normalize to 0-1
        sentiment = features.get("social_sentiment_combined", 0.5)
        bullish_ratio = features.get("news_bullish_ratio", 0.5)

        # Weighted combination
        score = (0.4 * hype + 0.4 * sentiment + 0.2 * bullish_ratio) * 10

        return round(min(max(score, 0), 10), 1)

    def get_interpretation(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate human-readable interpretation of social features.

        Args:
            features: Dictionary of extracted features

        Returns:
            Interpretation dictionary
        """
        hype_index = features.get("social_hype_index", 50.0)
        sentiment_combined = features.get("social_sentiment_combined", 0.5)
        news_bullish = features.get("news_bullish_ratio", 0.5)
        news_volume = features.get("news_volume", 0)

        # Hype level
        if hype_index >= 70:
            hype_level = "High"
            hype_warning = "High news volume and positive sentiment detected"
        elif hype_index >= 40:
            hype_level = "Moderate"
            hype_warning = None
        else:
            hype_level = "Low"
            hype_warning = "Limited news coverage"

        # Sentiment interpretation
        if sentiment_combined >= 0.7:
            sentiment_label = "Very Bullish"
        elif sentiment_combined >= 0.55:
            sentiment_label = "Bullish"
        elif sentiment_combined >= 0.45:
            sentiment_label = "Neutral"
        elif sentiment_combined >= 0.3:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Very Bearish"

        return {
            "hype_level": hype_level,
            "hype_index": hype_index,
            "hype_warning": hype_warning,
            "sentiment_label": sentiment_label,
            "sentiment_combined": sentiment_combined,
            "news_volume": news_volume,
            "news_bullish_pct": f"{news_bullish:.0%}",
            "source": "FMP Stock News (NLP analyzed)"
        }


# Convenience functions for backward compatibility

def extract_social_features(
    stock_news: Optional[List[Any]] = None
) -> Dict[str, float]:
    """
    Extract social features from stock news.

    Convenience function for backward compatibility.

    Args:
        stock_news: List of StockNews objects or raw dicts

    Returns:
        Dictionary of features
    """
    extractor = SocialFeatureExtractor()

    # Convert dicts to Pydantic models if needed
    if stock_news and len(stock_news) > 0:
        if isinstance(stock_news[0], dict):
            stock_news = [StockNews.model_validate(item) for item in stock_news]

    return extractor.extract_features(stock_news)


def calculate_social_score(features: Dict[str, float]) -> float:
    """
    Calculate social score from features.

    Convenience function for backward compatibility.

    Args:
        features: Extracted features dict

    Returns:
        Score from 0 to 10
    """
    extractor = SocialFeatureExtractor()
    return extractor.calculate_social_score(features)
