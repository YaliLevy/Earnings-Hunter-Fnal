"""
Sentiment analysis features using VADER and TextBlob.

Provides utility functions used by both transcript and social analysis.
"""

from typing import Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentFeatureExtractor:
    """
    Extract sentiment features using VADER and TextBlob.

    VADER is better for social media and informal text.
    TextBlob is better for formal/professional text.
    Combined approach gives more robust results.
    """

    def __init__(self):
        """Initialize sentiment analyzers."""
        self.vader = SentimentIntensityAnalyzer()

    def analyze_text_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze text using VADER sentiment.

        Args:
            text: Text to analyze

        Returns:
            Dict with VADER scores (neg, neu, pos, compound)
        """
        if not text or not text.strip():
            return {
                "neg": 0.0,
                "neu": 1.0,
                "pos": 0.0,
                "compound": 0.0
            }

        return self.vader.polarity_scores(text)

    def analyze_text_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze text using TextBlob sentiment.

        Args:
            text: Text to analyze

        Returns:
            Dict with polarity (-1 to 1) and subjectivity (0 to 1)
        """
        if not text or not text.strip():
            return {
                "polarity": 0.0,
                "subjectivity": 0.0
            }

        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text using both VADER and TextBlob.

        Args:
            text: Text to analyze

        Returns:
            Dict with combined sentiment scores
        """
        vader_scores = self.analyze_text_vader(text)
        textblob_scores = self.analyze_text_textblob(text)

        # Combined sentiment (weighted average)
        # VADER compound is -1 to 1, TextBlob polarity is -1 to 1
        combined = (vader_scores["compound"] * 0.6) + (textblob_scores["polarity"] * 0.4)

        return {
            "vader_compound": vader_scores["compound"],
            "vader_positive": vader_scores["pos"],
            "vader_negative": vader_scores["neg"],
            "vader_neutral": vader_scores["neu"],
            "textblob_polarity": textblob_scores["polarity"],
            "textblob_subjectivity": textblob_scores["subjectivity"],
            "combined_sentiment": combined
        }

    def analyze_texts(self, texts: List[str]) -> Dict[str, float]:
        """
        Analyze multiple texts and aggregate results.

        Args:
            texts: List of texts to analyze

        Returns:
            Dict with aggregated sentiment features
        """
        if not texts:
            return self._empty_features()

        # Analyze each text
        vader_compounds = []
        vader_positives = []
        vader_negatives = []
        textblob_polarities = []
        textblob_subjectivities = []

        for text in texts:
            if not text or not text.strip():
                continue

            vader = self.analyze_text_vader(text)
            textblob = self.analyze_text_textblob(text)

            vader_compounds.append(vader["compound"])
            vader_positives.append(vader["pos"])
            vader_negatives.append(vader["neg"])
            textblob_polarities.append(textblob["polarity"])
            textblob_subjectivities.append(textblob["subjectivity"])

        if not vader_compounds:
            return self._empty_features()

        # Aggregate statistics
        features = {
            # VADER features
            "vader_compound_mean": float(np.mean(vader_compounds)),
            "vader_compound_std": float(np.std(vader_compounds)),
            "vader_compound_min": float(np.min(vader_compounds)),
            "vader_compound_max": float(np.max(vader_compounds)),
            "vader_positive_mean": float(np.mean(vader_positives)),
            "vader_negative_mean": float(np.mean(vader_negatives)),

            # TextBlob features
            "textblob_polarity_mean": float(np.mean(textblob_polarities)),
            "textblob_polarity_std": float(np.std(textblob_polarities)),
            "textblob_subjectivity_mean": float(np.mean(textblob_subjectivities)),

            # Combined sentiment
            "combined_sentiment": float(
                np.mean(vader_compounds) * 0.6 + np.mean(textblob_polarities) * 0.4
            ),

            # Distribution features
            "positive_ratio": float(np.mean([1 if c > 0.05 else 0 for c in vader_compounds])),
            "negative_ratio": float(np.mean([1 if c < -0.05 else 0 for c in vader_compounds])),
            "text_count": float(len(vader_compounds))
        }

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return empty sentiment features."""
        return {
            "vader_compound_mean": 0.0,
            "vader_compound_std": 0.0,
            "vader_compound_min": 0.0,
            "vader_compound_max": 0.0,
            "vader_positive_mean": 0.0,
            "vader_negative_mean": 0.0,
            "textblob_polarity_mean": 0.0,
            "textblob_polarity_std": 0.0,
            "textblob_subjectivity_mean": 0.0,
            "combined_sentiment": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "text_count": 0.0
        }

    def classify_sentiment(self, score: float) -> str:
        """
        Classify sentiment score into category.

        Args:
            score: Sentiment score (-1 to 1)

        Returns:
            Category string
        """
        if score >= 0.5:
            return "Very Positive"
        elif score >= 0.1:
            return "Positive"
        elif score >= -0.1:
            return "Neutral"
        elif score >= -0.5:
            return "Negative"
        else:
            return "Very Negative"

    def get_sentiment_summary(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Generate human-readable sentiment summary.

        Args:
            features: Dict of sentiment features

        Returns:
            Summary dict with classification and description
        """
        combined = features.get("combined_sentiment", 0)
        classification = self.classify_sentiment(combined)

        positive_ratio = features.get("positive_ratio", 0)
        negative_ratio = features.get("negative_ratio", 0)

        return {
            "classification": classification,
            "score": combined,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "confidence": 1 - features.get("vader_compound_std", 0),  # Lower std = higher confidence
            "description": self._generate_description(combined, positive_ratio, negative_ratio)
        }

    def _generate_description(
        self,
        combined: float,
        positive_ratio: float,
        negative_ratio: float
    ) -> str:
        """Generate text description of sentiment."""
        classification = self.classify_sentiment(combined)

        if classification == "Very Positive":
            return "Sentiment is strongly positive with widespread optimism."
        elif classification == "Positive":
            return "Sentiment is generally positive with bullish undertones."
        elif classification == "Neutral":
            return "Sentiment is mixed or neutral with no clear direction."
        elif classification == "Negative":
            return "Sentiment is generally negative with bearish undertones."
        else:
            return "Sentiment is strongly negative with widespread pessimism."
