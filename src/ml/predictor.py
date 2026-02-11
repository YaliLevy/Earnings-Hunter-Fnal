"""
Prediction module for The Earnings Hunter.

Provides prediction capabilities with multiple modes:
- best: Use only the best performing model
- all: Return predictions from all models
- consensus: Return consensus from all models
"""

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.ml.model_comparison import ModelComparison
from config.settings import Constants

logger = get_logger(__name__)


class EarningsPredictor:
    """
    Make predictions using trained models.

    Supports three prediction modes:
    - best: Use only the best performing model
    - all: Return predictions from all models
    - consensus: Return consensus voting from all models
    """

    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize predictor.

        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.model_comparison = ModelComparison()
        self.is_loaded = False

    def load_models(self) -> None:
        """Load trained models from disk."""
        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"Models directory not found: {self.models_dir}"
            )

        self.model_comparison.load_all_models(str(self.models_dir))
        self.is_loaded = True
        logger.info("Models loaded successfully")

    def ensure_loaded(self) -> None:
        """Ensure models are loaded."""
        if not self.is_loaded:
            self.load_models()

    def predict(
        self,
        features: Dict[str, float],
        mode: str = "best"
    ) -> Dict[str, Any]:
        """
        Make prediction from feature dictionary.

        Args:
            features: Dict of feature values
            mode: Prediction mode ("best", "all", "consensus")

        Returns:
            Prediction result dict
        """
        self.ensure_loaded()

        # Convert to DataFrame
        X = pd.DataFrame([features])

        # Get expected feature names
        expected_features = self.model_comparison.feature_names

        # Align features with expected columns
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0.0

        # Select only expected features in correct order
        X = X[expected_features]

        # Fill any remaining NaN
        X = X.fillna(0)

        # Make prediction
        result = self.model_comparison.predict(X, mode=mode)

        return result

    def predict_batch(
        self,
        features_list: list[Dict[str, float]],
        mode: str = "best"
    ) -> list[Dict[str, Any]]:
        """
        Make predictions for multiple samples.

        Args:
            features_list: List of feature dictionaries
            mode: Prediction mode

        Returns:
            List of prediction results
        """
        self.ensure_loaded()

        results = []
        for features in features_list:
            result = self.predict(features, mode)
            results.append(result)

        return results

    def get_prediction_with_explanation(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Get prediction with feature-based explanation.

        Returns the prediction along with the most influential
        features for the decision.

        Args:
            features: Dict of feature values

        Returns:
            Prediction with explanation
        """
        self.ensure_loaded()

        # Get consensus prediction for robustness
        prediction = self.predict(features, mode="consensus")

        # Get feature importance
        importance = self.model_comparison.get_feature_importance(top_n=10)

        # Get best model's importance
        best_model = self.model_comparison.best_model_name
        best_importance = importance.get(best_model, [])

        # Map feature values to importance
        influential_features = []
        for feat_name, imp_value in best_importance[:5]:
            feat_value = features.get(feat_name, 0)
            influential_features.append({
                "feature": feat_name,
                "importance": float(imp_value),
                "value": float(feat_value),
                "interpretation": self._interpret_feature(feat_name, feat_value)
            })

        prediction["explanation"] = {
            "influential_features": influential_features,
            "model_agreement": prediction.get("agreement_ratio", 1.0)
        }

        return prediction

    def _interpret_feature(self, feature_name: str, value: float) -> str:
        """
        Generate human-readable interpretation of a feature value.

        Args:
            feature_name: Name of the feature
            value: Feature value

        Returns:
            Interpretation string
        """
        # Financial features
        if "eps_surprise" in feature_name:
            if value > 0.1:
                return "Strong EPS beat"
            elif value > 0:
                return "Slight EPS beat"
            elif value < -0.1:
                return "Significant EPS miss"
            else:
                return "Slight EPS miss"

        if "revenue_surprise" in feature_name:
            if value > 0.05:
                return "Revenue beat"
            elif value < -0.05:
                return "Revenue miss"
            else:
                return "Revenue in-line"

        # CEO tone features
        if "ceo_confidence" in feature_name:
            if value > 0.3:
                return "CEO very confident"
            elif value > 0:
                return "CEO moderately confident"
            elif value < -0.3:
                return "CEO defensive"
            else:
                return "CEO cautious"

        if "uncertainty_ratio" in feature_name:
            if value > 0.1:
                return "High uncertainty language"
            else:
                return "Low uncertainty language"

        # Social features
        if "hype_index" in feature_name:
            if value > 70:
                return "High social buzz"
            elif value > 40:
                return "Moderate social interest"
            else:
                return "Low social activity"

        if "sentiment" in feature_name:
            if value > 0.3:
                return "Bullish sentiment"
            elif value < -0.3:
                return "Bearish sentiment"
            else:
                return "Neutral sentiment"

        if "wsb_ratio" in feature_name:
            if value > 0.5:
                return "High WSB interest (speculative)"
            else:
                return "Normal retail interest"

        # Default
        if value > 0:
            return "Positive indicator"
        elif value < 0:
            return "Negative indicator"
        else:
            return "Neutral"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.

        Returns:
            Dict with model information
        """
        self.ensure_loaded()

        return {
            "best_model": self.model_comparison.best_model_name,
            "available_models": list(self.model_comparison.trained_models.keys()),
            "n_features": len(self.model_comparison.feature_names),
            "comparison_results": self.model_comparison.results.to_dict("records")
            if not self.model_comparison.results.empty else []
        }

    def get_confidence_level(self, confidence: Optional[float]) -> str:
        """
        Convert numeric confidence to level string.

        Args:
            confidence: Confidence value (0-1)

        Returns:
            Confidence level string
        """
        if confidence is None:
            return "Unknown"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        else:
            return "Low"

    def format_prediction_for_display(
        self,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format prediction result for UI display.

        Args:
            prediction: Raw prediction result

        Returns:
            Formatted result for display
        """
        mode = prediction.get("mode", "best")

        if mode == "best":
            return {
                "prediction": prediction["prediction"],
                "confidence": prediction.get("confidence"),
                "confidence_level": self.get_confidence_level(prediction.get("confidence")),
                "model": prediction.get("model_used"),
                "probabilities": prediction.get("probabilities"),
                "disclaimer": prediction.get("disclaimer")
            }

        elif mode == "consensus":
            return {
                "prediction": prediction["consensus_prediction"],
                "confidence": prediction.get("best_model_confidence"),
                "confidence_level": self.get_confidence_level(
                    prediction.get("best_model_confidence")
                ),
                "agreement": f"{prediction['models_agree']}/{prediction['models_total']}",
                "agreement_ratio": prediction["agreement_ratio"],
                "vote_distribution": prediction.get("vote_distribution"),
                "model_predictions": prediction.get("model_predictions"),
                "disclaimer": prediction.get("disclaimer")
            }

        else:  # all
            return {
                "model_predictions": prediction["model_predictions"],
                "disclaimer": prediction.get("disclaimer")
            }
