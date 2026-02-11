"""
Model training pipeline for The Earnings Hunter.

Trains ML models for 3-class earnings prediction:
- Growth: >5% 3-day return
- Stagnation: -5% to +5% 3-day return
- Risk: <-5% 3-day return
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from src.utils.logger import get_logger
from src.ml.model_comparison import ModelComparison
from config.settings import Constants, get_settings

logger = get_logger(__name__)


class EarningsModelTrainer:
    """
    Train earnings prediction models.

    Target variable: 3-day post-earnings stock movement
    Labels:
        - Growth: >5% (significant positive reaction)
        - Stagnation: -5% to +5% (neutral reaction)
        - Risk: <-5% (significant negative reaction)
    """

    # Default feature columns (will be dynamically determined)
    EXCLUDE_COLUMNS = [
        "symbol", "earnings_date", "label", "outcome",
        "price_before", "price_after", "return_pct"
    ]

    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize trainer.

        Args:
            models_dir: Directory for model storage
        """
        settings = get_settings()

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.growth_threshold = settings.growth_threshold
        self.risk_threshold = settings.risk_threshold
        self.min_samples = settings.min_training_samples

        self.model_comparison = ModelComparison()
        self.training_metadata = {}

    def calculate_outcome(
        self,
        price_before: float,
        price_after: float
    ) -> str:
        """
        Calculate outcome label based on price change.

        Args:
            price_before: Stock price before earnings
            price_after: Stock price 3 days after earnings

        Returns:
            Label: "Growth", "Stagnation", or "Risk"
        """
        if price_before <= 0:
            return Constants.PREDICTION_STAGNATION

        return_pct = ((price_after - price_before) / price_before) * 100

        if return_pct > self.growth_threshold:
            return Constants.PREDICTION_GROWTH
        elif return_pct < self.risk_threshold:
            return Constants.PREDICTION_RISK
        else:
            return Constants.PREDICTION_STAGNATION

    def prepare_data(
        self,
        df: pd.DataFrame,
        label_column: str = "label"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.

        Args:
            df: Raw DataFrame with features and labels
            label_column: Name of the label column

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        # Identify feature columns
        feature_columns = [
            col for col in df.columns
            if col not in self.EXCLUDE_COLUMNS and col != label_column
        ]

        # Filter to numeric columns only
        numeric_cols = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()

        X = df[numeric_cols].copy()
        y = df[label_column].copy()

        # Fill NaN values
        X = X.fillna(0)

        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)

        logger.info(f"Prepared {len(X)} samples with {len(numeric_cols)} features")
        return X, y

    def train(
        self,
        df: pd.DataFrame,
        label_column: str = "label",
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train models on provided data.

        Args:
            df: Training DataFrame with features and labels
            label_column: Name of label column
            cv_folds: Number of cross-validation folds

        Returns:
            Dict with training results and metrics
        """
        # Check minimum samples
        if len(df) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} samples, got {len(df)}"
            )

        logger.info(f"Training on {len(df)} samples...")

        # Prepare data
        X, y = self.prepare_data(df, label_column)

        # Check class distribution
        class_dist = y.value_counts()
        logger.info(f"Class distribution:\n{class_dist}")

        # Train and compare models
        results = self.model_comparison.train_and_compare(X, y, cv_folds)

        # Store metadata
        self.training_metadata = {
            "train_date": datetime.now().isoformat(),
            "n_samples": len(df),
            "n_features": len(X.columns),
            "feature_names": list(X.columns),
            "class_distribution": class_dist.to_dict(),
            "thresholds": {
                "growth": self.growth_threshold,
                "risk": self.risk_threshold
            },
            "best_model": self.model_comparison.best_model_name,
            "cv_folds": cv_folds
        }

        # Print report
        self.model_comparison.print_comparison_report()

        return {
            "comparison_results": results,
            "metadata": self.training_metadata,
            "best_model": self.model_comparison.best_model_name,
            "best_accuracy": results.iloc[0]["cv_accuracy_mean"],
        }

    def save(self) -> None:
        """Save trained models and metadata."""
        # Save models
        self.model_comparison.save_all_models(str(self.models_dir))

        # Save metadata
        metadata_path = self.models_dir / "training_metadata.pkl"
        joblib.dump(self.training_metadata, metadata_path)

        logger.info(f"Saved models to {self.models_dir}/")

    def load(self) -> None:
        """Load trained models and metadata."""
        # Load models
        self.model_comparison.load_all_models(str(self.models_dir))

        # Load metadata
        metadata_path = self.models_dir / "training_metadata.pkl"
        if metadata_path.exists():
            self.training_metadata = joblib.load(metadata_path)

        logger.info(f"Loaded models from {self.models_dir}/")

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance from trained models.

        Args:
            top_n: Number of top features to return

        Returns:
            Dict mapping model name to feature importance list
        """
        return self.model_comparison.get_feature_importance(top_n)

    def evaluate(
        self,
        df: pd.DataFrame,
        label_column: str = "label"
    ) -> Dict[str, Any]:
        """
        Evaluate models on held-out data.

        Args:
            df: Evaluation DataFrame
            label_column: Name of label column

        Returns:
            Dict with evaluation metrics
        """
        X, y = self.prepare_data(df, label_column)

        # Get predictions from all models
        predictions = {}
        for name, model in self.model_comparison.trained_models.items():
            X_scaled = self.model_comparison.scaler.transform(X)
            y_pred = model.predict(X_scaled)
            predictions[name] = {
                "accuracy": float((y_pred == y).mean()),
                "predictions": y_pred.tolist()
            }

        return {
            "n_samples": len(df),
            "model_results": predictions
        }

    @staticmethod
    def create_training_dataset(
        features_list: List[Dict[str, Any]],
        labels: List[str]
    ) -> pd.DataFrame:
        """
        Create training dataset from feature dicts and labels.

        Args:
            features_list: List of feature dictionaries
            labels: List of corresponding labels

        Returns:
            DataFrame ready for training
        """
        df = pd.DataFrame(features_list)
        df["label"] = labels
        return df
