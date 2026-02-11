"""
Multi-model training and comparison system.

Trains and compares multiple ML models to find the best performer:
- Random Forest (baseline)
- XGBoost (high accuracy)
- LightGBM (fast, good with large data)
- Logistic Regression (interpretable)
- Neural Network MLP (complex patterns)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from tqdm import tqdm

from src.utils.logger import get_logger
from config.settings import Constants

logger = get_logger(__name__)


class ModelComparison:
    """
    Train and compare multiple ML models to find the best performer.

    Supports:
    - Single best model prediction
    - All models prediction
    - Consensus voting prediction
    """

    # Label mapping
    LABEL_MAP = {
        0: Constants.PREDICTION_GROWTH,
        1: Constants.PREDICTION_STAGNATION,
        2: Constants.PREDICTION_RISK
    }

    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    def __init__(self):
        """Initialize model comparison with all available models."""
        self.models = self._create_models()
        self.scaler = StandardScaler()
        self.results = pd.DataFrame()
        self.trained_models = {}
        self.best_model_name = None
        self.feature_names = []

    def _create_models(self) -> Dict[str, Any]:
        """Create model instances."""
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "neural_network": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
        }

        # Add XGBoost if available
        if HAS_XGBOOST:
            models["xgboost"] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="mlogloss"
            )

        # Add LightGBM if available
        if HAS_LIGHTGBM:
            models["lightgbm"] = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

        return models

    def train_and_compare(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> pd.DataFrame:
        """
        Train all models and compare their performance.

        Uses TimeSeriesSplit for cross-validation to respect
        temporal ordering of financial data.

        Args:
            X: Feature DataFrame
            y: Target Series
            cv_folds: Number of cross-validation folds

        Returns:
            DataFrame with comparison metrics for all models
        """
        logger.info(f"Starting multi-model comparison...")
        logger.info(f"Dataset: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Cross-validation: {cv_folds} folds (TimeSeriesSplit)")

        # Store feature names
        self.feature_names = list(X.columns)

        # Convert labels to numeric if needed
        if y.dtype == object:
            y = y.map(self.REVERSE_LABEL_MAP)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # TimeSeriesSplit for financial data
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        results = []

        for name, model in tqdm(self.models.items(), desc="Training models"):
            logger.info(f"Training {name}...")

            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_scaled, y,
                    cv=tscv,
                    scoring="accuracy"
                )

                # Train on full data
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)

                # Calculate metrics
                metrics = {
                    "model": name,
                    "cv_accuracy_mean": cv_scores.mean(),
                    "cv_accuracy_std": cv_scores.std(),
                    "train_accuracy": accuracy_score(y, y_pred),
                    "precision_weighted": precision_score(
                        y, y_pred, average="weighted", zero_division=0
                    ),
                    "recall_weighted": recall_score(
                        y, y_pred, average="weighted", zero_division=0
                    ),
                    "f1_weighted": f1_score(
                        y, y_pred, average="weighted", zero_division=0
                    ),
                }

                results.append(metrics)
                self.trained_models[name] = model

                logger.info(
                    f"  CV Accuracy: {metrics['cv_accuracy_mean']:.2%} "
                    f"(+/- {metrics['cv_accuracy_std']:.2%})"
                )

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                continue

        # Create comparison DataFrame
        self.results = pd.DataFrame(results)
        self.results = self.results.sort_values("cv_accuracy_mean", ascending=False)

        # Select best model
        if len(self.results) > 0:
            self.best_model_name = self.results.iloc[0]["model"]
            logger.info(f"Best model: {self.best_model_name}")

        return self.results

    def get_best_model(self) -> Tuple[str, Any]:
        """
        Return the best performing model.

        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.best_model_name:
            raise ValueError("No models have been trained yet")

        return self.best_model_name, self.trained_models[self.best_model_name]

    def predict(
        self,
        X: pd.DataFrame,
        mode: str = "best"
    ) -> Dict[str, Any]:
        """
        Make predictions with specified mode.

        Args:
            X: Feature DataFrame (single row or multiple)
            mode: Prediction mode ("best", "all", "consensus")

        Returns:
            Dict with predictions based on mode
        """
        if not self.trained_models:
            raise ValueError("No models have been trained yet")

        # Scale features
        X_scaled = self.scaler.transform(X)

        if mode == "best":
            return self._predict_best(X_scaled)
        elif mode == "all":
            return self._predict_all(X_scaled)
        elif mode == "consensus":
            return self._predict_consensus(X_scaled)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'best', 'all', or 'consensus'")

    def _predict_best(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Predict using best model only."""
        model_name, model = self.get_best_model()
        pred = model.predict(X_scaled)[0]

        # Get probability/confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            confidence = float(max(proba))
            probabilities = {
                self.LABEL_MAP[i]: float(p)
                for i, p in enumerate(proba)
            }
        else:
            confidence = None
            probabilities = None

        return {
            "prediction": self.LABEL_MAP.get(pred, str(pred)),
            "confidence": confidence,
            "probabilities": probabilities,
            "model_used": model_name,
            "mode": "best",
            "disclaimer": "This is NOT financial advice"
        }

    def _predict_all(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Predict using all models."""
        predictions = []

        for name, model in self.trained_models.items():
            pred = model.predict(X_scaled)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)[0]
                confidence = float(max(proba))
            else:
                confidence = None

            predictions.append({
                "model": name,
                "prediction": self.LABEL_MAP.get(pred, str(pred)),
                "confidence": confidence
            })

        # Sort by confidence
        predictions.sort(
            key=lambda x: x["confidence"] if x["confidence"] else 0,
            reverse=True
        )

        return {
            "model_predictions": predictions,
            "mode": "all",
            "disclaimer": "This is NOT financial advice"
        }

    def _predict_consensus(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Get consensus prediction from all models."""
        all_predictions = self._predict_all(X_scaled)

        # Count votes
        votes = {}
        for pred in all_predictions["model_predictions"]:
            label = pred["prediction"]
            votes[label] = votes.get(label, 0) + 1

        # Find consensus
        consensus_pred = max(votes.items(), key=lambda x: x[1])[0]
        agreement_count = votes[consensus_pred]
        total_models = len(self.trained_models)
        agreement_ratio = agreement_count / total_models

        # Get best model's prediction
        best_pred = self._predict_best(X_scaled)

        return {
            "consensus_prediction": consensus_pred,
            "agreement_ratio": agreement_ratio,
            "models_agree": agreement_count,
            "models_total": total_models,
            "vote_distribution": votes,
            "model_predictions": all_predictions["model_predictions"],
            "best_model_name": self.best_model_name,
            "best_model_prediction": best_pred["prediction"],
            "best_model_confidence": best_pred["confidence"],
            "mode": "consensus",
            "disclaimer": "This is NOT financial advice"
        }

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for models that support it.

        Args:
            top_n: Number of top features to return

        Returns:
            Dict mapping model name to list of (feature, importance) tuples
        """
        importance = {}

        for name, model in self.trained_models.items():
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                feature_imp = sorted(
                    zip(self.feature_names, imp),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                importance[name] = feature_imp

            elif hasattr(model, "coef_"):
                # For logistic regression
                coef = np.abs(model.coef_).mean(axis=0)
                feature_imp = sorted(
                    zip(self.feature_names, coef),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                importance[name] = feature_imp

        return importance

    def save_all_models(self, directory: str = "data/models") -> None:
        """
        Save all trained models and metadata.

        Args:
            directory: Directory to save models
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for name, model in self.trained_models.items():
            joblib.dump(model, dir_path / f"{name}.pkl")

        # Save scaler
        joblib.dump(self.scaler, dir_path / "scaler.pkl")

        # Save feature names
        joblib.dump(self.feature_names, dir_path / "feature_names.pkl")

        # Save comparison results
        self.results.to_csv(dir_path / "model_comparison.csv", index=False)

        # Save best model info
        with open(dir_path / "best_model.txt", "w") as f:
            f.write(self.best_model_name or "")

        logger.info(f"Saved {len(self.trained_models)} models to {directory}/")

    def load_all_models(self, directory: str = "data/models") -> None:
        """
        Load all trained models and metadata.

        Args:
            directory: Directory containing saved models
        """
        dir_path = Path(directory)

        # Load scaler
        scaler_path = dir_path / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        # Load feature names
        features_path = dir_path / "feature_names.pkl"
        if features_path.exists():
            self.feature_names = joblib.load(features_path)

        # Load each model
        for name in self.models.keys():
            model_path = dir_path / f"{name}.pkl"
            if model_path.exists():
                self.trained_models[name] = joblib.load(model_path)

        # Load comparison results
        results_path = dir_path / "model_comparison.csv"
        if results_path.exists():
            self.results = pd.read_csv(results_path)

        # Load best model info
        best_model_path = dir_path / "best_model.txt"
        if best_model_path.exists():
            with open(best_model_path, "r") as f:
                self.best_model_name = f.read().strip()

        logger.info(f"Loaded {len(self.trained_models)} models from {directory}/")

    def print_comparison_report(self) -> None:
        """Print formatted comparison report."""
        if self.results.empty:
            print("No comparison results available. Train models first.")
            return

        print("\n" + "=" * 70)
        print("MODEL COMPARISON RESULTS")
        print("=" * 70)

        for _, row in self.results.iterrows():
            star = " *" if row["model"] == self.best_model_name else ""
            print(f"\n{row['model'].upper()}{star}")
            print(f"  CV Accuracy:  {row['cv_accuracy_mean']:.2%} (+/- {row['cv_accuracy_std']:.2%})")
            print(f"  F1 Score:     {row['f1_weighted']:.2%}")
            print(f"  Precision:    {row['precision_weighted']:.2%}")
            print(f"  Recall:       {row['recall_weighted']:.2%}")

        print("\n" + "=" * 70)
        print(f"BEST MODEL: {self.best_model_name}")
        print("=" * 70)
