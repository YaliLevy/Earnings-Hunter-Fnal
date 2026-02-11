#!/usr/bin/env python3
"""
Model Training Script for The Earnings Hunter.

Trains and compares multiple ML models for earnings prediction:
- RandomForest
- XGBoost
- LightGBM
- Logistic Regression
- Neural Network (MLP)

Uses 5% threshold for Growth/Risk classification.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.ml.model_comparison import ModelComparison
from src.ml.trainer import EarningsModelTrainer
from src.utils.logger import get_logger
from config.settings import Constants

logger = get_logger(__name__)


def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate training data.

    Args:
        data_path: Path to training CSV

    Returns:
        Validated DataFrame
    """
    logger.info(f"Loading training data from {data_path}")

    df = pd.read_csv(data_path)

    # Validate required columns
    required_cols = ["label"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove unknown labels
    df = df[df["label"].isin(["Growth", "Stagnation", "Risk"])]

    logger.info(f"Loaded {len(df)} training samples")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for training.

    Args:
        df: Training DataFrame

    Returns:
        (X, y) tuple
    """
    # Define feature columns (exclude metadata and target)
    exclude_cols = [
        "symbol", "year", "quarter", "collection_date",
        "earnings_date", "label", "pre_price", "post_price",
        "price_change_pct", "has_transcript"
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Select numeric columns only
    X = df[feature_cols].select_dtypes(include=[np.number])

    # Fill NaN with 0
    X = X.fillna(0)

    # Target
    y = df["label"]

    logger.info(f"Features prepared: {len(X.columns)} features, {len(X)} samples")

    return X, y


def train_models(
    data_path: str,
    output_dir: str = "data/models",
    test_size: float = 0.2
) -> dict:
    """
    Train and compare all models.

    Args:
        data_path: Path to training data CSV
        output_dir: Directory to save models
        test_size: Fraction for test set

    Returns:
        Training results dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_training_data(data_path)

    if len(df) < Constants.MIN_TRAINING_SAMPLES:
        logger.warning(f"Only {len(df)} samples, minimum recommended is {Constants.MIN_TRAINING_SAMPLES}")

    # Prepare features
    X, y = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

    # Initialize model comparison
    comparison = ModelComparison()

    # Train and compare all models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60 + "\n")

    results = comparison.train_and_compare(X_train, y_train, cv_folds=5)

    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(results.to_string(index=False))

    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    test_results = []
    for model_name in comparison.trained_models.keys():
        y_pred = comparison.trained_models[model_name].predict(X_test)
        accuracy = (y_pred == y_test).mean()
        test_results.append({
            "Model": model_name,
            "Test Accuracy": f"{accuracy:.4f}"
        })
        print(f"{model_name}: {accuracy:.4f}")

    # Get best model
    best_model, best_name = comparison.get_best_model()
    print(f"\nBest Model: {best_name}")

    # Save all models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)

    comparison.save_all_models(str(output_path))

    # Save feature names
    feature_names_path = output_path / "feature_names.json"
    import json
    with open(feature_names_path, "w") as f:
        json.dump(list(X.columns), f, indent=2)
    print(f"Saved feature names to {feature_names_path}")

    # Save training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "data_path": str(data_path),
        "num_samples": len(df),
        "num_features": len(X.columns),
        "feature_names": list(X.columns),
        "label_distribution": df["label"].value_counts().to_dict(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_model": best_name,
        "cv_results": results.to_dict("records"),
        "test_results": test_results
    }

    metadata_path = output_path / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {metadata_path}")

    # Feature importance
    print("\n" + "="*60)
    print("TOP 10 FEATURE IMPORTANCE")
    print("="*60)

    importance = comparison.get_feature_importance(top_n=10)
    for model_name, features in importance.items():
        print(f"\n{model_name}:")
        for feat_name, imp_value in features[:10]:
            print(f"  {feat_name}: {imp_value:.4f}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Models saved to: {output_path}")
    print(f"Best model: {best_name}")

    return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train ML models for The Earnings Hunter"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Output directory for models (default: data/models)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("THE EARNINGS HUNTER - MODEL TRAINING")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print("="*60 + "\n")

    # Run training
    train_models(
        data_path=args.data,
        output_dir=args.output_dir,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()
