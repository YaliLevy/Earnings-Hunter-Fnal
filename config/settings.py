"""
Configuration management using Pydantic BaseSettings.

All environment variables and application settings are centralized here.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ===== API Keys =====
    fmp_api_key: str = Field(..., description="FMP API key (Ultimate subscription)")
    openai_api_key: str = Field(..., description="OpenAI API key for CrewAI agents")
    # Reddit removed - using FMP Social Sentiment instead
    reddit_client_id: Optional[str] = Field(default=None, description="Reddit app client ID (not used)")
    reddit_client_secret: Optional[str] = Field(default=None, description="Reddit app client secret (not used)")
    reddit_user_agent: str = Field(
        default="EarningsHunter/1.0",
        description="Reddit API user agent (not used)"
    )

    # ===== OpenAI Settings =====
    openai_model_name: str = Field(
        default="gpt-4",
        description="OpenAI model for CrewAI agents"
    )

    # ===== Rate Limiting =====
    # Note: FMP has unlimited access (Ultimate subscription)
    reddit_requests_per_minute: int = Field(
        default=60,
        description="Reddit API rate limit (requests per minute)"
    )

    # ===== Golden Triangle Weights =====
    hard_data_weight: float = Field(
        default=0.40,
        description="Weight for financial data (FMP)"
    )
    soft_data_weight: float = Field(
        default=0.35,
        description="Weight for CEO tone (transcripts)"
    )
    street_psychology_weight: float = Field(
        default=0.25,
        description="Weight for social sentiment (Reddit)"
    )

    # ===== Prediction Thresholds =====
    growth_threshold: float = Field(
        default=5.0,
        description="Minimum % gain for 'Growth' classification"
    )
    risk_threshold: float = Field(
        default=-5.0,
        description="Maximum % loss for 'Risk' classification"
    )

    # ===== Model Settings =====
    model_path: str = Field(
        default="data/models",
        description="Directory for trained models"
    )
    min_training_samples: int = Field(
        default=500,
        description="Minimum earnings events for training"
    )

    # ===== Cache Settings =====
    cache_expiry_hours: int = Field(
        default=24,
        description="Cache expiry time in hours"
    )
    cache_directory: str = Field(
        default="data/cache",
        description="Directory for cached analysis results"
    )

    # ===== Logging =====
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )

    # ===== Paths =====
    data_raw_path: str = Field(default="data/raw")
    data_processed_path: str = Field(default="data/processed")
    data_models_path: str = Field(default="data/models")

    @field_validator("hard_data_weight", "soft_data_weight", "street_psychology_weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate that weights are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {v}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper

    @property
    def weights_sum(self) -> float:
        """Return sum of all weights (should be 1.0)."""
        return self.hard_data_weight + self.soft_data_weight + self.street_psychology_weight

    def validate_weights_sum(self) -> bool:
        """Check if weights sum to 1.0."""
        return abs(self.weights_sum - 1.0) < 0.001

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_raw_path,
            self.data_processed_path,
            self.data_models_path,
            self.cache_directory,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Constants that don't change
class Constants:
    """Application constants."""

    # Training requirements
    MIN_TRAINING_SAMPLES = 500

    # Prediction labels
    PREDICTION_GROWTH = "Growth"
    PREDICTION_STAGNATION = "Stagnation"
    PREDICTION_RISK = "Risk"

    # Feature prefixes
    FIN_PREFIX = "fin_"
    CEO_PREFIX = "ceo_"
    SOCIAL_PREFIX = "social_"

    # Reddit subreddits to monitor
    FINANCE_SUBREDDITS = [
        "wallstreetbets",
        "stocks",
        "investing",
        "options",
        "stockmarket",
    ]

    # Confidence words for transcript analysis
    CONFIDENCE_WORDS = [
        "strong", "confident", "exceed", "robust", "outstanding",
        "record", "momentum", "accelerating", "beat", "exceptional",
        "excellent", "phenomenal", "incredible", "remarkable", "solid"
    ]

    # Uncertainty words for transcript analysis
    UNCERTAINTY_WORDS = [
        "challenging", "headwinds", "uncertain", "pressure",
        "might", "possibly", "trying", "hope", "difficult",
        "concerned", "cautious", "volatile", "risk", "careful"
    ]

    # Forward guidance keywords
    GUIDANCE_KEYWORDS = [
        "expect", "anticipate", "guidance", "outlook",
        "next quarter", "full year", "forecast", "project",
        "target", "goal", "plan"
    ]

    # Disclaimer texts
    DISCLAIMER_BANNER = (
        "This tool is for educational and informational purposes only. "
        "Nothing presented here constitutes financial, investment, or trading advice. "
        "Always consult a licensed financial advisor before making investment decisions."
    )

    DISCLAIMER_HEBREW = (
        "כלי זה מיועד למטרות לימודיות ומידעיות בלבד. "
        "אין לראות בתוכן המוצג ייעוץ פיננסי, המלצת השקעה או המלצת מסחר. "
        "יש להתייעץ עם יועץ השקעות מורשה לפני קבלת החלטות השקעה."
    )
