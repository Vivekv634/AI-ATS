"""
Configuration management for AI-ATS application.

Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
CONFIGS_DIR = ROOT_DIR / "configs"
RESOURCES_DIR = ROOT_DIR / "resources"


class DatabaseSettings(BaseSettings):
    """MongoDB database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 27017
    name: str = "ai_ats"
    username: str | None = None
    password: str | None = None

    @property
    def connection_string(self) -> str:
        """Generate MongoDB connection string."""
        if self.username and self.password:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"mongodb://{self.host}:{self.port}"


class VectorStoreSettings(BaseSettings):
    """Vector database configuration for embeddings."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_")

    provider: Literal["chromadb", "faiss"] = "chromadb"
    persist_directory: Path = DATA_DIR / "vectors"
    collection_name: str = "resume_embeddings"


class MLSettings(BaseSettings):
    """Machine Learning model configuration."""

    model_config = SettingsConfigDict(env_prefix="ML_")

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # spaCy model
    spacy_model: str = "en_core_web_trf"

    # Device settings
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"

    # Batch processing
    batch_size: int = 32

    # Model paths
    models_directory: Path = DATA_DIR / "models"

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Auto-detect device if set to auto."""
        if v == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return v


class UISettings(BaseSettings):
    """User interface configuration."""

    model_config = SettingsConfigDict(env_prefix="UI_")

    theme: Literal["light", "dark", "system"] = "system"
    language: str = "en"
    window_width: int = 1400
    window_height: int = 900
    font_size: int = 12


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    file_path: Path = ROOT_DIR / "logs" / "ai_ats.log"
    rotation: str = "10 MB"
    retention: str = "30 days"
    console_output: bool = True


class AppSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    name: str = "AI-ATS"
    version: str = "0.1.0"
    description: str = "AI-powered Applicant Tracking System"
    debug: bool = False

    # Environment
    environment: Literal["development", "production", "testing"] = "development"

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    ui: UISettings = Field(default_factory=UISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


# Global settings instance (singleton pattern)
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """Force reload settings from environment."""
    global _settings
    _settings = AppSettings()
    return _settings


# Convenience exports
settings = get_settings()
