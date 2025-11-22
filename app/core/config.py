"""
Application Configuration

Centralized configuration management using pydantic-settings.
Supports environment variables and .env files.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with support for environment variables.
    
    Set these via:
    1. Environment variables (e.g., KREDITAX_APP_NAME="MyApp")
    2. .env file in project root
    3. Default values (below)
    """
    
    # Application
    app_name: str = Field(default="KreditaX", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # API
    api_v1_prefix: str = Field(default="/api/v1", description="API v1 prefix")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    
    # Model
    model_path: str = Field(
        default="ml/artifacts/xgboost_model_latest.joblib",
        description="Path to trained model"
    )
    preprocessor_path: str = Field(
        default="ml/artifacts/preprocessor.joblib",
        description="Path to preprocessor"
    )
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration in minutes"
    )
    
    # Audit Log
    audit_log_path: str = Field(
        default="data/audit_logs.csv",
        description="Path to audit log file"
    )
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    
    # POJK Compliance
    default_risk_threshold: float = Field(
        default=0.5,
        description="Default risk threshold for approve/reject"
    )
    enable_explainability: bool = Field(
        default=True,
        description="Enable SHAP explanations"
    )
    
    # CORS
    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "KREDITAX_"
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (dependency injection)"""
    return settings
