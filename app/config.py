import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    gemini_daily_limit: int = int(os.getenv("GEMINI_DAILY_LIMIT", "45"))
    gemini_min_interval: float = float(os.getenv("GEMINI_MIN_INTERVAL", "1.5"))
    gemini_enable_rate_limiting: bool = os.getenv("GEMINI_ENABLE_RATE_LIMITING", "true").lower() == "true"
    
    gemini_fallback_model: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash")
    gemini_max_retries: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    gemini_retry_delay: float = float(os.getenv("GEMINI_RETRY_DELAY", "2.0"))
    
    emergency_mode: bool = os.getenv("EMERGENCY_MODE", "false").lower() == "true"
    emergency_daily_limit: int = int(os.getenv("EMERGENCY_DAILY_LIMIT", "20"))
    
    usage_data_file: str = os.getenv("USAGE_DATA_FILE", "data/gemini_usage.json")
    usage_log_requests: bool = os.getenv("USAGE_LOG_REQUESTS", "true").lower() == "true"
    
    daytona_api_url: str = os.getenv("DAYTONA_API_URL", "https://api.daytona.io/api")
    daytona_api_key: str = os.getenv("DAYTONA_API_KEY", "")
    
    max_retry_attempts: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "5"))
    execution_timeout: int = int(os.getenv("EXECUTION_TIMEOUT", "30"))
    max_output_length: int = int(os.getenv("MAX_OUTPUT_LENGTH", "10000"))
    
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    allowed_origins: list = ["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()