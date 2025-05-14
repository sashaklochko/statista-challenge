import os
from typing import Dict, Any

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# API configuration
API_TITLE = "Statista Context API"
API_DESCRIPTION = "API for retrieving relevant Statista data based on natural language queries"
API_VERSION = os.getenv("API_VERSION", "1.0.0")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "9000"))
API_DEBUG = os.getenv("API_DEBUG", "True").lower() in ("true", "1", "t")

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CORS_METHODS = os.getenv("CORS_METHODS", "*").split(",")
CORS_HEADERS = os.getenv("CORS_HEADERS", "*").split(",")

# Elasticsearch configuration
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "statistics")
ES_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "")
ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")
ES_TIMEOUT = int(os.getenv("ELASTICSEARCH_TIMEOUT", "30"))
ES_MAX_RETRIES = int(os.getenv("ELASTICSEARCH_MAX_RETRIES", "10"))
ES_RETRY_DELAY = int(os.getenv("ELASTICSEARCH_RETRY_DELAY", "5"))

# Search configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Vector search parameters
VECTOR_DIMENSION = 384  # Dimension for the all-MiniLM-L6-v2 model

# Get dictionary of all configuration
def get_all_config() -> Dict[str, Any]:
    """Return all configuration as a dictionary for logging purposes"""
    # Exclude sensitive data like passwords
    config_dict = {k: v for k, v in globals().items() 
                  if not k.startswith("_") and k.isupper() and k != "ES_PASSWORD"}
    return config_dict 