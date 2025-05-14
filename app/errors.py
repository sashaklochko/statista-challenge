from typing import Dict, Any, Optional
from fastapi import HTTPException, status

# Base error class
class BaseError(Exception):
    """Base class for all application-specific exceptions"""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "An error occurred"
    error_code: str = "internal_error"
    
    def __init__(
        self,
        detail: Optional[str] = None,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.detail = detail or self.detail
        self.status_code = status_code or self.status_code
        self.error_code = error_code or self.error_code
        self.context = context or {}
        super().__init__(self.detail)
    
    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException"""
        error_detail = {
            "error_code": self.error_code,
            "message": self.detail
        }
        
        # Add context if available
        if self.context:
            error_detail["context"] = self.context
            
        return HTTPException(
            status_code=self.status_code,
            detail=error_detail
        )


# Configuration errors
class ConfigurationError(BaseError):
    """Error related to system configuration"""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Configuration error"
    error_code = "configuration_error"


# Elasticsearch errors
class ElasticsearchError(BaseError):
    """Base class for Elasticsearch-related errors"""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Elasticsearch error"
    error_code = "elasticsearch_error"


class ElasticsearchConnectionError(ElasticsearchError):
    """Error connecting to Elasticsearch"""
    detail = "Cannot connect to Elasticsearch"
    error_code = "elasticsearch_connection_error"


class ElasticsearchQueryError(ElasticsearchError):
    """Error executing query in Elasticsearch"""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Error executing Elasticsearch query"
    error_code = "elasticsearch_query_error"


# Search errors
class SearchError(BaseError):
    """Base class for search-related errors"""
    status_code = status.HTTP_400_BAD_REQUEST
    detail = "Search error"
    error_code = "search_error"


class InvalidSearchTypeError(SearchError):
    """Invalid search type specified"""
    detail = "Invalid search type"
    error_code = "invalid_search_type"
    
    def __init__(
        self,
        search_type: str,
        valid_types: list,
        **kwargs
    ):
        context = {
            "provided": search_type,
            "valid_types": valid_types
        }
        super().__init__(
            detail=f"Invalid search type: {search_type}. Valid types are: {', '.join(valid_types)}",
            context=context,
            **kwargs
        )


class EmptyQueryError(SearchError):
    """Empty query provided"""
    detail = "Query cannot be empty"
    error_code = "empty_query"


# Embedding errors
class EmbeddingError(BaseError):
    """Base class for embedding-related errors"""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Embedding error"
    error_code = "embedding_error"


class ModelLoadError(EmbeddingError):
    """Error loading embedding model"""
    detail = "Error loading embedding model"
    error_code = "model_load_error"


class EmbeddingGenerationError(EmbeddingError):
    """Error generating embeddings"""
    detail = "Error generating embeddings"
    error_code = "embedding_generation_error"


# Data conversion errors
class DataConversionError(BaseError):
    """Error converting data between formats"""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Data conversion error"
    error_code = "data_conversion_error" 