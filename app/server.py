import os
from datetime import datetime
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
import time

from app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    API_HOST,
    API_PORT,
    API_DEBUG,
    CORS_ORIGINS,
    CORS_METHODS,
    CORS_HEADERS,
    ES_URL,
    EMBEDDING_MODEL,
    get_all_config
)
from app.logging_setup import setup_logging
from app.errors import (
    BaseError,
    InvalidSearchTypeError,
    EmptyQueryError,
    ElasticsearchQueryError,
    EmbeddingGenerationError,
    ElasticsearchConnectionError,
)
from app.retriever.embedding import EmbeddingService
from app.retriever.search import ElasticsearchService
from app.retriever.views import Document, QueryRequest, QueryResponse
from app.retriever.constants import (
    SEARCH_TYPE_SEMANTIC,
    SEARCH_TYPE_KEYWORD,
    SEARCH_TYPE_HYBRID,
)

# Define search types mapping
SEARCH_TYPES = {
    SEARCH_TYPE_SEMANTIC: "Vector-based similarity using sentence embeddings",
    SEARCH_TYPE_KEYWORD: "Traditional text search with BM25 ranking",
    SEARCH_TYPE_HYBRID: "Combines semantic and keyword search using Reciprocal Rank Fusion",
}

logger = setup_logging()

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=None,  # Disable default docs to use custom Swagger UI
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# Exception handler for custom errors
@app.exception_handler(BaseError)
async def baseerror_exception_handler(request: Request, exc: BaseError):
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder({
            "error_code": exc.error_code,
            "message": exc.detail,
            "context": exc.context if exc.context else None
        }),
    )

# Initialize services
logger.info("Initializing services with configuration: %s", get_all_config())
start_time = time.perf_counter_ns()

try:
    embedding_service = EmbeddingService()
    elasticsearch_service = ElasticsearchService(embedding_service.model)

    time_delta = time.perf_counter_ns() - start_time
    initialization_time = time_delta / 1_000_000
    logger.info("Service initialization complete in %.2f ms", initialization_time)
except Exception as e:
    logger.error("Initialization error: %s", str(e), exc_info=True)
    raise

async def log_request(request: Request):
    logger.debug("Request received: %s %s", request.method, request.url.path)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1}
    )

@app.post(
    "/forward-context",
    response_model=QueryResponse,
    summary="Get contextual information based on a natural language query",
    description="""
    This endpoint accepts a natural language query and returns the most relevant
    statistical documents from the Statista database.

    You can specify the search type as:
    - `semantic`: Vector-based similarity using sentence embeddings
    - `keyword`: Traditional text search with BM25 ranking
    - `hybrid`: Combines semantic and keyword search for best results

    The response includes similarity scores and metadata for each result.
    """,
    dependencies=[Depends(log_request)]
)
def forward_context(request: QueryRequest):
    try:
        if not request.query.strip():
            raise EmptyQueryError()

        if request.search_type not in SEARCH_TYPES:
            raise InvalidSearchTypeError(
                request.search_type,
                list(SEARCH_TYPES.keys())
            )

        # Check if Elasticsearch is ready
        if not elasticsearch_service.is_ready():
            raise ElasticsearchConnectionError(detail=f"Elasticsearch is not available at {ES_URL}")

        query_start_time = time.perf_counter_ns()
        logger.info("Processing query: '%s' with search type: %s, limit: %d",
                   request.query, request.search_type, request.limit)

        # Generate query embedding
        try:
            query_embedding = embedding_service.encode_query(request.query)
        except Exception as e:
            logger.error("Error encoding query '%s': %s", request.query, str(e), exc_info=True)
            raise EmbeddingGenerationError(detail=f"Error encoding query: {str(e)}")

        # Execute search based on type
        try:
            if request.search_type == SEARCH_TYPE_SEMANTIC:
                results = elasticsearch_service.es_vector_search(
                    query_embedding, request.limit
                )
            elif request.search_type == SEARCH_TYPE_KEYWORD:
                results = elasticsearch_service.es_text_search(
                    request.query, request.limit
                )
            elif request.search_type == SEARCH_TYPE_HYBRID:
                results = elasticsearch_service.es_hybrid_search_bool(
                    request.query, query_embedding, request.limit
                )
        except Exception as e:
            logger.error("Search error for query '%s' (%s): %s",
                        request.query, request.search_type, str(e), exc_info=True)
            raise ElasticsearchQueryError(detail=f"Error executing search: {str(e)}")

        execution_time = (time.perf_counter_ns() - query_start_time) / 1_000_000
        logger.info(
            "Query executed: '%s' (%s) | Results: %d | Execution time: %.2f ms",
            request.query, request.search_type, len(results), execution_time
        )

        return QueryResponse(
            results=results,
            query=request.query,
            search_type=request.search_type,
            timestamp=datetime.now(),
            execution_time_ms=execution_time,
            search_engine="elasticsearch"
        )

    except BaseError as e:
        raise e.to_http_exception()
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get(
    "/ready",
    summary="Health check endpoint",
    description="Returns status 200 if the service is ready to accept requests, 423 otherwise",
    dependencies=[Depends(log_request)]
)
def ready():
    try:
        if elasticsearch_service.is_ready():
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=jsonable_encoder({
                    "status": "ok",
                    "message": "Service is ready",
                    "timestamp": datetime.now().isoformat()
                }),
            )
        else:
            logger.warning("Health check failed - Elasticsearch not ready at %s",
                          elasticsearch_service.es_url)
            return JSONResponse(
                status_code=status.HTTP_423_LOCKED,
                content=jsonable_encoder({
                    "status": "not_ready",
                    "message": "Service is starting up",
                    "timestamp": datetime.now().isoformat()
                }),
            )
    except Exception as e:
        logger.error("Health check error: %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=jsonable_encoder({
                "status": "error",
                "message": f"Health check error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }),
        )


@app.get(
    "/",
    summary="Root endpoint",
    description="Redirects to API documentation",
)
def root():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder({
            "service": API_TITLE,
            "version": API_VERSION,
            "docs_url": "/docs",
            "redoc_url": "/redoc"
        }),
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG,
    )
