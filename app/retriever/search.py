import os
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime
from functools import wraps
import time

import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from app.retriever.constants import (
    DEFAULT_ES_URL,
    DEFAULT_ES_INDEX,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_NUM_CANDIDATES,
    TITLE_WEIGHT,
    SUBJECT_WEIGHT,
    MINIMUM_SHOULD_MATCH,
    HYBRID_KNN_CANDIDATES,
    HYBRID_BOOST_FACTOR,
)
from app.retriever.views import Document

logger = logging.getLogger("statista-api")


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()

        elapsed_ms = (end_time - start_time) / 1_000_000
        logger.info(f"{func.__name__} execution time: {elapsed_ms:.2f}ms")
        return result

    return wrapper


class ElasticsearchService:
    def __init__(self, embedding_model: SentenceTransformer):
        """Initialize the Elasticsearch service with documents and embedding model"""
        self.embedding_model = embedding_model
        self.es_url = os.environ.get("ELASTICSEARCH_URL", DEFAULT_ES_URL)
        self.es_index = os.environ.get("ELASTICSEARCH_INDEX", DEFAULT_ES_INDEX)
        self.es_client = None
        try:
            es_client = Elasticsearch(self.es_url)
            if es_client.ping():
                logger.info(f"Connected to Elasticsearch at {self.es_url}")
                self.es_client = es_client
            else:
                logger.error("Failed to connect to Elasticsearch")
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")

    def is_ready(self) -> bool:
        return self.es_client is not None

    def _execute_search(
        self, query_body: Dict[str, Any], operation_name: str
    ) -> List[Document]:
        """
        Execute a search query and process the results

        Args:
            query_body: The Elasticsearch query body
            operation_name: Name of the operation for error logging

        Returns:
            List of Document objects
        """
        if not self.es_client:
            logger.warning("Elasticsearch client not available")
            return []

        try:
            response = self.es_client.search(index=self.es_index, body=query_body)

            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                score = hit["_score"]

                doc_copy = source.copy()
                doc_copy["id"] = str(doc_copy["id"])
                try:
                    if isinstance(doc_copy["date"], str):
                        doc_copy["date"] = datetime.fromisoformat(
                            doc_copy["date"].replace("Z", "+00:00")
                        )
                except Exception as e:
                    logger.warning("Error parsing date '%s': %s", doc_copy["date"], str(e))
                    doc_copy["date"] = datetime.now()

                doc_copy["similarity_score"] = float(score)
                results.append(Document(**doc_copy))

            return results

        except Exception as e:
            logger.error(f"Error in Elasticsearch {operation_name}: {e}")
            return []

    def _get_base_query_body(self, k: int) -> Dict[str, Any]:
        """
        Get the base query body with common fields

        Args:
            k: Number of results to return

        Returns:
            Base query body dictionary
        """
        return {
            "_source": [
                "id",
                "title",
                "subject",
                "description",
                "link",
                "date",
                "teaser_image_url",
            ],
            "size": k,
        }

    @timing_decorator
    def es_text_search(
        self, query: str, k: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Document]:
        """
        Perform a text search in Elasticsearch

        Args:
            query: The text query
            k: Number of results to return

        Returns:
            List of Document objects
        """
        query_body = self._get_base_query_body(k)
        query_body["query"] = {
            "multi_match": {
                "query": query,
                "fields": [
                    f"title^{TITLE_WEIGHT}",
                    f"subject^{SUBJECT_WEIGHT}",
                    "description",
                ],
                "type": "best_fields",
                "operator": "OR",
                "minimum_should_match": MINIMUM_SHOULD_MATCH,
            }
        }

        return self._execute_search(query_body, "text search")

    @timing_decorator
    def es_vector_search(
        self, query_embedding: np.ndarray, k: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Document]:
        """
        Perform a vector search in Elasticsearch

        Args:
            query_embedding: The embedding vector for the query
            k: Number of results to return

        Returns:
            List of Document objects
        """
        query_body = self._get_base_query_body(k)
        query_body["knn"] = {
            "field": "content_vector",
            "query_vector": query_embedding.tolist(),
            "k": k,
            "num_candidates": DEFAULT_NUM_CANDIDATES,
        }

        return self._execute_search(query_body, "vector search")

    @timing_decorator
    def es_hybrid_search_bool(
        self, query: str, query_embedding: np.ndarray, k: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Document]:
        """
        Perform a hybrid search in Elasticsearch using a bool query
        combining multi_match text search and kNN vector search

        Args:
            query: The text query
            query_embedding: The embedding vector for the query
            k: Number of results to return

        Returns:
            List of Document objects
        """
        query_body = self._get_base_query_body(k)
        query_body["query"] = {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                f"title^{TITLE_WEIGHT}",
                                f"subject^{SUBJECT_WEIGHT}",
                                "description",
                            ],
                            "type": "best_fields",
                            "operator": "OR",
                            "minimum_should_match": MINIMUM_SHOULD_MATCH,
                        }
                    },
                    {
                        "knn": {
                            "field": "content_vector",
                            "query_vector": query_embedding.tolist(),
                            "k": HYBRID_KNN_CANDIDATES,
                            "num_candidates": DEFAULT_NUM_CANDIDATES,
                            "boost": HYBRID_BOOST_FACTOR,
                        }
                    },
                ]
            }
        }

        return self._execute_search(query_body, "hybrid search with bool query")
