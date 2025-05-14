import json
import logging
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL
from app.errors import ModelLoadError, EmbeddingGenerationError

logger = logging.getLogger("statista-api")


class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedding service with a specific model"""
        self.model_name = model_name
        self.model = self._initialize_model()
        self.document_embeddings = None

    def _initialize_model(self) -> SentenceTransformer:
        """Initialize the sentence transformer model"""
        try:
            model = SentenceTransformer(self.model_name)
            logger.info("Successfully loaded embedding model: %s", self.model_name)
            return model
        except Exception as e:
            logger.error("Error initializing embedding model: %s", str(e), exc_info=True)
            raise ModelLoadError(detail=f"Failed to load embedding model: {str(e)}")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into an embedding vector"""
        if not query or not query.strip():
            raise EmbeddingGenerationError(detail="Cannot encode empty query")

        try:
            return self.model.encode(query)
        except Exception as e:
            logger.error("Error encoding query: %s", str(e), exc_info=True)
            raise EmbeddingGenerationError(detail=f"Error generating embedding: {str(e)}")
