import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from elasticsearch import Elasticsearch

from app.retriever.embedding import EmbeddingService
from app.retriever.search import ElasticsearchService


class TestEmbeddingService:
    """Tests for the EmbeddingService class"""
    
    @patch('app.retriever.embedding.SentenceTransformer')
    def test_initialize_model(self, mock_transformer):
        """Test model initialization"""
        # Setup mock
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Create service
        service = EmbeddingService(model_name="test-model")
        
        # Assertions
        mock_transformer.assert_called_once_with("test-model")
        assert service.model == mock_model
        
    @patch('app.retriever.embedding.SentenceTransformer')
    def test_encode_query(self, mock_transformer):
        """Test query encoding"""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_transformer.return_value = mock_model
        
        # Create service and encode
        service = EmbeddingService()
        result = service.encode_query("test query")
        
        # Assertions
        mock_model.encode.assert_called_once_with("test query")
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([0.1, 0.2, 0.3]))


class TestElasticsearchService:
    """Tests for the ElasticsearchService class"""
    
    @patch('app.retriever.search.Elasticsearch')
    def test_initialization(self, mock_es_class):
        """Test initialization of Elasticsearch service"""
        # Setup mock
        mock_es = MagicMock()
        mock_es.ping.return_value = True
        mock_es_class.return_value = mock_es
        
        # Mock embedding model
        mock_model = MagicMock()
        
        # Create service
        service = ElasticsearchService(mock_model)
        
        # Assertions
        assert service.embedding_model == mock_model
        assert service.es_client == mock_es
        mock_es.ping.assert_called_once()
    
    @patch('app.retriever.search.Elasticsearch')
    def test_initialization_failure(self, mock_es_class):
        """Test initialization failure handling"""
        # Setup mock to fail ping
        mock_es = MagicMock()
        mock_es.ping.return_value = False
        mock_es_class.return_value = mock_es
        
        # Mock embedding model
        mock_model = MagicMock()
        
        # Create service
        service = ElasticsearchService(mock_model)
        
        # Assertions
        assert service.es_client is None
        assert not service.is_ready()
    
    @patch('app.retriever.search.Elasticsearch')
    def test_text_search(self, mock_es_class):
        """Test text search functionality"""
        # Sample data
        sample_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {"id": "1", "title": "Test Doc"},
                        "_score": 0.95
                    },
                    {
                        "_source": {"id": "2", "title": "Another Doc"},
                        "_score": 0.85
                    }
                ]
            }
        }
        
        # Setup mock
        mock_es = MagicMock()
        mock_es.ping.return_value = True
        mock_es.search.return_value = sample_response
        mock_es_class.return_value = mock_es
        
        # Mock embedding model
        mock_model = MagicMock()
        
        # Create service and perform search
        service = ElasticsearchService(mock_model)
        results = service.es_text_search("test query", k=2)
        
        # Assertions
        assert len(results) == 2
        assert results[0][0]["id"] == "1"
        assert results[0][1] == 0.95
        assert results[1][0]["id"] == "2"
        assert results[1][1] == 0.85
        
        # Verify search was called with correct parameters
        mock_es.search.assert_called_once()
        call_args = mock_es.search.call_args[1]
        assert call_args["index"] == service.es_index
        assert "multi_match" in call_args["body"]["query"]
    
    @patch('app.retriever.search.Elasticsearch')
    def test_vector_search(self, mock_es_class):
        """Test vector search functionality"""
        # Sample data
        sample_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {"id": "1", "title": "Test Doc"},
                        "_score": 0.95
                    }
                ]
            }
        }
        
        # Setup mock
        mock_es = MagicMock()
        mock_es.ping.return_value = True
        mock_es.search.return_value = sample_response
        mock_es_class.return_value = mock_es
        
        # Mock embedding model
        mock_model = MagicMock()
        
        # Create service and perform search
        service = ElasticsearchService(mock_model)
        query_vector = np.array([0.1, 0.2, 0.3])
        results = service.es_vector_search(query_vector, k=1)
        
        # Assertions
        assert len(results) == 1
        assert results[0][0]["id"] == "1"
        assert results[0][1] == 0.95
        
        # Verify search was called with correct parameters
        mock_es.search.assert_called_once()
        call_args = mock_es.search.call_args[1]
        assert call_args["index"] == service.es_index
        assert "knn" in call_args["body"]
        assert call_args["body"]["knn"]["query_vector"] == query_vector.tolist()
    
    @patch('app.retriever.search.Elasticsearch')
    def test_hybrid_search(self, mock_es_class):
        """Test hybrid search functionality"""
        # Sample data
        sample_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {"id": "1", "title": "Test Doc"},
                        "_score": 0.95
                    }
                ]
            }
        }
        
        # Setup mock
        mock_es = MagicMock()
        mock_es.ping.return_value = True
        mock_es.search.return_value = sample_response
        mock_es_class.return_value = mock_es
        
        # Mock embedding model
        mock_model = MagicMock()
        
        # Create service and perform search
        service = ElasticsearchService(mock_model)
        query_vector = np.array([0.1, 0.2, 0.3])
        results = service.es_hybrid_search_bool("test query", query_vector, k=1)
        
        # Assertions
        assert len(results) == 1
        assert results[0][0]["id"] == "1"
        assert results[0][1] == 0.95
        
        # Verify search was called with correct parameters
        mock_es.search.assert_called_once()
        call_args = mock_es.search.call_args[1]
        assert call_args["index"] == service.es_index
        assert "bool" in call_args["body"]["query"]
        assert len(call_args["body"]["query"]["bool"]["should"]) == 2  # Both text and vector components 