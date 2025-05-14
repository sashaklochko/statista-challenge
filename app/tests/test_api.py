import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from app.server import app
from app.retriever.views import Document, QueryResponse

client = TestClient(app)


@pytest.fixture
def mock_services(sample_docs, sample_embedding):
    """Fixture to mock the embedding and elasticsearch services"""
    with patch('app.server.embedding_service') as mock_embedding_service, \
         patch('app.server.elasticsearch_service') as mock_elasticsearch_service:
        
        # Configure mock embedding service
        mock_embedding_service.encode_query.return_value = sample_embedding
        
        # Configure mock elasticsearch service
        mock_elasticsearch_service.is_ready.return_value = True
        mock_elasticsearch_service.es_vector_search.return_value = [(doc, 0.85) for doc in sample_docs[:2]]
        mock_elasticsearch_service.es_text_search.return_value = [(doc, 0.75) for doc in sample_docs[:2]]
        mock_elasticsearch_service.es_hybrid_search_bool.return_value = [(doc, 0.95) for doc in sample_docs[:2]]
        
        yield


def test_forward_context_semantic_search(mock_services):
    """Test the forward-context endpoint with semantic search"""
    response = client.post(
        "/forward-context",
        json={"query": "AI companies", "search_type": "semantic", "limit": 2}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["query"] == "AI companies"
    assert data["search_type"] == "semantic"
    assert "execution_time_ms" in data
    assert len(data["results"]) == 2
    
    # Check first result
    result = data["results"][0]
    assert result["id"] == "1"
    assert result["title"] == "AI Unicorn Companies"
    assert result["similarity_score"] == 0.85


def test_forward_context_keyword_search(mock_services):
    """Test the forward-context endpoint with keyword search"""
    response = client.post(
        "/forward-context",
        json={"query": "smartphone market", "search_type": "keyword", "limit": 2}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["query"] == "smartphone market"
    assert data["search_type"] == "keyword"
    assert len(data["results"]) == 2
    
    # Check results have similarity scores
    for result in data["results"]:
        assert "similarity_score" in result
        assert result["similarity_score"] == 0.75


def test_forward_context_hybrid_search(mock_services):
    """Test the forward-context endpoint with hybrid search (default)"""
    response = client.post(
        "/forward-context",
        json={"query": "technology trends"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["query"] == "technology trends"
    assert data["search_type"] == "hybrid"  # Default
    assert len(data["results"]) <= 5  # Default limit
    
    # Check similarity scores
    for result in data["results"]:
        assert result["similarity_score"] == 0.95


def test_forward_context_invalid_search_type(mock_services):
    """Test the forward-context endpoint with an invalid search type"""
    response = client.post(
        "/forward-context",
        json={"query": "test query", "search_type": "invalid_type"}
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Invalid search_type" in data["detail"]


def test_ready_endpoint():
    """Test the ready endpoint"""
    with patch('app.server.elasticsearch_service') as mock_es:
        # Test when service is ready
        mock_es.is_ready.return_value = True
        response = client.get("/ready")
        assert response.status_code == 200
        
        # Test when service is not ready
        mock_es.is_ready.return_value = False
        response = client.get("/ready")
        assert response.status_code == 423
