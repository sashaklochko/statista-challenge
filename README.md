# Statista Context API

A lightweight web service that provides relevant Statista data based on natural language queries. The API uses Elasticsearch for efficient search and retrieval, combining vector embeddings and keyword search techniques.

## Features

- Multiple search modes:
  - **Semantic Search**: Vector-based similarity using sentence embeddings
  - **Keyword Search**: Traditional text search with BM25 ranking
  - **Hybrid Search**: Combines semantic and keyword search
- Elasticsearch integration for efficient indexing and retrieval
- Fast response times with pre-computed embeddings

## Project Structure

- `app/` - Main application code
  - `server.py` - FastAPI server implementation
  - `retriever/` - Search and embedding services
  - `data/` - Sample statistics data (JSON)
- `scripts/` - Utility scripts for data loading
- `test_query.py` - Sample client for testing the API
- `Makefile` - Convenience commands for setup and running

## Setup Instructions

This is the quickest way to get started with a fully working environment.

1. Make sure Docker and Docker Compose are installed on your system

2. Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd statista-challenge
```

3. Build and start the Docker Compose stack:
```bash
docker-compose up --build
```

This will:
- Start Elasticsearch in a container
- Load sample statistical data into Elasticsearch with vector embeddings
- Start the Statista API application

The API will be available at http://localhost:9000 once initialization is complete.

### Setup with Make (Recommended for local development)

The project includes a Makefile with helpful commands for setting up and running the application.

1. Make sure you have Python 3.11+ installed

2. Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd statista-challenge
```

3. Prepare the environment (creates a virtual environment and installs dependencies):
```bash
make prepare-env
```

4. Run Elasticsearch
```bash
docker compose up --build
```

5. Load sample data into Elasticsearch:
```bash
make load-data
```

6. Start the application:
```bash
make run-server
```

The API will be available at http://localhost:9000.


## Using the API

### Health Check
```
GET /ready
```

Returns 200 OK if the service is ready to accept requests.

### Get Relevant Context
```
POST /forward-context
```

Request body:
```json
{
  "query": "Your natural language query here",
  "limit": 5,  // Optional, defaults to 5
  "search_type": "hybrid"  // Optional: "semantic", "keyword", or "hybrid"
}
```

Example response:
```json
{
  "results": [
    {
      "id": "1557288",
      "title": "New and total AI unicorns by region/country 2024",
      "subject": "New and total artificial intelligence (AI) unicorns worldwide...",
      "description": "...",
      "link": "https://www.statista.com/statistics/1557288/...",
      "date": "2025-01-29T23:00:00+00:00",
      "teaser_image_url": "https://api.statista.ai/image-service/...",
      "similarity_score": 0.8764
    },
    // More results...
  ],
  "query": "Your natural language query here",
  "search_type": "hybrid",
  "search_engine": "elasticsearch",
  "timestamp": "2024-07-01T12:34:56.789",
  "execution_time_ms": 120.45
}
```

```
curl -X POST -H "Content-Type: application/json" -d '{"query": "Your natural language query here", "limit": 5, "search_type": "hybrid"}' localhost:9000/forward-context
```

## Testing with Sample Queries

A test script is provided to help you verify the API is working correctly:

```bash
make test-query
```

This will run a set of sample queries using different search types.

Run unit test

```bash
make run-test
```

## Search Types and Algorithms

- **semantic**: Uses vector similarity (cosine similarity) between query and document embeddings
- **keyword**: Uses traditional text-based search with Elasticsearch BM25 algorithm
- **hybrid**: Combines both semantic and keyword search

## Elasticsearch Integration

The system uses Elasticsearch to:
- Store all statistical documents with their metadata
- Store vector embeddings for each document (using the all-MiniLM-L6-v2 model)
- Perform efficient vector search with k-Nearest Neighbors (kNN)
- Execute text search with BM25 ranking