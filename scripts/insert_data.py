#!/usr/bin/env python
import json
import os
import time
import logging
import datetime
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("es-loader")

# Elasticsearch configuration
ES_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.environ.get("ELASTICSEARCH_INDEX", "statistics")
DATA_PATH = os.environ.get("DATA_PATH", "app/data/statistics.json")

# Vector dimension for the all-MiniLM-L6-v2 model
VECTOR_DIMENSION = 384


def wait_for_elasticsearch(es_client, max_retries=10, delay=5):
    """Wait for Elasticsearch to be available."""
    for i in range(max_retries):
        try:
            if es_client.ping():
                logger.info("Elasticsearch is up and running!")
                return True
        except ConnectionError:
            logger.warning(
                f"Elasticsearch not available yet, retrying in {delay} seconds..."
            )
            time.sleep(delay)

    logger.error(f"Could not connect to Elasticsearch after {max_retries} retries")
    return False


def create_index(es_client):
    """Create the Elasticsearch index with appropriate mappings."""
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text"},
                "subject": {"type": "text"},
                "description": {"type": "text"},
                "link": {"type": "keyword"},
                "date": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis",
                },
                "teaser_image_url": {"type": "keyword"},
                "content_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMENSION,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }

    if es_client.indices.exists(index=ES_INDEX):
        logger.info(f"Index {ES_INDEX} already exists, deleting it...")
        es_client.indices.delete(index=ES_INDEX)

    logger.info(f"Creating index {ES_INDEX}...")
    es_client.indices.create(index=ES_INDEX, body=index_settings)
    logger.info(f"Index {ES_INDEX} created successfully!")


def load_data(file_path):
    """Load data from JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} documents from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []


def format_date(date_str):
    """Convert date string to ISO format that Elasticsearch can parse."""
    try:
        # Remove timezone part if exists
        if "+" in date_str:
            date_str = date_str.split("+")[0]

        # Parse the date
        dt = datetime.datetime.fromisoformat(date_str.replace("Z", ""))

        # Return in ISO format with 'Z' for UTC
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        logger.error(f"Error formatting date {date_str}: {e}")
        # Return a fallback date if parsing fails
        return "2023-01-01T00:00:00Z"


def index_documents(es_client, documents):
    """Index documents in Elasticsearch."""
    try:
        from sentence_transformers import SentenceTransformer

        # Initialize embedding model
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create bulks for indexing
        actions = []
        for i, doc in enumerate(documents):
            try:
                # Create embeddings for the document
                text_to_embed = f"{doc['title']} {doc['subject']} {doc['description']}"
                vector = model.encode(text_to_embed).tolist()

                # Create a copy of the document to modify
                doc_copy = doc.copy()

                # Fix date format
                if "date" in doc_copy:
                    original_date = doc_copy["date"]
                    doc_copy["date"] = format_date(original_date)
                    logger.debug(
                        f"Document {doc_copy['id']}: Converted date from '{original_date}' to '{doc_copy['date']}'"
                    )

                # Prepare document for indexing
                action = {
                    "_index": ES_INDEX,
                    "_id": str(doc_copy["id"]),
                    "_source": {**doc_copy, "content_vector": vector},
                }
                actions.append(action)

                # Log progress
                if (i + 1) % 100 == 0:
                    logger.info(
                        f"Prepared {i + 1}/{len(documents)} documents for indexing"
                    )

            except Exception as e:
                logger.error(f"Error preparing document {doc.get('id', i)}: {e}")
                logger.error(f"Document content: {json.dumps(doc)[:200]}...")

        # Bulk index with detailed error tracking
        logger.info(f"Indexing {len(actions)} documents...")

        # Use the bulk API with error tracking
        try:
            results = helpers.bulk(
                es_client, actions, stats_only=False, raise_on_error=False
            )
            success_count = results[0]
            failed_items = []

            if len(results) > 1 and isinstance(results[1], list):
                failed_items = results[1]

            logger.info(f"Indexed {success_count} documents successfully")

            if failed_items:
                logger.error(f"Failed to index {len(failed_items)} documents")
                for item in failed_items[:10]:  # Show first 10 failures
                    doc_id = item.get("index", {}).get("_id", "unknown")
                    error = item.get("index", {}).get("error", {})
                    error_type = error.get("type", "unknown")
                    error_reason = error.get("reason", "unknown")
                    logger.error(
                        f"Document {doc_id} failed: {error_type} - {error_reason}"
                    )
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")

    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        logger.exception("Detailed traceback:")


def main():
    logger.info(f"Connecting to Elasticsearch at {ES_URL}...")
    es_client = Elasticsearch(ES_URL)

    if not wait_for_elasticsearch(es_client):
        logger.error("Elasticsearch not available, exiting.")
        return

    create_index(es_client)

    documents = load_data(DATA_PATH)
    if not documents:
        logger.error("No documents loaded, exiting.")
        return

    index_documents(es_client, documents)

    logger.info("Data loading complete!")


if __name__ == "__main__":
    main()
