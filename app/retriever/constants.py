"""Constants used throughout the retriever module."""

# Search types
SEARCH_TYPE_SEMANTIC = "semantic"
SEARCH_TYPE_KEYWORD = "keyword"
SEARCH_TYPE_HYBRID = "hybrid"
SEARCH_TYPES = [SEARCH_TYPE_SEMANTIC, SEARCH_TYPE_KEYWORD, SEARCH_TYPE_HYBRID]

# Default query parameters
DEFAULT_SEARCH_LIMIT = 5

# Text search parameters
TITLE_WEIGHT = 3
SUBJECT_WEIGHT = 2
MINIMUM_SHOULD_MATCH = "50%"

# Vector search parameters
DEFAULT_NUM_CANDIDATES = 100
HYBRID_KNN_CANDIDATES = 50  # Used in hybrid search
HYBRID_BOOST_FACTOR = 2.0  # Boost factor for kNN in hybrid search

# Elasticsearch parameters
DEFAULT_ES_URL = "http://localhost:9200"
DEFAULT_ES_INDEX = "statistics"

# Embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
