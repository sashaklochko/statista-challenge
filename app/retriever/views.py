from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.retriever.constants import DEFAULT_SEARCH_LIMIT, SEARCH_TYPE_HYBRID


class Document(BaseModel):
    id: str
    title: str
    subject: str
    description: str
    link: str
    date: datetime
    teaser_image_url: str
    similarity_score: Optional[float] = None


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        title="Query",
        description="Natural language query to search for",
        min_length=1,
        example="What is the gold price trend in 2024?"
    )
    limit: int = Field(
        DEFAULT_SEARCH_LIMIT,
        title="Result limit",
        description="Maximum number of results to return",
        ge=1,
        le=100,
        example=5
    )
    search_type: str = Field(
        SEARCH_TYPE_HYBRID,
        title="Search type",
        description="Type of search to perform: semantic, keyword, or hybrid",
        example="hybrid"
    )


class QueryResponse(BaseModel):
    results: List[Document] = Field(
        ...,
        title="Results",
        description="List of documents matching the query"
    )
    query: str = Field(
        ...,
        title="Query",
        description="The original query"
    )
    search_type: str = Field(
        ...,
        title="Search type",
        description="Type of search performed"
    )
    search_engine: str = Field(
        "elasticsearch",
        title="Search engine",
        description="The search engine used for retrieval"
    )
    timestamp: datetime = Field(
        ...,
        title="Timestamp",
        description="Timestamp when the query was processed"
    )
    execution_time_ms: float = Field(
        ...,
        title="Execution time",
        description="Time taken to execute the query in milliseconds"
    )
