"""
Pytest configuration file with shared fixtures
"""

import pytest
import numpy as np
from datetime import datetime


@pytest.fixture
def sample_docs():
    """Sample document data for testing"""
    return [
        {
            "id": "1",
            "title": "AI Unicorn Companies",
            "subject": "Technology",
            "description": "Statistics about AI startups valued at over $1 billion",
            "link": "https://statista.com/statistics/1",
            "date": datetime.now(),
            "teaser_image_url": "https://statista.com/images/1.jpg"
        },
        {
            "id": "2",
            "title": "Smartphone Market Trends",
            "subject": "Consumer Electronics",
            "description": "Latest statistics on global smartphone sales",
            "link": "https://statista.com/statistics/2",
            "date": datetime.now(),
            "teaser_image_url": "https://statista.com/images/2.jpg"
        },
        {
            "id": "3",
            "title": "Renewable Energy Growth",
            "subject": "Energy",
            "description": "Statistics on global renewable energy adoption",
            "link": "https://statista.com/statistics/3",
            "date": datetime.now(),
            "teaser_image_url": "https://statista.com/images/3.jpg"
        }
    ]


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing"""
    return np.array([0.1, 0.2, 0.3, 0.4]) 