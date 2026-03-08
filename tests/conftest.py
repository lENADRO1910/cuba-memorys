"""Shared fixtures for cuba-memorys test suite."""
import pytest


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample corpus for TF-IDF testing."""
    return [
        "Python is a programming language used for machine learning",
        "FastAPI provides async web framework for building APIs",
        "PostgreSQL is a relational database management system",
        "Docker containers provide isolated runtime environments",
        "Redis is an in-memory data structure store",
    ]


@pytest.fixture
def sample_rankings() -> list[list[dict]]:
    """Sample rankings for RRF fusion testing."""
    return [
        [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.7},
            {"id": "c", "score": 0.3},
        ],
        [
            {"id": "b", "score": 0.95},
            {"id": "a", "score": 0.6},
            {"id": "d", "score": 0.4},
        ],
    ]
