import os
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-api-key-for-testing",
        "LANGSMITH_API_KEY": "test-langsmith-key",
        "LANGSMITH_TRACING": "false",
    }):
        yield
