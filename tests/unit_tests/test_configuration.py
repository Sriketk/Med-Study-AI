import os
from unittest.mock import patch

import pytest
from langgraph.pregel import Pregel

from agent.graph import graph


def test_placeholder() -> None:
    # TODO: You can add actual unit tests
    # for your graph and other logic here.
    assert isinstance(graph, Pregel)


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
def test_graph_initialization_with_api_key() -> None:
    """Test that the graph can be initialized when API key is provided."""
    # This should not raise an error
    assert isinstance(graph, Pregel)


def test_graph_initialization_without_api_key() -> None:
    """Test that the graph initialization fails without API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            # This should trigger the import and raise the error
            import importlib

            import agent.graph
            importlib.reload(agent.graph)
