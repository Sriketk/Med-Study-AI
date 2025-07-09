import os
from unittest.mock import patch

import pytest

from agent import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
@patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
async def test_agent_simple_passthrough() -> None:
    inputs = {"changeme": "some_val"}
    res = await graph.ainvoke(inputs)
    assert res is not None
