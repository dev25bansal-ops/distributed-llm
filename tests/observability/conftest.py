from unittest.mock import MagicMock
import pytest


@pytest.fixture
def mock_metrics_registry():
    registry = MagicMock()
    registry.counter = MagicMock(return_value=MagicMock())
    registry.histogram = MagicMock(return_value=MagicMock())
    registry.gauge = MagicMock(return_value=MagicMock())
    return registry
