"""
pytest configuration for mini-sglang tests.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires multi-GPU and HF models)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires special hardware like multi-GPU)",
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip integration tests unless --run-integration is specified."""
    if config.getoption("--run-integration"):
        # Run all tests including integration tests
        return

    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
