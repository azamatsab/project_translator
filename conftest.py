import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="skip slow tests"
    )
    parser.addoption(
        "--skip-integration", action="store_true", default=False, help="skip integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration_test: mark test as integration test to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        # # --runslow given in cli: do not skip slow tests
        # return
        skip_slow = pytest.mark.skip(reason="need to remove --skip-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if config.getoption("--skip-integration"):
        # # --runslow given in cli: do not skip slow tests
        # return
        skip_slow = pytest.mark.skip(reason="need to remove --skip-integration option to run")
        for item in items:
            if "integration_test" in item.keywords:
                item.add_marker(skip_slow)
