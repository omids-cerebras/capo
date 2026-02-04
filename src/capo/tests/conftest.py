"""Pytest configuration for CAPO tests.

This conftest.py ensures that the vendored VERL is available when running tests.
"""

import sys
from pathlib import Path


def pytest_configure(config):
    """Add vendored VERL to path before tests run."""
    # Add experiments directory to path so vendored verl is importable
    experiments_dir = Path(__file__).parent.parent / "experiments"
    if experiments_dir.exists():
        experiments_path = str(experiments_dir)
        if experiments_path not in sys.path:
            sys.path.insert(0, experiments_path)
