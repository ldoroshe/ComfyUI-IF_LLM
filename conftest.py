"""Prevent pytest from importing the project's __init__.py by temporarily renaming it."""

import os
import shutil


_INIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__init__.py')
_BAK_PATH = _INIT_PATH + '.bak'


def pytest_configure(config):
    """Rename __init__.py before test collection to prevent import errors."""
    if os.path.exists(_INIT_PATH):
        shutil.move(_INIT_PATH, _BAK_PATH)


def pytest_unconfigure(config):
    """Restore __init__.py after tests complete."""
    if os.path.exists(_BAK_PATH):
        shutil.move(_BAK_PATH, _INIT_PATH)
