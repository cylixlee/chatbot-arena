"""
Module to provide some common paths as pathlib.Path objects.
"""

import pathlib

__all__ = ["SOURCE_DIR", "PROJECT_DIR", "CACHE_DIR"]

SOURCE_DIR = pathlib.Path(__file__).parent
PROJECT_DIR = SOURCE_DIR.parent

CACHE_DIR = PROJECT_DIR / "cache"
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)
