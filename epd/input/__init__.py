"""
Inputs module.

This module contains the definitions for reading the logs later used by the drift detection algorithm.
A specific implementation for parsing CSV files from apromore is included.
All log parsers are expected to return an iterator, which will be used later by the detection algorithm.
"""

from .mapping import Mapping

__all__ = ['Mapping', "csv", "json"]
