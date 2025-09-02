"""
IDShield Utilities Module

This module provides utility functions for file handling,
logging, and system operations.

Components:
- FileHandler: File I/O operations and format conversion
- RedactionLogger: Audit trail and compliance logging
- Helper utilities for document processing
"""

from .file_handler import FileHandler
from .logger import RedactionLogger as Logger

__version__ = "1.0.0"
__author__ = "IDShield Team"

__all__ = ['FileHandler', 'RedactionLogger']