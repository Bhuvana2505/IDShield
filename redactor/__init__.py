


"""
IDShield Redactor Module

This module handles entity detection and document redaction
for sensitive information in Indian government documents.

Components:
- EntityDetector: Pattern matching and NER for sensitive data
- DocumentRedactor: Image redaction and anonymization
- Support for multiple redaction modes
"""

from .entity_detector import EntityDetector
from .document_redactor import DocumentRedactor

__version__ = "1.0.0"
__author__ = "IDShield Team"

__all__ = ['EntityDetector', 'DocumentRedactor']