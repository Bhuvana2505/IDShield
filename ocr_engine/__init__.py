"""
IDShield OCR Engine Module

This module handles text extraction and optical character recognition
for Indian government documents.

Components:
- TextExtractor: Main OCR processing using Tesseract
- Image preprocessing utilities
- Multi-language support preparation
"""

from .text_extractor import TextExtractor

__version__ = "1.0.0"
__author__ = "IDShield Team"

__all__ = ['TextExtractor']