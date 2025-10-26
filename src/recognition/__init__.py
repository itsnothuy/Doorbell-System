#!/usr/bin/env python3
"""
Face Recognition Module

Provides face encoding, similarity matching, caching, and recognition utilities
for the doorbell security system.
"""

from src.recognition.recognition_result import (
    FaceRecognitionResult,
    PersonMatch,
    RecognitionMetadata
)

__all__ = [
    'FaceRecognitionResult',
    'PersonMatch',
    'RecognitionMetadata'
]
