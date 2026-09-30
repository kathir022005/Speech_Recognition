"""
asr_engine.py - Backward Compatibility Shim
============================================

Exports ASREngine and functions from the modern `asr` package.
"""

from asr.engine import (
    ASREngine,
    create_engine,
    AVAILABLE_MODELS,
    SUPPORTED_FORMATS,
)

__all__ = ["ASREngine", "create_engine", "AVAILABLE_MODELS", "SUPPORTED_FORMATS"]
