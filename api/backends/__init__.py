# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Backend implementations for Qwen3-TTS.
"""

from .base import TTSBackend
from .factory import (
    get_backend,
    get_batch_scheduler,
    initialize_backend,
    unload_backend,
    update_activity,
    start_auto_unload,
    stop_auto_unload,
    get_inactivity_seconds,
    INACTIVITY_TIMEOUT_MINUTES,
)

__all__ = [
    "TTSBackend",
    "get_backend",
    "get_batch_scheduler",
    "initialize_backend",
    "unload_backend",
    "update_activity",
    "start_auto_unload",
    "stop_auto_unload",
    "get_inactivity_seconds",
    "INACTIVITY_TIMEOUT_MINUTES",
]
