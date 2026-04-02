# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating TTS backend instances.
"""

import asyncio
import os
import logging
import time
from typing import Optional

from .base import TTSBackend
from .official_qwen3_tts import OfficialQwen3TTSBackend
from .vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend
from .batch_scheduler import BatchScheduler

logger = logging.getLogger(__name__)

# Global backend instance
_backend_instance: Optional[TTSBackend] = None
# Global batch scheduler (replaces model pool)
_batch_scheduler: Optional[BatchScheduler] = None
# Lifecycle lock to prevent concurrent init/unload
_lifecycle_lock: Optional[asyncio.Lock] = None

def _get_lifecycle_lock() -> asyncio.Lock:
    global _lifecycle_lock
    if _lifecycle_lock is None:
        _lifecycle_lock = asyncio.Lock()
    return _lifecycle_lock

# Activity tracking for auto-unload
_last_activity_time: float = 0.0
_auto_unload_task: Optional[asyncio.Task] = None
_auto_unload_enabled: bool = False

# Configuration
INACTIVITY_TIMEOUT_MINUTES = int(os.getenv("TTS_INACTIVITY_TIMEOUT_MINUTES", "0"))  # 0 = disabled


def update_activity() -> None:
    """Update the last activity timestamp. Called on each TTS request."""
    global _last_activity_time
    _last_activity_time = time.time()


def get_inactivity_seconds() -> float:
    """Get seconds since last activity."""
    if _last_activity_time == 0:
        return 0
    return time.time() - _last_activity_time


def get_backend() -> TTSBackend:
    """Get or create the global TTS backend instance."""
    global _backend_instance

    if _backend_instance is not None:
        return _backend_instance

    backend_type = os.getenv("TTS_BACKEND", "official").lower()
    model_name = os.getenv("TTS_MODEL_NAME")

    logger.info(f"Initializing TTS backend: {backend_type}")

    if backend_type == "official":
        if model_name:
            _backend_instance = OfficialQwen3TTSBackend(model_name=model_name)
        else:
            _backend_instance = OfficialQwen3TTSBackend()
        logger.info(f"Using official Qwen3-TTS backend with model: {_backend_instance.get_model_id()}")

    elif backend_type in ("vllm_omni", "vllm-omni", "vllm"):
        if model_name:
            _backend_instance = VLLMOmniQwen3TTSBackend(model_name=model_name)
        else:
            _backend_instance = VLLMOmniQwen3TTSBackend(
                model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            )
        logger.info(f"Using vLLM-Omni backend with model: {_backend_instance.get_model_id()}")

    else:
        raise ValueError(
            f"Unknown TTS_BACKEND: {backend_type}. Supported: 'official', 'vllm_omni'"
        )

    return _backend_instance


def get_batch_scheduler() -> Optional[BatchScheduler]:
    """Get the global batch scheduler (only for official backend)."""
    return _batch_scheduler


async def initialize_backend(warmup: bool = False) -> TTSBackend:
    """Initialize the backend. For official backend, loads 1 model and starts batch scheduler."""
    global _batch_scheduler

    async with _get_lifecycle_lock():
        # Skip if already initialized
        if _batch_scheduler is not None and _batch_scheduler.is_running:
            return get_backend()

        backend = get_backend()
        backend_type = os.getenv("TTS_BACKEND", "official").lower()

        if backend_type == "official":
            await backend.initialize()
            _batch_scheduler = BatchScheduler(backend)
            await _batch_scheduler.start()
            logger.info("Batch scheduler started on single model instance")
        else:
            # vllm: single instance, no batching
            await backend.initialize()

            if warmup:
                warmup_enabled = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
                if warmup_enabled:
                    logger.info("Performing backend warmup...")
                    try:
                        await backend.generate_speech(
                            text="Hello, this is a warmup test.",
                            voice="Vivian",
                            language="English",
                        )
                        logger.info("Backend warmup completed successfully")
                    except Exception as e:
                        logger.warning(f"Backend warmup failed (non-critical): {e}")

        return backend


def reset_backend() -> None:
    """Reset the global backend instance (useful for testing)."""
    global _backend_instance
    _backend_instance = None


async def unload_backend() -> bool:
    """Unload the TTS backend to free GPU VRAM."""
    global _backend_instance, _batch_scheduler, _last_activity_time

    async with _get_lifecycle_lock():
        unloaded = False

        if _batch_scheduler is not None:
            try:
                await _batch_scheduler.stop()
                _batch_scheduler = None
                logger.info("Batch scheduler stopped")
            except Exception as e:
                logger.error(f"Failed to stop batch scheduler: {e}")

        if _backend_instance is not None and _backend_instance.is_ready():
            try:
                await _backend_instance.unload()
                unloaded = True
                logger.info("Backend unloaded successfully")
            except Exception as e:
                logger.error(f"Failed to unload backend: {e}")

        if unloaded:
            _last_activity_time = 0.0
        else:
            logger.info("Nothing to unload")

        return unloaded


async def _auto_unload_checker() -> None:
    """Background task that checks for inactivity and unloads the model."""
    global _auto_unload_enabled

    timeout_seconds = INACTIVITY_TIMEOUT_MINUTES * 60
    check_interval = min(60, timeout_seconds / 4)

    logger.info(f"Auto-unload checker started (timeout: {INACTIVITY_TIMEOUT_MINUTES} minutes)")

    while _auto_unload_enabled:
        try:
            await asyncio.sleep(check_interval)

            # Skip if nothing is loaded
            backend_active = _backend_instance is not None and _backend_instance.is_ready()
            if not backend_active:
                continue

            if _last_activity_time == 0:
                continue

            inactivity = get_inactivity_seconds()

            if inactivity >= timeout_seconds:
                logger.info(
                    f"Model inactive for {inactivity/60:.1f} minutes "
                    f"(threshold: {INACTIVITY_TIMEOUT_MINUTES} minutes) - unloading"
                )
                await unload_backend()
            else:
                remaining = (timeout_seconds - inactivity) / 60
                logger.debug(f"Model active - will unload in {remaining:.1f} minutes if idle")

        except asyncio.CancelledError:
            logger.info("Auto-unload checker cancelled")
            break
        except Exception as e:
            logger.error(f"Error in auto-unload checker: {e}")
            await asyncio.sleep(10)


async def start_auto_unload() -> None:
    """Start the auto-unload background task if configured."""
    global _auto_unload_task, _auto_unload_enabled

    if INACTIVITY_TIMEOUT_MINUTES <= 0:
        logger.info("Auto-unload disabled (TTS_INACTIVITY_TIMEOUT_MINUTES not set or 0)")
        return

    if _auto_unload_task is not None and not _auto_unload_task.done():
        logger.info("Auto-unload checker already running")
        return

    _auto_unload_enabled = True
    _auto_unload_task = asyncio.create_task(_auto_unload_checker())
    logger.info(f"Auto-unload enabled: model will unload after {INACTIVITY_TIMEOUT_MINUTES} minutes of inactivity")


async def stop_auto_unload() -> None:
    """Stop the auto-unload background task."""
    global _auto_unload_task, _auto_unload_enabled

    _auto_unload_enabled = False

    if _auto_unload_task is not None:
        _auto_unload_task.cancel()
        try:
            await _auto_unload_task
        except asyncio.CancelledError:
            pass
        _auto_unload_task = None
        logger.info("Auto-unload checker stopped")
