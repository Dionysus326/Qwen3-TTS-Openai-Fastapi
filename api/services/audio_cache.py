"""Server-side persistent audio cache for Qwen3-TTS.

Stores generated audio keyed by hash(text, voice, model, speed, format) so
repeated requests skip synthesis entirely. Useful when a mobile client
reconnects and needs previously generated audio.

Speed is rounded to 2 decimal places in the cache key, so 1.0 and 1.001
produce the same entry. This is intentional.
"""

import asyncio
import hashlib
import logging
import os
from typing import Optional, Tuple, List

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)

# Default cache directory (lives inside container, wiped on restart)
CACHE_DIR = os.environ.get("TTS_CACHE_DIR", "/app/audio_cache")
# Max cache size in MB (default 10 GB)
MAX_CACHE_SIZE_MB = int(os.environ.get("TTS_CACHE_MAX_SIZE_MB", "10240"))
# TTL: delete cached files older than this (default 1 hour)
CACHE_TTL_SECONDS = int(os.environ.get("TTS_CACHE_TTL_SECONDS", "3600"))
# Run cleanup every N writes
_CLEANUP_INTERVAL = 100
_write_count = 0
_cleanup_running = False
_periodic_task = None


def _cache_key(text: str, voice: str, speed: float, fmt: str,
               language: str = "Auto", instruct: str = "") -> str:
    """Deterministic hash of all synthesis-affecting parameters."""
    raw = f"{text}|{voice}|{speed:.2f}|{fmt}|{language}|{instruct}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str, fmt: str) -> str:
    """Return the file path for a cache entry (2-level subdirectory)."""
    return os.path.join(CACHE_DIR, key[:2], key[2:4], f"{key}.{fmt}")


async def get_cached(
    text: str, voice: str, speed: float, fmt: str,
    language: str = "Auto", instruct: str = "",
) -> Optional[str]:
    """Return path to cached audio file if it exists, else None."""
    key = _cache_key(text, voice, speed, fmt, language, instruct)
    path = _cache_path(key, fmt)
    if await aiofiles.os.path.exists(path):
        # Touch access time in thread pool so LRU cleanup works
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, os.utime, path)
        except OSError:
            pass
        logger.debug(f"Cache HIT: {key[:12]}...")
        return path
    return None


async def put_cached(
    text: str, voice: str, speed: float, fmt: str, data: bytes,
    language: str = "Auto", instruct: str = "",
) -> str:
    """Write audio data to cache atomically and return the file path."""
    global _write_count
    key = _cache_key(text, voice, speed, fmt, language, instruct)
    path = _cache_path(key, fmt)
    tmp_path = path + ".tmp"
    directory = os.path.dirname(path)
    await aiofiles.os.makedirs(directory, exist_ok=True)
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(data)
    os.replace(tmp_path, path)  # atomic on same filesystem
    logger.info(f"Cache STORE: {key[:12]}... ({len(data)} bytes)")

    _write_count += 1
    if _write_count % _CLEANUP_INTERVAL == 0:
        asyncio.create_task(cleanup_cache())

    return path


def _scan_cache_files() -> Tuple[List[Tuple[str, float, float, int]], int]:
    """Synchronous directory scan (run in thread pool).
    Returns (list of (path, atime, mtime, size), total_size)."""
    import time as _time
    files = []
    total_size = 0
    for root, _dirs, filenames in os.walk(CACHE_DIR):
        for fname in filenames:
            if fname.endswith(".tmp"):
                continue
            fpath = os.path.join(root, fname)
            try:
                stat = os.stat(fpath)
                files.append((fpath, stat.st_atime, stat.st_mtime, stat.st_size))
                total_size += stat.st_size
            except OSError:
                continue
    return files, total_size


async def cleanup_cache() -> None:
    """Remove oldest files if cache exceeds size limit (LRU by access time)."""
    global _cleanup_running
    if _cleanup_running:
        return
    _cleanup_running = True

    try:
        await _do_cleanup()
    finally:
        _cleanup_running = False


async def _do_cleanup() -> None:
    import time as _time

    if not await aiofiles.os.path.exists(CACHE_DIR):
        return

    loop = asyncio.get_running_loop()
    files, total_size = await loop.run_in_executor(None, _scan_cache_files)

    now = _time.time()
    removed = 0

    # Phase 1: Remove expired files (older than TTL based on creation time)
    surviving = []
    for fpath, atime, mtime, fsize in files:
        if now - mtime > CACHE_TTL_SECONDS:
            try:
                await aiofiles.os.remove(fpath)
                total_size -= fsize
                removed += 1
            except OSError:
                surviving.append((fpath, atime, mtime, fsize))
        else:
            surviving.append((fpath, atime, mtime, fsize))

    # Phase 2: If still over size limit, remove oldest-accessed files (LRU)
    max_bytes = MAX_CACHE_SIZE_MB * 1024 * 1024
    if total_size > max_bytes:
        surviving.sort(key=lambda x: x[1])  # sort by atime ascending
        for fpath, _atime, _mtime, fsize in surviving:
            if total_size <= max_bytes:
                break
            try:
                await aiofiles.os.remove(fpath)
                total_size -= fsize
                removed += 1
            except OSError:
                continue

    if removed > 0:
        logger.info(f"Cache cleanup: removed {removed} files, {total_size // (1024*1024)} MB remaining")


async def start_periodic_cleanup() -> None:
    """Start a background task that runs cleanup every TTL interval."""
    global _periodic_task

    async def _loop():
        while True:
            await asyncio.sleep(CACHE_TTL_SECONDS)
            await cleanup_cache()

    _periodic_task = asyncio.create_task(_loop())
    logger.info(f"Cache periodic cleanup started (every {CACHE_TTL_SECONDS}s)")


async def stop_periodic_cleanup() -> None:
    """Stop the periodic cleanup task."""
    global _periodic_task
    if _periodic_task and not _periodic_task.done():
        _periodic_task.cancel()
        try:
            await _periodic_task
        except asyncio.CancelledError:
            pass
    _periodic_task = None
