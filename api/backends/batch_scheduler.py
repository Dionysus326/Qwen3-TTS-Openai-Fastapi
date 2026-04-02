"""Batch inference scheduler for Qwen3-TTS.

Collects concurrent requests into batches and runs them through a single
model forward pass. Uses the future-per-request pattern: each HTTP handler
gets a Future that resolves when the batch completes.

Batch fires when max_batch_size items are queued OR max_wait_ms has elapsed
since the first item entered the current batch (whichever comes first).
Items are grouped by (voice, language, instruct) before batching.
"""

import asyncio
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BATCH_MAX_SIZE = int(os.getenv("TTS_BATCH_MAX_SIZE", "4"))
BATCH_MAX_WAIT_MS = int(os.getenv("TTS_BATCH_MAX_WAIT_MS", "50"))


@dataclass
class BatchItem:
    text: str
    voice: str
    language: str
    instruct: Optional[str]
    speed: float
    future: asyncio.Future = field(default_factory=lambda: None)


class BatchScheduler:
    """Batches concurrent TTS requests into single model forward passes."""

    def __init__(self, backend):
        self._backend = backend
        self._queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self):
        self._shutdown = False
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info(
            f"BatchScheduler started (max_batch={BATCH_MAX_SIZE}, "
            f"max_wait={BATCH_MAX_WAIT_MS}ms)"
        )

    async def stop(self):
        """Stop the scheduler. Fails any pending futures gracefully."""
        self._shutdown = True
        # Push sentinel to unblock the loop
        await self._queue.put(None)
        if self._processor_task and not self._processor_task.done():
            try:
                await asyncio.wait_for(self._processor_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass
        # Drain any remaining items
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item is not None and item.future and not item.future.done():
                item.future.set_exception(RuntimeError("Scheduler stopped"))
        logger.info("BatchScheduler stopped")

    async def submit(
        self, text: str, voice: str, language: str,
        instruct: Optional[str], speed: float,
    ) -> Tuple[np.ndarray, int]:
        if self._shutdown:
            raise RuntimeError("BatchScheduler is stopped")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = BatchItem(
            text=text, voice=voice, language=language,
            instruct=instruct, speed=speed, future=future,
        )
        await self._queue.put(item)
        return await future

    async def _process_loop(self):
        batch: List[BatchItem] = []
        try:
            while not self._shutdown:
                # Wait for first item
                first = await self._queue.get()
                if first is None:  # sentinel
                    break
                batch = [first]

                # Collect more within time window
                deadline = asyncio.get_event_loop().time() + BATCH_MAX_WAIT_MS / 1000
                while len(batch) < BATCH_MAX_SIZE:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                        if item is None:  # sentinel
                            self._shutdown = True
                            break
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    logger.info(f"Executing batch of {len(batch)} item(s)")
                    await self._execute_batch(batch)
                    batch = []

        except asyncio.CancelledError:
            pass
        finally:
            # Fail any items that were dequeued but not resolved
            for item in batch:
                if item.future and not item.future.done():
                    item.future.set_exception(RuntimeError("Scheduler stopped"))

    async def _execute_batch(self, batch: List[BatchItem]):
        """Run batch, grouping by (voice, language, instruct) for correctness."""
        if not self._backend.is_ready():
            try:
                await self._backend.initialize()
            except Exception as e:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
                return

        # Group items by synthesis parameters
        groups: Dict[tuple, List[BatchItem]] = defaultdict(list)
        for item in batch:
            key = (item.voice, item.language, item.instruct or "")
            groups[key].append(item)

        for (voice, language, instruct), group_items in groups.items():
            try:
                texts = [item.text for item in group_items]
                inst = instruct if instruct else None

                loop = asyncio.get_running_loop()
                wavs, sr = await loop.run_in_executor(
                    None,
                    self._backend.generate_batch_sync,
                    texts, voice, language, inst,
                )

                for i, item in enumerate(group_items):
                    if item.future.done():
                        continue
                    try:
                        audio = wavs[i]
                        if item.speed != 1.0:
                            try:
                                import librosa
                                audio = librosa.effects.time_stretch(
                                    audio.astype(np.float32), rate=item.speed
                                )
                            except ImportError:
                                logger.warning("librosa not installed, speed adjustment skipped")
                        item.future.set_result((audio, sr))
                    except Exception as e:
                        item.future.set_exception(
                            RuntimeError(f"Post-processing failed: {e}")
                        )

            except Exception as e:
                logger.error(f"Batch group execution failed: {e}")
                for item in group_items:
                    if not item.future.done():
                        item.future.set_exception(
                            RuntimeError(f"Batch inference failed: {e}")
                        )

    @property
    def is_running(self) -> bool:
        return self._processor_task is not None and not self._processor_task.done()
