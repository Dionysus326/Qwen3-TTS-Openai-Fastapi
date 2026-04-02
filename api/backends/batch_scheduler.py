"""Batch inference scheduler for Qwen3-TTS.

Collects concurrent requests into batches and runs them through a single
model forward pass. Uses the future-per-request pattern: each HTTP handler
gets a Future that resolves when the batch completes.

Batch fires when max_batch_size items are queued OR max_wait_ms has elapsed
since the first item entered the current batch (whichever comes first).
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Config from env
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
        """
        Args:
            backend: An initialized OfficialQwen3TTSBackend instance.
        """
        self._backend = backend
        self._queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self):
        """Start the background batch processor loop."""
        self._shutdown = False
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info(
            f"BatchScheduler started (max_batch={BATCH_MAX_SIZE}, "
            f"max_wait={BATCH_MAX_WAIT_MS}ms)"
        )

    async def stop(self):
        """Stop the scheduler and cancel pending items."""
        self._shutdown = True
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        # Fail any remaining items in the queue
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item.future and not item.future.done():
                item.future.set_exception(RuntimeError("Scheduler stopped"))
        logger.info("BatchScheduler stopped")

    async def submit(
        self,
        text: str,
        voice: str,
        language: str,
        instruct: Optional[str],
        speed: float,
    ) -> Tuple[np.ndarray, int]:
        """Submit a single TTS request. Blocks until the batch completes.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self._shutdown:
            raise RuntimeError("BatchScheduler is stopped")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = BatchItem(
            text=text,
            voice=voice,
            language=language,
            instruct=instruct,
            speed=speed,
            future=future,
        )
        await self._queue.put(item)
        return await future

    async def _process_loop(self):
        """Main loop: collect items into batches, execute them."""
        while not self._shutdown:
            try:
                # Wait for at least one item
                first = await self._queue.get()
                batch = [first]

                # Collect more items within the time window
                deadline = asyncio.get_event_loop().time() + BATCH_MAX_WAIT_MS / 1000
                while len(batch) < BATCH_MAX_SIZE:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                logger.info(f"Executing batch of {len(batch)} item(s)")
                await self._execute_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor loop: {e}")
                await asyncio.sleep(0.1)

    async def _execute_batch(self, batch: List[BatchItem]):
        """Run a batch through the model and resolve each item's future."""
        # Ensure model is loaded
        if not self._backend.is_ready():
            try:
                await self._backend.initialize()
            except Exception as e:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
                return

        try:
            texts = [item.text for item in batch]
            voice = batch[0].voice
            language = batch[0].language
            instruct = batch[0].instruct

            # Run batch inference in thread executor (model call is synchronous)
            loop = asyncio.get_running_loop()
            wavs, sr = await loop.run_in_executor(
                None,
                self._backend.generate_batch_sync,
                texts, voice, language, instruct,
            )

            # Fan out results + per-item speed adjustment
            for i, item in enumerate(batch):
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
                            pass
                    item.future.set_result((audio, sr))
                except Exception as e:
                    item.future.set_exception(
                        RuntimeError(f"Post-processing failed for batch item {i}: {e}")
                    )

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(
                        RuntimeError(f"Batch inference failed: {e}")
                    )

    @property
    def is_running(self) -> bool:
        return self._processor_task is not None and not self._processor_task.done()
