"""Cache management utilities with optimized batch operations."""

import json
import os
from typing import Set, Dict, List, Tuple
import asyncio

import aiofiles
import logging

logger = logging.getLogger(__name__)


async def load_cache(cache_file: str) -> Set[str]:
    """Load completed translation keys from cache file."""
    completed_keys = set()
    if not os.path.exists(cache_file):
        return completed_keys
    
    logger.info(f"Loading cache from {cache_file}...")
    try:
        async with aiofiles.open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
            content = await f.read()
            for line in content.splitlines():
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if isinstance(entry, dict):
                            completed_keys.update(entry.keys())
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Error reading cache file: {e}")
    
    logger.info(f"Loaded {len(completed_keys)} entries from cache.")
    return completed_keys


async def append_to_cache(cache_file: str, key: str, value: str):
    """Append a single key-value pair to the cache file."""
    try:
        entry = {key: value}
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        async with aiofiles.open(cache_file, 'a', encoding='utf-8') as f:
            await f.write(line)
    except Exception as e:
        logger.error(f"Error writing to cache: {e}")


async def append_batch_to_cache(cache_file: str, items: List[Tuple[str, str]]):
    """
    Append multiple key-value pairs to the cache file in a single I/O operation.
    Much faster than individual writes for large batches.
    
    Args:
        cache_file: Path to cache file
        items: List of (key, value) tuples to append
    """
    if not items:
        return
    
    try:
        lines = []
        for key, value in items:
            entry = {key: value}
            lines.append(json.dumps(entry, ensure_ascii=False))
        
        content = "\n".join(lines) + "\n"
        async with aiofiles.open(cache_file, 'a', encoding='utf-8') as f:
            await f.write(content)
    except Exception as e:
        logger.error(f"Error writing batch to cache: {e}")


class BatchCacheWriter:
    """
    High-performance cache writer with batching and buffering.
    Reduces I/O operations by buffering writes and flushing periodically.
    """
    
    def __init__(
        self, 
        cache_file: str, 
        buffer_size: int = 100,
        flush_interval: float = 2.0
    ):
        self.cache_file = cache_file
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._buffer: List[Tuple[str, str]] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task = None
        self._running = False
    
    async def start(self):
        """Start the background flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
    
    async def stop(self):
        """Stop the writer and flush remaining items."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()
    
    async def add(self, key: str, value: str):
        """Add item to buffer, auto-flush if buffer is full."""
        async with self._lock:
            self._buffer.append((key, value))
            if len(self._buffer) >= self.buffer_size:
                await self._flush_internal()
    
    async def add_batch(self, items: List[Tuple[str, str]]):
        """Add multiple items to buffer."""
        async with self._lock:
            self._buffer.extend(items)
            if len(self._buffer) >= self.buffer_size:
                await self._flush_internal()
    
    async def flush(self):
        """Flush buffer to disk."""
        async with self._lock:
            await self._flush_internal()
    
    async def _flush_internal(self):
        """Internal flush without lock (caller must hold lock)."""
        if not self._buffer:
            return
        
        items_to_write = self._buffer.copy()
        self._buffer.clear()
        
        try:
            lines = []
            for key, value in items_to_write:
                entry = {key: value}
                lines.append(json.dumps(entry, ensure_ascii=False))
            
            content = "\n".join(lines) + "\n"
            async with aiofiles.open(self.cache_file, 'a', encoding='utf-8') as f:
                await f.write(content)
        except Exception as e:
            logger.error(f"Error flushing cache buffer: {e}")
            # Put items back in buffer on failure
            self._buffer = items_to_write + self._buffer
    
    async def _periodic_flush(self):
        """Periodically flush buffer."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
    
    @property
    def pending_count(self) -> int:
        """Get number of pending items in buffer."""
        return len(self._buffer)


async def load_translated_map(cache_file: str) -> Dict:
    """Load all translated entries from cache file into a dictionary."""
    translated_map = {}
    if os.path.exists(cache_file):
        try:
            async with aiofiles.open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
                content = await f.read()
                for line in content.splitlines():
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            translated_map.update(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Error reading cache for finalization: {e}")
    return translated_map
