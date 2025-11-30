"""Cache management utilities."""

import json
import os
from typing import Set, Dict

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
        # Use errors='replace' to handle invalid UTF-8 bytes gracefully
        async with aiofiles.open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
            async for line in f:
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
    """Append a key-value pair to the cache file."""
    try:
        entry = {key: value}
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        async with aiofiles.open(cache_file, 'a', encoding='utf-8') as f:
            await f.write(line)
    except Exception as e:
        logger.error(f"Error writing to cache: {e}")


async def load_translated_map(cache_file: str) -> Dict:
    """Load all translated entries from cache file into a dictionary."""
    translated_map = {}
    if os.path.exists(cache_file):
        try:
            async with aiofiles.open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
                async for line in f:
                    try:
                        entry = json.loads(line)
                        translated_map.update(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading cache for finalization: {e}")
    return translated_map

