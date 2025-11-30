"""Translation manager that orchestrates translation services with high-performance optimizations."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from translators import (
    GoogleTranslatorService,
    BingTranslatorService,
    LMStudioTranslatorService,
    CerebrasTranslatorService,
)
from utils.cache import load_cache, append_to_cache, append_batch_to_cache, BatchCacheWriter
from utils.file_handler import load_json_file, save_json_file, finalize_output
from utils.validators import compile_ignore_patterns, should_ignore, has_duplicate_token_error

logger = logging.getLogger(__name__)


class TranslationManager:
    """Manages translation workflow across different services with optimized processing."""
    
    def __init__(
        self,
        service: str,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None,
        batch_size: int = 10,
        ignore_patterns: Optional[List[str]] = None
    ):
        self.service = service.lower()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.api_key = api_key
        self.concurrency = concurrency
        self.batch_size = batch_size
        
        # Compile ignore regex patterns
        self.ignore_patterns = compile_ignore_patterns(ignore_patterns or [])
        if ignore_patterns:
            for pattern in ignore_patterns:
                logger.info(f"Added ignore pattern: {pattern}")
        
        # Initialize translator service
        self.translator = self._create_translator()
        self.google_fallback = None
    
    def _create_translator(self):
        """Create the appropriate translator service."""
        if self.service == 'google':
            return GoogleTranslatorService(
                self.source_lang, self.target_lang, self.concurrency, self.api_key
            )
        elif self.service == 'bing':
            return BingTranslatorService(
                self.source_lang, self.target_lang, self.concurrency, self.api_key
            )
        elif self.service == 'lmstudio':
            return LMStudioTranslatorService(
                self.source_lang, self.target_lang, self.concurrency, self.api_key
            )
        elif self.service == 'cerebras':
            return CerebrasTranslatorService(
                self.source_lang, self.target_lang, self.concurrency, self.api_key, items_per_batch=self.batch_size
            )
        else:
            logger.warning(f"Service '{self.service}' not explicitly supported. Defaulting to Google.")
            return GoogleTranslatorService(
                self.source_lang, self.target_lang, self.concurrency, self.api_key
            )
    
    def _should_ignore(self, text: str) -> bool:
        """Check if text matches any ignore pattern."""
        return should_ignore(text, self.ignore_patterns)
    
    async def _get_google_fallback(self):
        """Get or create Google translator for fallback."""
        if self.google_fallback is None:
            self.google_fallback = GoogleTranslatorService(
                self.source_lang, self.target_lang, self.concurrency, self.api_key
            )
            await self.google_fallback.initialize()
        return self.google_fallback
    
    async def process_file(self, input_file: str, output_file: str, cache_file: str):
        """Process translation file."""
        start_time = time.time()
        
        # Load input data
        try:
            data = load_json_file(input_file)
        except Exception as e:
            logger.error(f"Failed to load input file: {e}")
            return
        
        if not isinstance(data, dict):
            logger.error("Input file must be a JSON object (key-value map).")
            return
        
        # Load cache
        completed_keys = await load_cache(cache_file)
        
        # Identify pending items
        pending_items = {k: v for k, v in data.items() if k not in completed_keys}
        logger.info(
            f"Total items: {len(data)}. "
            f"Completed: {len(completed_keys)}. "
            f"Pending: {len(pending_items)}."
        )
        
        if not pending_items:
            logger.info("All items already translated.")
            await finalize_output(data, cache_file, output_file)
            await self.translator.cleanup()
            return
        
        logger.info(
            f"Starting translation from {self.source_lang} to {self.target_lang} "
            f"using {self.service}..."
        )
        
        try:
            # Initialize translator
            await self.translator.initialize()
            
            # Process based on service type
            if self.service == 'lmstudio':
                await self._process_lmstudio(pending_items, cache_file)
            elif self.service == 'google':
                await self._process_google_optimized(pending_items, cache_file)
            elif self.service == 'cerebras':
                await self._process_cerebras(pending_items, cache_file)
            else:
                await self._process_individual(pending_items, cache_file)
        finally:
            await self.translator.cleanup()
            if self.google_fallback:
                await self.google_fallback.cleanup()
        
        elapsed = time.time() - start_time
        items_per_sec = len(pending_items) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Translation completed in {elapsed:.1f}s "
            f"({items_per_sec:.1f} items/sec)"
        )
        
        await finalize_output(data, cache_file, output_file)
    
    async def _process_lmstudio(self, pending_items: Dict, cache_file: str):
        """Process items using LM Studio."""
        keys = list(pending_items.keys())
        with tqdm(total=len(keys), desc="Translating (LM Studio Batch)", unit="item") as pbar:
            for i in range(0, len(keys), self.batch_size):
                batch_keys = keys[i:i+self.batch_size]
                batch_items = {k: pending_items[k] for k in batch_keys}
                
                items_to_translate = {}
                ignored_items = {}
                for k, v in batch_items.items():
                    if isinstance(v, str) and self._should_ignore(v):
                        ignored_items[k] = v
                    else:
                        items_to_translate[k] = v
                
                translated_batch = {}
                if items_to_translate:
                    translated_batch = await self.translator.translate_batch_dict(
                        items_to_translate
                    )
                
                if not translated_batch and items_to_translate:
                    logger.warning(f"Batch {i//self.batch_size} failed or returned empty.")
                else:
                    for k, v in translated_batch.items():
                        await append_to_cache(cache_file, k, v)
                    for k, v in ignored_items.items():
                        await append_to_cache(cache_file, k, v)
                
                pbar.update(len(batch_keys))
    
    async def _process_google_optimized(self, pending_items: Dict, cache_file: str):
        """
        High-performance Google Translate processing.
        Uses large batches with parallel processing and buffered cache writes.
        """
        keys = list(pending_items.keys())
        values = [pending_items[k] for k in keys]
        
        # Separate items to translate from ignored items
        to_translate_indices: List[int] = []
        to_translate_values: List[str] = []
        ignored_items: List[Tuple[str, str]] = []
        
        for idx, (key, val) in enumerate(zip(keys, values)):
            if isinstance(val, str):
                if self._should_ignore(val):
                    ignored_items.append((key, val))
                else:
                    to_translate_indices.append(idx)
                    to_translate_values.append(val)
            else:
                ignored_items.append((key, val))
        
        logger.info(
            f"Processing {len(to_translate_values)} items to translate, "
            f"{len(ignored_items)} ignored/non-string items"
        )
        
        # Write ignored items in batch
        if ignored_items:
            await append_batch_to_cache(cache_file, ignored_items)
        
        if not to_translate_values:
            return
        
        # Use batch_size from CLI arguments
        batch_size = self.batch_size
        
        # Initialize cache writer
        cache_writer = BatchCacheWriter(
            cache_file, 
            buffer_size=min(500, len(to_translate_values) // 4 + 1),
            flush_interval=3.0
        )
        await cache_writer.start()
        
        try:
            with tqdm(
                total=len(to_translate_values), 
                desc="Translating (Google Optimized)", 
                unit="item"
            ) as pbar:
                for batch_start in range(0, len(to_translate_values), batch_size):
                    batch_end = min(batch_start + batch_size, len(to_translate_values))
                    batch_values = to_translate_values[batch_start:batch_end]
                    batch_indices = to_translate_indices[batch_start:batch_end]
                    
                    # Translate batch using optimized parallel method
                    translated_values = await self.translator.translate_batch(batch_values)
                    
                    # Prepare cache items
                    cache_items = []
                    for i, (idx, translated) in enumerate(zip(batch_indices, translated_values)):
                        key = keys[idx]
                        cache_items.append((key, translated))
                    
                    # Add to buffered cache writer
                    await cache_writer.add_batch(cache_items)
                    
                    pbar.update(len(batch_values))
        finally:
            await cache_writer.stop()
        
        # Print stats if available
        if hasattr(self.translator, 'get_stats'):
            stats = self.translator.get_stats()
            logger.info(f"Translation stats: {stats}")
    
    async def _process_google_batch(self, pending_items: Dict, cache_file: str):
        """Legacy batch processing (kept for compatibility)."""
        await self._process_google_optimized(pending_items, cache_file)
    
    async def _process_cerebras(self, pending_items: Dict, cache_file: str):
        """
        Process items using Cerebras LLM with batch optimization.
        Uses batch translation within single API calls + concurrent requests.
        """
        keys = list(pending_items.keys())
        values = [pending_items[k] for k in keys]
        
        # Separate items to translate from ignored items
        to_translate_indices: List[int] = []
        to_translate_values: List[str] = []
        ignored_items: List[Tuple[str, str]] = []
        
        for idx, (key, val) in enumerate(zip(keys, values)):
            if isinstance(val, str):
                if self._should_ignore(val):
                    ignored_items.append((key, val))
                else:
                    to_translate_indices.append(idx)
                    to_translate_values.append(val)
            else:
                ignored_items.append((key, val))
        
        logger.info(
            f"Cerebras: Processing {len(to_translate_values)} items to translate, "
            f"{len(ignored_items)} ignored/non-string items"
        )
        
        # Write ignored items in batch
        if ignored_items:
            await append_batch_to_cache(cache_file, ignored_items)
        
        if not to_translate_values:
            return
        
        # Initialize cache writer with appropriate buffer size
        cache_writer = BatchCacheWriter(
            cache_file,
            buffer_size=min(200, len(to_translate_values) // 4 + 1),
            flush_interval=3.0
        )
        await cache_writer.start()
        
        try:
            # Process in larger chunks to leverage batch API efficiency
            chunk_size = self.batch_size
            
            with tqdm(
                total=len(to_translate_values),
                desc="Translating (Cerebras)",
                unit="item"
            ) as pbar:
                for chunk_start in range(0, len(to_translate_values), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(to_translate_values))
                    chunk_values = to_translate_values[chunk_start:chunk_end]
                    chunk_indices = to_translate_indices[chunk_start:chunk_end]
                    
                    # Translate chunk using batch method (internally handles batching)
                    translated_values = await self.translator.translate_batch(chunk_values)
                    
                    # Prepare cache items
                    cache_items = []
                    for idx, translated in zip(chunk_indices, translated_values):
                        key = keys[idx]
                        cache_items.append((key, translated))
                    
                    # Add to buffered cache writer
                    await cache_writer.add_batch(cache_items)
                    
                    pbar.update(len(chunk_values))
        finally:
            await cache_writer.stop()
        
        # Print stats
        if hasattr(self.translator, 'get_stats'):
            stats = self.translator.get_stats()
            logger.info(f"Cerebras translation stats: {stats}")
    
    async def _process_individual(self, pending_items: Dict, cache_file: str):
        """Process items individually (for services like Bing)."""
        cache_writer = BatchCacheWriter(cache_file, buffer_size=50, flush_interval=2.0)
        await cache_writer.start()
        
        try:
            with tqdm(total=len(pending_items), desc="Translating", unit="key") as pbar:
                semaphore = asyncio.Semaphore(self.concurrency)
                
                async def process_item(k: str, v):
                    async with semaphore:
                        if isinstance(v, str):
                            if self._should_ignore(v):
                                await cache_writer.add(k, v)
                            else:
                                translated_val = await self.translator.translate_single(v)
                                await cache_writer.add(k, translated_val)
                        else:
                            await cache_writer.add(k, v)
                        pbar.update(1)
                
                tasks = [process_item(k, v) for k, v in pending_items.items()]
                await asyncio.gather(*tasks)
        finally:
            await cache_writer.stop()
