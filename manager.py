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
    NvidiaOpenAITranslatorService,
)
from utils.cache import load_cache, append_to_cache, append_batch_to_cache, BatchCacheWriter
from utils.file_handler import load_json_file, save_json_file, finalize_output
from utils.validators import compile_ignore_patterns, should_ignore, has_duplicate_token_error

logger = logging.getLogger(__name__)


class TranslationManager:
    """Manages translation workflow across different services with optimized processing."""
    
    def __init__(
        self,
        services: List[str],  # Changed to accept list of services
        source_lang: str,
        target_lang: str,
        service_concurrency: Optional[Dict[str, int]] = None,
        api_key: Optional[str] = None,
        api_key_cerebras: Optional[str] = None,
        api_key_nvidia: Optional[str] = None,
        service_batch_size: Optional[Dict[str, int]] = None,
        ignore_patterns: Optional[List[str]] = None,
        # Backward compatibility parameters
        concurrency: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        # Support both single service (backward compatibility) and multiple services
        if isinstance(services, str):
            services = [services]
        self.services = [s.lower().strip() for s in services if s.strip()]
        if not self.services:
            raise ValueError("At least one service must be specified")
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.api_key = api_key
        self.api_key_cerebras = api_key_cerebras
        self.api_key_nvidia = api_key_nvidia
        
        # Handle per-service configurations with backward compatibility
        if service_concurrency is None:
            # Backward compatibility: use single concurrency value for all services
            default_concurrency = concurrency if concurrency is not None else 100
            self.service_concurrency = {s: default_concurrency for s in self.services}
        else:
            self.service_concurrency = service_concurrency
        
        if service_batch_size is None:
            # Backward compatibility: use single batch_size value for all services
            default_batch_size = batch_size if batch_size is not None else 200
            self.service_batch_size = {s: default_batch_size for s in self.services}
        else:
            self.service_batch_size = service_batch_size
        
        # Set defaults for services not in the dicts
        for service in self.services:
            if service not in self.service_concurrency:
                self.service_concurrency[service] = 100
            if service not in self.service_batch_size:
                self.service_batch_size[service] = 200
        
        # For backward compatibility
        self.concurrency = self.service_concurrency.get(self.services[0], 100)
        self.batch_size = self.service_batch_size.get(self.services[0], 200)
        
        # Compile ignore regex patterns
        self.ignore_patterns = compile_ignore_patterns(ignore_patterns or [])
        if ignore_patterns:
            for pattern in ignore_patterns:
                logger.info(f"Added ignore pattern: {pattern}")
        
        # Initialize translator services
        self.translators = {}  # Dict mapping service name to translator instance
        self._create_translators()
        
        # For backward compatibility
        self.service = self.services[0] if len(self.services) == 1 else ','.join(self.services)
        self.translator = self.translators[self.services[0]] if len(self.services) == 1 else None
        self.google_fallback = None
    
    def _create_translators(self):
        """Create translator services for all specified services with per-service configurations."""
        for service in self.services:
            if service in self.translators:
                continue  # Already created
            
            # Get per-service configuration
            concurrency = self.service_concurrency.get(service, 100)
            batch_size = self.service_batch_size.get(service, 200)
                
            if service == 'google':
                self.translators[service] = GoogleTranslatorService(
                    self.source_lang, self.target_lang, concurrency, self.api_key
                )
            elif service == 'bing':
                self.translators[service] = BingTranslatorService(
                    self.source_lang, self.target_lang, concurrency, self.api_key
                )
            elif service == 'lmstudio':
                self.translators[service] = LMStudioTranslatorService(
                    self.source_lang, self.target_lang, concurrency, self.api_key, items_per_batch=batch_size
                )
            elif service == 'cerebras':
                # Use service-specific API key if provided, otherwise fall back to general api_key
                cerebras_api_key = self.api_key_cerebras or self.api_key
                self.translators[service] = CerebrasTranslatorService(
                    self.source_lang, self.target_lang, concurrency, cerebras_api_key, items_per_batch=batch_size
                )
            elif service == 'nvidia' or service == 'nvidia-openai':
                # Use service-specific API key if provided, otherwise fall back to general api_key
                nvidia_api_key = self.api_key_nvidia or self.api_key
                self.translators[service] = NvidiaOpenAITranslatorService(
                    self.source_lang, self.target_lang, concurrency, nvidia_api_key, items_per_batch=batch_size
                )
            else:
                logger.warning(f"Service '{service}' not explicitly supported. Defaulting to Google.")
                self.translators[service] = GoogleTranslatorService(
                    self.source_lang, self.target_lang, concurrency, self.api_key
                )
        
        logger.info(
            f"Initialized {len(self.translators)} translator service(s): {', '.join(self.translators.keys())}"
        )
    
    def _should_ignore(self, text: str) -> bool:
        """Check if text matches any ignore pattern."""
        return should_ignore(text, self.ignore_patterns)
    
    async def _get_google_fallback(self):
        """Get or create Google translator for fallback."""
        if self.google_fallback is None:
            fallback_concurrency = self.service_concurrency.get('google', 100)
            self.google_fallback = GoogleTranslatorService(
                self.source_lang, self.target_lang, fallback_concurrency, self.api_key
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
            f"using {len(self.services)} service(s): {', '.join(self.services)}..."
        )
        
        try:
            # Initialize all translators
            for service_name, translator in self.translators.items():
                await translator.initialize()
            
            # Process based on number of services
            if len(self.services) > 1:
                # Multiple services: use intelligent task division
                await self._process_mixed_mode(pending_items, cache_file)
            else:
                # Single service: use existing optimized processing
                service = self.services[0]
                translator = self.translators[service]
                
                if service == 'lmstudio':
                    await self._process_lmstudio(pending_items, cache_file, translator)
                elif service == 'google':
                    await self._process_google_optimized(pending_items, cache_file, translator)
                elif service == 'cerebras':
                    await self._process_cerebras(pending_items, cache_file, translator)
                elif service == 'nvidia' or service == 'nvidia-openai':
                    await self._process_nvidia(pending_items, cache_file, translator)
                else:
                    await self._process_individual(pending_items, cache_file, translator)
        finally:
            # Cleanup all translators
            for translator in self.translators.values():
                await translator.cleanup()
            if self.google_fallback:
                await self.google_fallback.cleanup()
        
        elapsed = time.time() - start_time
        items_per_sec = len(pending_items) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Translation completed in {elapsed:.1f}s "
            f"({items_per_sec:.1f} items/sec)"
        )
        
        await finalize_output(data, cache_file, output_file)
    
    def _get_service_characteristics(self, service: str) -> Dict:
        """
        Get characteristics for a service to optimize task division.
        Returns: dict with 'preferred_batch_size', 'concurrency_factor', 'speed_factor'
        """
        characteristics = {
            'google': {
                'preferred_batch_size': 200,  # Large batches work well
                'concurrency_factor': 1.0,    # High concurrency
                'speed_factor': 1.0,          # Fast
                'supports_batch': True,
                'preferred_item_size': 'small'  # Good for many small items
            },
            'cerebras': {
                'preferred_batch_size': 30,   # Smaller batches due to token limits
                'concurrency_factor': 0.2,     # Lower concurrency due to rate limits
                'speed_factor': 0.3,          # Slower but high quality
                'supports_batch': True,
                'preferred_item_size': 'medium'  # Good for medium batches
            },
            'lmstudio': {
                'preferred_batch_size': 50,   # Medium batches
                'concurrency_factor': 0.5,     # Moderate concurrency
                'speed_factor': 0.5,          # Moderate speed
                'supports_batch': True,
                'preferred_item_size': 'medium'  # Good for medium batches
            },
            'bing': {
                'preferred_batch_size': 1,     # Individual items
                'concurrency_factor': 0.8,     # Good concurrency
                'speed_factor': 0.9,          # Fast
                'supports_batch': False,
                'preferred_item_size': 'small'  # Good for many small items
            },
            'nvidia': {
                'preferred_batch_size': 50,   # Medium batches
                'concurrency_factor': 0.6,     # Moderate concurrency
                'speed_factor': 0.7,          # Good speed
                'supports_batch': True,
                'preferred_item_size': 'medium'  # Good for medium batches
            },
            'nvidia-openai': {
                'preferred_batch_size': 50,   # Medium batches
                'concurrency_factor': 0.6,     # Moderate concurrency
                'speed_factor': 0.7,          # Good speed
                'supports_batch': True,
                'preferred_item_size': 'medium'  # Good for medium batches
            }
        }
        return characteristics.get(service, {
            'preferred_batch_size': 100,
            'concurrency_factor': 0.5,
            'speed_factor': 0.5,
            'supports_batch': True,
            'preferred_item_size': 'medium'
        })
    
    def _divide_tasks_intelligently(
        self, 
        pending_items: Dict[str, str],
        services: List[str]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Intelligently divide translation tasks among multiple services.
        
        Algorithm:
        1. Calculate service capacity based on speed and concurrency factors
        2. Assign items proportionally based on capacity
        3. Optimize batch sizes for each service
        4. Balance load to maximize parallel processing
        
        Returns: Dict mapping service name to list of (key, value) tuples
        """
        if not pending_items:
            return {service: [] for service in services}
        
        # Get characteristics for all services
        service_chars = {s: self._get_service_characteristics(s) for s in services}
        
        # Calculate total capacity (weighted by speed and concurrency)
        total_capacity = sum(
            chars['speed_factor'] * chars['concurrency_factor'] 
            for chars in service_chars.values()
        )
        
        if total_capacity == 0:
            # Fallback: equal distribution
            items_per_service = len(pending_items) // len(services)
            remainder = len(pending_items) % len(services)
            
            items_list = list(pending_items.items())
            result = {}
            start_idx = 0
            
            for i, service in enumerate(services):
                end_idx = start_idx + items_per_service + (1 if i < remainder else 0)
                result[service] = items_list[start_idx:end_idx]
                start_idx = end_idx
            
            return result
        
        # Calculate proportional distribution based on capacity
        service_weights = {
            s: service_chars[s]['speed_factor'] * service_chars[s]['concurrency_factor']
            for s in services
        }
        
        # Distribute items proportionally
        items_list = list(pending_items.items())
        total_items = len(items_list)
        
        result = {}
        start_idx = 0
        
        for i, service in enumerate(services):
            if i == len(services) - 1:
                # Last service gets remaining items
                result[service] = items_list[start_idx:]
            else:
                weight = service_weights[service] / total_capacity
                count = max(1, int(total_items * weight))
                end_idx = start_idx + count
                result[service] = items_list[start_idx:end_idx]
                start_idx = end_idx
        
        logger.info(
            f"Task division: {', '.join([f'{s}: {len(result[s])} items' for s in services])}"
        )
        
        return result
    
    async def _process_mixed_mode(self, pending_items: Dict, cache_file: str):
        """
        Process items using multiple services in parallel with intelligent task division.
        Optimizes for maximum speed by dividing work intelligently across services.
        """
        keys = list(pending_items.keys())
        values = [pending_items[k] for k in keys]
        
        # Separate items to translate from ignored items
        to_translate_items: List[Tuple[str, str]] = []
        ignored_items: List[Tuple[str, str]] = []
        
        for key, val in pending_items.items():
            if isinstance(val, str):
                if self._should_ignore(val):
                    ignored_items.append((key, val))
                else:
                    to_translate_items.append((key, val))
            else:
                ignored_items.append((key, val))
        
        logger.info(
            f"Mixed mode: Processing {len(to_translate_items)} items to translate, "
            f"{len(ignored_items)} ignored/non-string items across {len(self.services)} services"
        )
        
        # Write ignored items in batch
        if ignored_items:
            await append_batch_to_cache(cache_file, ignored_items)
        
        if not to_translate_items:
            return
        
        # Intelligently divide tasks among services
        service_tasks = self._divide_tasks_intelligently(
            dict(to_translate_items), 
            self.services
        )
        
        # Initialize cache writer
        cache_writer = BatchCacheWriter(
            cache_file,
            buffer_size=min(500, len(to_translate_items) // 4 + 1),
            flush_interval=3.0
        )
        await cache_writer.start()
        
        try:
            # Shared progress bar
            pbar = tqdm(
                total=len(to_translate_items),
                desc=f"Translating (Mixed: {','.join(self.services)})",
                unit="item"
            )
            
            # Process all services in parallel
            async def process_service_tasks(service: str, tasks: List[Tuple[str, str]], retry_count: int = 0):
                """Process tasks assigned to a specific service. Retries indefinitely until success."""
                if not tasks:
                    return
                
                translator = self.translators[service]
                service_chars = self._get_service_characteristics(service)
                
                # Convert to dict for processing
                task_dict = dict(tasks)
                task_keys = list(task_dict.keys())
                task_values = list(task_dict.values())
                
                try:
                    if service == 'lmstudio':
                        # LMStudio uses batch_dict
                        translated_dict = await translator.translate_batch_dict(task_dict)
                        cache_items = [(k, translated_dict.get(k, v)) for k, v in task_dict.items()]
                        await cache_writer.add_batch(cache_items)
                        pbar.update(len(cache_items))
                        
                    elif service == 'google':
                        # Google: process in optimized batches
                        # Use configured batch_size or fall back to preferred
                        batch_size = self.service_batch_size.get(service, service_chars['preferred_batch_size'])
                        for batch_start in range(0, len(task_values), batch_size):
                            batch_end = min(batch_start + batch_size, len(task_values))
                            batch_values = task_values[batch_start:batch_end]
                            batch_keys = task_keys[batch_start:batch_end]
                            
                            translated_values = await translator.translate_batch(batch_values)
                            cache_items = [(k, v) for k, v in zip(batch_keys, translated_values)]
                            await cache_writer.add_batch(cache_items)
                            pbar.update(len(cache_items))
                            
                    elif service == 'cerebras':
                        # Cerebras: process in batches
                        # Use configured batch_size or fall back to preferred
                        batch_size = self.service_batch_size.get(service, service_chars['preferred_batch_size'])
                        for batch_start in range(0, len(task_values), batch_size):
                            batch_end = min(batch_start + batch_size, len(task_values))
                            batch_values = task_values[batch_start:batch_end]
                            batch_keys = task_keys[batch_start:batch_end]
                            
                            translated_values = await translator.translate_batch(batch_values)
                            cache_items = [(k, v) for k, v in zip(batch_keys, translated_values)]
                            await cache_writer.add_batch(cache_items)
                            pbar.update(len(cache_items))
                            
                    elif service == 'nvidia' or service == 'nvidia-openai':
                        # NVIDIA OpenAI: process in batches
                        # Use configured batch_size or fall back to preferred
                        batch_size = self.service_batch_size.get(service, service_chars['preferred_batch_size'])
                        for batch_start in range(0, len(task_values), batch_size):
                            batch_end = min(batch_start + batch_size, len(task_values))
                            batch_values = task_values[batch_start:batch_end]
                            batch_keys = task_keys[batch_start:batch_end]
                            
                            translated_values = await translator.translate_batch(batch_values)
                            cache_items = [(k, v) for k, v in zip(batch_keys, translated_values)]
                            await cache_writer.add_batch(cache_items)
                            pbar.update(len(cache_items))
                            
                    else:
                        # Other services: process individually
                        service_concurrency = self.service_concurrency.get(service, self.concurrency)
                        semaphore = asyncio.Semaphore(service_concurrency)
                        async def process_item(key: str, value: str):
                            async with semaphore:
                                translated = await translator.translate_single(value)
                                await cache_writer.add(key, translated)
                                pbar.update(1)
                        
                        await asyncio.gather(*[
                            process_item(k, v) for k, v in task_dict.items()
                        ])
                        
                except Exception as e:
                    # Retry indefinitely instead of falling back to original text
                    delay = min(2 * (2 ** retry_count), 120)
                    logger.warning(f"Error processing {service} tasks (attempt {retry_count + 1}): {e}, retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                    await process_service_tasks(service, tasks, retry_count + 1)
            
            # Create tasks for each service
            service_tasks_list = [
                process_service_tasks(service, service_tasks.get(service, []))
                for service in self.services
            ]
            
            # Run all services in parallel
            await asyncio.gather(*service_tasks_list, return_exceptions=True)
            
            pbar.close()
                
        finally:
            await cache_writer.stop()
        
        # Print stats for all services
        for service, translator in self.translators.items():
            if hasattr(translator, 'get_stats'):
                stats = translator.get_stats()
                logger.info(f"{service.capitalize()} stats: {stats}")
    
    async def _process_lmstudio(self, pending_items: Dict, cache_file: str, translator):
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
                    translated_batch = await translator.translate_batch_dict(
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
    
    async def _process_google_optimized(self, pending_items: Dict, cache_file: str, translator):
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
                    translated_values = await translator.translate_batch(batch_values)
                    
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
        if hasattr(translator, 'get_stats'):
            stats = translator.get_stats()
            logger.info(f"Translation stats: {stats}")
    
    async def _process_google_batch(self, pending_items: Dict, cache_file: str, translator):
        """Legacy batch processing (kept for compatibility)."""
        await self._process_google_optimized(pending_items, cache_file, translator)
    
    async def _process_cerebras(self, pending_items: Dict, cache_file: str, translator):
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
                    translated_values = await translator.translate_batch(chunk_values)
                    
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
        if hasattr(translator, 'get_stats'):
            stats = translator.get_stats()
            logger.info(f"Cerebras translation stats: {stats}")
    
    async def _process_nvidia(self, pending_items: Dict, cache_file: str, translator):
        """
        Process items using NVIDIA OpenAI with batch optimization.
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
            f"NVIDIA OpenAI: Processing {len(to_translate_values)} items to translate, "
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
                desc="Translating (NVIDIA OpenAI)",
                unit="item"
            ) as pbar:
                for chunk_start in range(0, len(to_translate_values), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(to_translate_values))
                    chunk_values = to_translate_values[chunk_start:chunk_end]
                    chunk_indices = to_translate_indices[chunk_start:chunk_end]
                    
                    # Translate chunk using batch method (internally handles batching)
                    translated_values = await translator.translate_batch(chunk_values)
                    
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
        if hasattr(translator, 'get_stats'):
            stats = translator.get_stats()
            logger.info(f"NVIDIA OpenAI translation stats: {stats}")
    
    async def _process_individual(self, pending_items: Dict, cache_file: str, translator):
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
                                translated_val = await translator.translate_single(v)
                                await cache_writer.add(k, translated_val)
                        else:
                            await cache_writer.add(k, v)
                        pbar.update(1)
                
                tasks = [process_item(k, v) for k, v in pending_items.items()]
                await asyncio.gather(*tasks)
        finally:
            await cache_writer.stop()
