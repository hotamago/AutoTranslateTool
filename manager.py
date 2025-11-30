"""Translation manager that orchestrates translation services."""

import asyncio
import logging
from typing import Dict, List, Optional, Set

from tqdm import tqdm

from translators import (
    GoogleTranslatorService,
    BingTranslatorService,
    NLLBTranslatorService,
    OpusMTTranslatorService,
    LMStudioTranslatorService,
)
from utils.cache import load_cache, append_to_cache
from utils.file_handler import load_json_file, save_json_file, finalize_output
from utils.validators import compile_ignore_patterns, should_ignore, has_duplicate_token_error

logger = logging.getLogger(__name__)


class TranslationManager:
    """Manages translation workflow across different services."""
    
    def __init__(
        self,
        service: str,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None,
        batch_size: int = 10,
        ignore_patterns: Optional[List[str]] = None,
        model_url: Optional[str] = None
    ):
        self.service = service.lower()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.api_key = api_key
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.model_url = model_url
        
        # Compile ignore regex patterns
        self.ignore_patterns = compile_ignore_patterns(ignore_patterns or [])
        if ignore_patterns:
            for pattern in ignore_patterns:
                logger.info(f"Added ignore pattern: {pattern}")
        
        # Initialize translator service
        self.translator = self._create_translator()
        self.google_fallback = None  # For fallback translations
    
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
        elif self.service == 'nllb':
            return NLLBTranslatorService(
                self.source_lang, self.target_lang, self.concurrency, self.api_key
            )
        elif self.service == 'opus-mt':
            return OpusMTTranslatorService(
                self.source_lang, self.target_lang, self.concurrency, self.api_key, self.model_url
            )
        elif self.service == 'lmstudio':
            return LMStudioTranslatorService(
                self.source_lang, self.target_lang, self.concurrency, self.api_key
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
            elif self.service in ['nllb', 'opus-mt']:
                await self._process_model_batch(pending_items, cache_file)
            elif self.service == 'google':
                await self._process_google_batch(pending_items, cache_file)
            else:
                await self._process_individual(pending_items, cache_file)
        finally:
            await self.translator.cleanup()
            if self.google_fallback:
                await self.google_fallback.cleanup()
        
        await finalize_output(data, cache_file, output_file)
    
    async def _process_lmstudio(self, pending_items: Dict, cache_file: str):
        """Process items using LM Studio."""
        keys = list(pending_items.keys())
        with tqdm(total=len(keys), desc="Translating (LM Studio Batch)", unit="item") as pbar:
            for i in range(0, len(keys), self.batch_size):
                batch_keys = keys[i:i+self.batch_size]
                batch_items = {k: pending_items[k] for k in batch_keys}
                
                # Separate ignored items from items to translate
                items_to_translate = {}
                ignored_items = {}
                for k, v in batch_items.items():
                    if isinstance(v, str) and self._should_ignore(v):
                        ignored_items[k] = v
                    else:
                        items_to_translate[k] = v
                
                # Translate only non-ignored items
                translated_batch = {}
                if items_to_translate:
                    translated_batch = await self.translator.translate_batch_dict(
                        items_to_translate
                    )
                
                if not translated_batch and items_to_translate:
                    logger.warning(f"Batch {i//self.batch_size} failed or returned empty.")
                else:
                    # Write translated items
                    for k, v in translated_batch.items():
                        await append_to_cache(cache_file, k, v)
                    # Write ignored items (original values)
                    for k, v in ignored_items.items():
                        await append_to_cache(cache_file, k, v)
                
                pbar.update(len(batch_keys))
    
    async def _process_model_batch(
        self, pending_items: Dict, cache_file: str
    ):
        """Process items using model-based translators (NLLB, Opus-MT)."""
        service_name = "NLLB" if self.service == 'nllb' else "opus-mt"
        keys = list(pending_items.keys())
        with tqdm(
            total=len(keys), desc=f"Translating ({service_name} Batch)", unit="item"
        ) as pbar:
            for i in range(0, len(keys), self.batch_size):
                batch_keys = keys[i:i+self.batch_size]
                batch_values = [pending_items[k] for k in batch_keys]
                
                # Filter out non-string values and ignored patterns
                to_translate_indices = []
                to_translate_values = []
                for idx, v in enumerate(batch_values):
                    if isinstance(v, str) and not self._should_ignore(v):
                        to_translate_indices.append(idx)
                        to_translate_values.append(v)
                
                if to_translate_values:
                    translated_values = await self.translator.translate_batch(
                        to_translate_values
                    )
                else:
                    translated_values = []
                
                # Reconstruct batch result and check for duplicate token errors
                trans_idx = 0
                for idx, key in enumerate(batch_keys):
                    val = batch_values[idx]
                    if isinstance(val, str):
                        if self._should_ignore(val):
                            await append_to_cache(cache_file, key, val)
                        elif trans_idx < len(translated_values):
                            translated_val = translated_values[trans_idx]
                            
                            # Check for duplicate token errors
                            if has_duplicate_token_error(translated_val, val):
                                logger.warning(
                                    f"{service_name} duplicate token error detected "
                                    f"for key '{key[:50]}...'. Falling back to Google Translate."
                                )
                                try:
                                    google_fallback = await self._get_google_fallback()
                                    translated_val = await google_fallback.translate_single(val)
                                    logger.info(
                                        f"Google Translate fallback successful "
                                        f"for key '{key[:50]}...'"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Google Translate fallback failed "
                                        f"for key '{key[:50]}...': {e}"
                                    )
                                    translated_val = val
                            
                            await append_to_cache(cache_file, key, translated_val)
                            trans_idx += 1
                        else:
                            logger.error(f"Mismatch in translation count for key {key}")
                    else:
                        await append_to_cache(cache_file, key, val)
                
                pbar.update(len(batch_keys))
    
    async def _process_google_batch(self, pending_items: Dict, cache_file: str):
        """Process items using Google Translate batch processing."""
        keys = list(pending_items.keys())
        with tqdm(
            total=len(keys), desc="Translating (Google Batch)", unit="item"
        ) as pbar:
            for i in range(0, len(keys), self.batch_size):
                batch_keys = keys[i:i+self.batch_size]
                batch_values = [pending_items[k] for k in batch_keys]
                
                # Filter out non-string values and ignored patterns
                to_translate_indices = []
                to_translate_values = []
                for idx, v in enumerate(batch_values):
                    if isinstance(v, str) and not self._should_ignore(v):
                        to_translate_indices.append(idx)
                        to_translate_values.append(v)
                
                if to_translate_values:
                    translated_values = await self.translator.translate_batch(
                        to_translate_values
                    )
                else:
                    translated_values = []
                
                # Reconstruct batch result
                trans_idx = 0
                for idx, key in enumerate(batch_keys):
                    val = batch_values[idx]
                    if isinstance(val, str):
                        if self._should_ignore(val):
                            await append_to_cache(cache_file, key, val)
                        elif trans_idx < len(translated_values):
                            translated_val = translated_values[trans_idx]
                            await append_to_cache(cache_file, key, translated_val)
                            trans_idx += 1
                        else:
                            logger.error(f"Mismatch in translation count for key {key}")
                    else:
                        await append_to_cache(cache_file, key, val)
                
                pbar.update(len(batch_keys))
    
    async def _process_individual(self, pending_items: Dict, cache_file: str):
        """Process items individually (for services like Bing)."""
        with tqdm(total=len(pending_items), desc="Translating", unit="key") as pbar:
            async def process_item(k, v):
                if isinstance(v, str):
                    if self._should_ignore(v):
                        await append_to_cache(cache_file, k, v)
                    else:
                        translated_val = await self.translator.translate_single(v)
                        await append_to_cache(cache_file, k, translated_val)
                    pbar.update(1)
                else:
                    await append_to_cache(cache_file, k, v)
                    pbar.update(1)
            
            tasks = [process_item(k, v) for k, v in pending_items.items()]
            await asyncio.gather(*tasks)

