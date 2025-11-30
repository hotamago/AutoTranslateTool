"""NVIDIA OpenAI translation service implementation with batch and concurrent optimization."""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import BaseTranslator
from utils.batch_manager import BatchManager, BatchItem, TokenAwareBatchBuilder

logger = logging.getLogger(__name__)

# Estimated tokens per character (conservative estimate for mixed content)
CHARS_PER_TOKEN = 3.5
# Max tokens to use per request (leaving room for response)
MAX_INPUT_TOKENS_PER_REQUEST = 8_000
# Max items to batch in a single request
MAX_ITEMS_PER_BATCH = 200


@dataclass
class TranslationStats:
    """Track translation statistics."""
    total_requests: int = 0
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    retry_count: int = 0
    avg_response_time: float = 0.0
    
    def record_success(self, items_count: int, input_tokens: int, output_tokens: int, response_time: float):
        self.total_requests += 1
        self.total_items += items_count
        self.successful_items += items_count
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        # Exponential moving average
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = 0.9 * self.avg_response_time + 0.1 * response_time
    
    def record_failure(self, items_count: int):
        self.total_requests += 1
        self.total_items += items_count
        self.failed_items += items_count
    
    def record_retry(self):
        self.retry_count += 1


class NvidiaOpenAITranslatorService(BaseTranslator):
    """
    NVIDIA OpenAI translation service with batch and concurrent optimization.
    
    Features:
    - Batch multiple texts in single API request for efficiency
    - Concurrent requests with semaphore control
    - Token-aware batching to respect API limits
    - Automatic retry with exponential backoff
    - Uses OpenAI SDK with NVIDIA's base URL
    """
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "moonshotai/kimi-k2-instruct-0905",
        max_retries: int = 3,
        items_per_batch: int = 50,  # Items to translate per API call
    ):
        super().__init__(source_lang, target_lang, concurrency, api_key)
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.items_per_batch = min(items_per_batch, MAX_ITEMS_PER_BATCH)
        
        # Use API key from env if not provided
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be provided either as parameter or environment variable")
        
        # Stats tracking
        self._stats = TranslationStats()
        
        # OpenAI client (initialized lazily)
        self._client = None
        
        # Load prompts from files
        prompts_dir = Path(__file__).parent.parent / "prompts"
        self._batch_prompt_template = (prompts_dir / "nvidia_batch_prompt.txt").read_text(encoding="utf-8")
        self._dict_prompt_template = (prompts_dir / "nvidia_dict_prompt.txt").read_text(encoding="utf-8")
        self._system_batch = (prompts_dir / "nvidia_system_batch.txt").read_text(encoding="utf-8").strip()
        self._system_dict = (prompts_dir / "nvidia_system_dict.txt").read_text(encoding="utf-8").strip()
    
    async def initialize(self):
        """Initialize NVIDIA OpenAI translator."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
                logger.info(
                    f"NVIDIA OpenAI translator initialized: model={self.model}, "
                    f"base_url={self.base_url}, concurrency={self.concurrency}, "
                    f"items_per_batch={self.items_per_batch}"
                )
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize NVIDIA OpenAI client: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text."""
        return max(1, int(len(text) / CHARS_PER_TOKEN))
    
    def _estimate_batch_tokens(self, texts: List[str]) -> int:
        """Estimate total tokens for a batch including prompt overhead."""
        content_tokens = sum(self._estimate_tokens(t) for t in texts)
        # Add overhead for prompt structure, JSON formatting, etc.
        overhead = 200 + len(texts) * 20
        return content_tokens + overhead
    
    def _create_batch_prompt(self, texts: List[str]) -> str:
        """Create a prompt for batch translation."""
        # Use indexed format for reliable parsing
        items_json = {str(i): text for i, text in enumerate(texts)}
        
        return self._batch_prompt_template.format(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            items_json=json.dumps(items_json, ensure_ascii=False, indent=2)
        )
    
    def _create_dict_prompt(self, items: Dict) -> str:
        """Create a prompt for dictionary translation."""
        return self._dict_prompt_template.format(
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            items_json=json.dumps(items, ensure_ascii=False, indent=2)
        )
    
    def _parse_batch_response(self, content: str, original_texts: List[str]) -> List[str]:
        """Parse the batch translation response."""
        # Try to extract JSON from the response
        content = content.strip()
        
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
        
        # Find JSON object boundaries
        start = content.find('{')
        end = content.rfind('}')
        
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No valid JSON object found in response")
        
        json_str = content[start:end + 1]
        result = json.loads(json_str)
        
        # Extract translations in order
        translations = []
        missing_indices = []
        for i in range(len(original_texts)):
            key = str(i)
            if key in result:
                translations.append(result[key])
            else:
                missing_indices.append(i)
        
        if missing_indices:
            raise ValueError(f"Missing translations for indices: {missing_indices}")
        
        return translations
    
    def _parse_dict_response(self, content: str, original_items: Dict) -> Tuple[Dict, List[str]]:
        """Parse the dictionary translation response.
        
        Returns:
            Tuple of (translated_dict, missing_keys) where missing_keys is a list of keys
            that were not found in the response.
        """
        # Try to extract JSON from the response
        content = content.strip()
        
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
        
        # Find JSON object boundaries
        start = content.find('{')
        end = content.rfind('}')
        
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No valid JSON object found in response")
        
        json_str = content[start:end + 1]
        result = json.loads(json_str)
        
        # Check for missing keys
        missing_keys = [key for key in original_items.keys() if key not in result]
        
        return result, missing_keys
    
    async def _translate_batch_request(
        self, 
        texts: List[str]
    ) -> List[str]:
        """Make a single batch translation request (no retries - handled by batch manager)."""
        if not texts:
            return []
        
        if self._client is None:
            await self.initialize()
        
        start_time = time.time()
        
        prompt = self._create_batch_prompt(texts)
        estimated_tokens = self._estimate_batch_tokens(texts)
        
        # Make the API call synchronously (OpenAI SDK is sync)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            self.executor,
            self._make_completion_request,
            prompt
        )
        
        if response is None:
            raise RuntimeError("Empty response from NVIDIA OpenAI API")
        
        # Extract content
        content = response.choices[0].message.content
        
        if not content:
            raise RuntimeError("Empty response content from NVIDIA OpenAI API")
        
        # Estimate tokens from usage if available
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else estimated_tokens
        output_tokens = usage.completion_tokens if usage else 0
        
        # Parse response
        translations = self._parse_batch_response(content, texts)
        
        elapsed = time.time() - start_time
        self._stats.record_success(len(texts), input_tokens, output_tokens, elapsed)
        
        logger.debug(
            f"Batch of {len(texts)} translated in {elapsed:.2f}s "
            f"(tokens: {input_tokens}+{output_tokens})"
        )
        
        return translations
    
    def _make_completion_request(self, prompt: str):
        """Make a synchronous completion request (called from executor)."""
        return self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._system_batch
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.6,
            top_p=0.9,
            max_tokens=4096,
            stream=False
        )
    
    async def _translate_dict_request(
        self, 
        items: Dict
    ) -> Dict:
        """Make a single dictionary translation request (no retries - handled by batch manager)."""
        if not items:
            return {}
        
        if self._client is None:
            await self.initialize()
        
        start_time = time.time()
        
        prompt = self._create_dict_prompt(items)
        estimated_tokens = self._estimate_batch_tokens([str(v) for v in items.values() if isinstance(v, str)])
        
        # Make the API call synchronously (OpenAI SDK is sync)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            self.executor,
            self._make_dict_completion_request,
            prompt
        )
        
        if response is None:
            raise RuntimeError("Empty response from NVIDIA OpenAI API")
        
        # Extract content
        content = response.choices[0].message.content
        
        if not content:
            raise RuntimeError("Empty response content from NVIDIA OpenAI API")
        
        # Estimate tokens from usage if available
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else estimated_tokens
        output_tokens = usage.completion_tokens if usage else 0
        
        # Parse response
        translated_dict, missing_keys = self._parse_dict_response(content, items)
        
        # If there are missing keys, raise exception to trigger requeue of entire batch
        if missing_keys:
            raise ValueError(f"Missing translations for {len(missing_keys)} keys: {missing_keys}")
        
        elapsed = time.time() - start_time
        item_count = len([v for v in items.values() if isinstance(v, str)])
        self._stats.record_success(item_count, input_tokens, output_tokens, elapsed)
        
        logger.debug(
            f"Dict of {len(items)} items translated in {elapsed:.2f}s "
            f"(tokens: {input_tokens}+{output_tokens})"
        )
        
        return translated_dict
    
    def _make_dict_completion_request(self, prompt: str):
        """Make a synchronous completion request for dictionary translation (called from executor)."""
        return self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._system_dict
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.6,
            top_p=0.9,
            max_tokens=4096,
            stream=False
        )
    
    async def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate multiple texts with batch optimization and concurrency.
        Uses queue-based batch manager for optimized retry handling.
        """
        if not texts:
            return []
        
        if self._client is None:
            await self.initialize()
        
        # Create token-aware batch builder
        batch_builder = TokenAwareBatchBuilder(
            max_tokens=MAX_INPUT_TOKENS_PER_REQUEST,
            estimate_tokens=self._estimate_tokens,
            max_items=self.items_per_batch,
        )
        
        # Create batch manager
        batch_manager = BatchManager(
            batch_size=self.items_per_batch,
            max_retries=0,  # Infinite retries
            batch_builder=batch_builder,
        )
        
        # Add items to queue (using index as key for list results)
        items = [(str(i), text) for i, text in enumerate(texts)]
        await batch_manager.add_items(items)
        
        # Process batches
        async def process_batch(batch: List[BatchItem]):
            """Process a single batch of items."""
            async with self.semaphore:
                # Extract texts from batch items
                batch_texts = [item.value for item in batch]
                # Translate
                translations = await self._translate_batch_request(batch_texts)
                return translations
        
        # Process with concurrency control
        results_dict = await batch_manager.process(
            processor=process_batch,
            concurrency=self.concurrency,
        )
        
        # Convert dict results back to list (preserving order)
        results = [None] * len(texts)
        for idx_str, translation in results_dict.items():
            idx = int(idx_str)
            if 0 <= idx < len(texts):
                results[idx] = translation
        
        # Verify all results are present
        missing_indices = [i for i, r in enumerate(results) if r is None]
        if missing_indices:
            logger.warning(f"Missing translations for {len(missing_indices)} indices, requeuing...")
            # Requeue missing items
            for idx in missing_indices:
                await batch_manager.add_item(str(idx), texts[idx])
            # Process again
            additional_results = await batch_manager.process(
                processor=process_batch,
                concurrency=self.concurrency,
            )
            for idx_str, translation in additional_results.items():
                idx = int(idx_str)
                if 0 <= idx < len(texts):
                    results[idx] = translation
        
        return results
    
    async def translate_batch_dict(self, items: Dict) -> Dict:
        """
        Translate a dictionary of items with batch optimization and concurrency.
        Uses queue-based batch manager for optimized retry handling.
        """
        if not items:
            return {}
        
        if self._client is None:
            await self.initialize()
        
        # Filter only string values for translation
        string_items = {k: v for k, v in items.items() if isinstance(v, str)}
        non_string_items = {k: v for k, v in items.items() if not isinstance(v, str)}
        
        if not string_items:
            return items
        
        # Create token-aware batch builder
        batch_builder = TokenAwareBatchBuilder(
            max_tokens=MAX_INPUT_TOKENS_PER_REQUEST,
            estimate_tokens=self._estimate_tokens,
            max_items=self.items_per_batch,
        )
        
        # Create batch manager
        batch_manager = BatchManager(
            batch_size=self.items_per_batch,
            max_retries=0,  # Infinite retries
            batch_builder=batch_builder,
        )
        
        # Add items to queue
        await batch_manager.add_items(list(string_items.items()))
        
        # Process batches
        async def process_batch(batch: List[BatchItem]):
            """Process a single batch of items."""
            async with self.semaphore:
                # Convert batch items to dict
                batch_dict = {item.key: item.value for item in batch}
                # Translate
                translated_dict = await self._translate_dict_request(batch_dict)
                return translated_dict
        
        # Process with concurrency control
        result = await batch_manager.process(
            processor=process_batch,
            concurrency=self.concurrency,
        )
        
        # Add non-string items back
        result.update(non_string_items)
        
        # Verify all original keys are present
        missing_keys = [key for key in string_items.keys() if key not in result]
        if missing_keys:
            logger.warning(f"Missing translations for {len(missing_keys)} keys, requeuing...")
            # Requeue missing items
            missing_items = [(k, string_items[k]) for k in missing_keys]
            await batch_manager.add_items(missing_items)
            # Process again
            additional_results = await batch_manager.process(
                processor=process_batch,
                concurrency=self.concurrency,
            )
            result.update(additional_results)
        
        return result
    
    def get_stats(self) -> Dict:
        """Get translation statistics."""
        return {
            'total_requests': self._stats.total_requests,
            'total_items': self._stats.total_items,
            'successful_items': self._stats.successful_items,
            'failed_items': self._stats.failed_items,
            'success_rate': f"{(self._stats.successful_items / max(1, self._stats.total_items)):.2%}",
            'total_input_tokens': self._stats.total_input_tokens,
            'total_output_tokens': self._stats.total_output_tokens,
            'retry_count': self._stats.retry_count,
            'avg_response_time': f"{self._stats.avg_response_time:.3f}s",
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        await super().cleanup()
        
        if self._stats.total_requests > 0:
            logger.info(
                f"NVIDIA OpenAI stats: {self._stats.successful_items}/{self._stats.total_items} items "
                f"({(self._stats.successful_items / max(1, self._stats.total_items)):.1%} success), "
                f"tokens: {self._stats.total_input_tokens}+{self._stats.total_output_tokens}, "
                f"retries: {self._stats.retry_count}"
            )

