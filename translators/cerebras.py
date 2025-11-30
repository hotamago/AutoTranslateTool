"""Cerebras LLM translation service implementation with batch and concurrent optimization."""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import BaseTranslator
from utils.batch_manager import BatchManager, BatchItem, TokenAwareBatchBuilder

logger = logging.getLogger(__name__)

# Cerebras rate limits for llama-3.3-70b
RATE_LIMITS = {
    "requests_per_minute": 30,
    "tokens_per_minute": 64_000,
    "requests_per_hour": 900,
    "requests_per_day": 14_400,
}

# Estimated tokens per character (conservative estimate for mixed content)
CHARS_PER_TOKEN = 3.5
# Max tokens to use per request (leaving room for response)
MAX_INPUT_TOKENS_PER_REQUEST = 8_000
# Max items to batch in a single request
MAX_ITEMS_PER_BATCH = 200


@dataclass
class RateLimitTracker:
    """Track rate limits across different time windows."""
    minute_requests: List[float] = field(default_factory=list)
    minute_tokens: List[Tuple[float, int]] = field(default_factory=list)
    hour_requests: List[float] = field(default_factory=list)
    
    def _clean_old_entries(self, current_time: float):
        """Remove entries older than their respective windows."""
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        self.minute_requests = [t for t in self.minute_requests if t > minute_ago]
        self.minute_tokens = [(t, tokens) for t, tokens in self.minute_tokens if t > minute_ago]
        self.hour_requests = [t for t in self.hour_requests if t > hour_ago]
    
    def can_make_request(self, estimated_tokens: int) -> Tuple[bool, float]:
        """Check if we can make a request and return wait time if not."""
        current_time = time.time()
        self._clean_old_entries(current_time)
        
        # Check minute request limit
        if len(self.minute_requests) >= RATE_LIMITS["requests_per_minute"]:
            oldest = min(self.minute_requests)
            wait_time = 60 - (current_time - oldest) + 0.5
            return False, max(0, wait_time)
        
        # Check minute token limit
        current_minute_tokens = sum(tokens for _, tokens in self.minute_tokens)
        if current_minute_tokens + estimated_tokens > RATE_LIMITS["tokens_per_minute"]:
            oldest = min(t for t, _ in self.minute_tokens) if self.minute_tokens else current_time
            wait_time = 60 - (current_time - oldest) + 0.5
            return False, max(0, wait_time)
        
        # Check hour request limit
        if len(self.hour_requests) >= RATE_LIMITS["requests_per_hour"]:
            oldest = min(self.hour_requests)
            wait_time = 3600 - (current_time - oldest) + 1
            return False, max(0, wait_time)
        
        return True, 0
    
    def record_request(self, tokens_used: int):
        """Record a completed request."""
        current_time = time.time()
        self.minute_requests.append(current_time)
        self.minute_tokens.append((current_time, tokens_used))
        self.hour_requests.append(current_time)


@dataclass
class TranslationStats:
    """Track translation statistics."""
    total_requests: int = 0
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    rate_limit_waits: int = 0
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
    
    def record_rate_limit_wait(self):
        self.rate_limit_waits += 1


class CerebrasTranslatorService(BaseTranslator):
    """
    Cerebras LLM translation service with batch and concurrent optimization.
    
    Features:
    - Batch multiple texts in single API request for efficiency
    - Concurrent requests with rate limiting
    - Token-aware batching to respect API limits
    - Automatic retry with exponential backoff
    """
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 5,  # Conservative default due to rate limits
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b",
        max_retries: int = 3,
        items_per_batch: int = 30,  # Items to translate per API call
    ):
        super().__init__(source_lang, target_lang, concurrency, api_key)
        self.model = model
        self.max_retries = max_retries
        self.items_per_batch = min(items_per_batch, MAX_ITEMS_PER_BATCH)
        
        # Use API key from env if not provided
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        
        # Rate limiting
        self._rate_tracker = RateLimitTracker()
        self._rate_lock = asyncio.Lock()
        
        # Stats tracking
        self._stats = TranslationStats()
        
        # Cerebras client (initialized lazily)
        self._client = None
    
    async def initialize(self):
        """Initialize the Cerebras client."""
        if self._client is None:
            try:
                from cerebras.cloud.sdk import Cerebras
                self._client = Cerebras(api_key=self.api_key)
                logger.info(
                    f"Cerebras translator initialized: model={self.model}, "
                    f"concurrency={self.concurrency}, items_per_batch={self.items_per_batch}"
                )
            except ImportError:
                raise ImportError(
                    "cerebras-cloud-sdk is required. Install with: pip install cerebras-cloud-sdk"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Cerebras client: {e}")
    
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
        
        prompt = f"""Translate the following texts from {self.source_lang} to {self.target_lang}.

IMPORTANT RULES:
1. Return ONLY a valid JSON object with the same keys
2. Preserve the exact meaning and tone
3. Keep special characters, punctuation, and formatting
4. Do not add explanations or comments
5. If a text cannot be translated, return it unchanged

Input JSON:
{json.dumps(items_json, ensure_ascii=False, indent=2)}

Output the translated JSON:"""
        return prompt
    
    def _parse_batch_response(self, content: str, original_texts: List[str]) -> Tuple[List[str], List[int]]:
        """Parse the batch translation response.
        
        Returns:
            Tuple of (translations, missing_indices) where translations is a list of
            translated texts (with None for missing ones) and missing_indices is a list
            of indices that were not found in the response.
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
        
        # Extract translations in order
        translations = []
        missing_indices = []
        for i in range(len(original_texts)):
            key = str(i)
            if key in result:
                translations.append(result[key])
            else:
                translations.append(None)
                missing_indices.append(i)
        
        return translations, missing_indices
    
    async def _wait_for_rate_limit(self, estimated_tokens: int):
        """Wait if necessary to respect rate limits."""
        async with self._rate_lock:
            while True:
                can_proceed, wait_time = self._rate_tracker.can_make_request(estimated_tokens)
                if can_proceed:
                    break
                
                self._stats.record_rate_limit_wait()
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
    
    async def _translate_batch_request(
        self, 
        texts: List[str]
    ) -> List[str]:
        """Make a single batch translation request (no retries - handled by batch manager)."""
        if not texts:
            return []
        
        estimated_tokens = self._estimate_batch_tokens(texts)
        
        # Wait for rate limit
        await self._wait_for_rate_limit(estimated_tokens)
        
        start_time = time.time()
        
        prompt = self._create_batch_prompt(texts)
        
        # Make the API call synchronously (Cerebras SDK is sync)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            self.executor,
            self._make_completion_request,
            prompt
        )
        
        if response is None:
            raise RuntimeError("Empty response from Cerebras API")
        
        # Record the request for rate limiting
        input_tokens = response.usage.prompt_tokens if response.usage else estimated_tokens
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        async with self._rate_lock:
            self._rate_tracker.record_request(input_tokens + output_tokens)
        
        # Parse response
        content = response.choices[0].message.content
        translations, missing_indices = self._parse_batch_response(content, texts)
        
        # If there are missing indices, raise exception to trigger requeue
        if missing_indices:
            logger.warning(f"Missing translations for {len(missing_indices)} indices: {missing_indices}")
            # Fill None for missing indices - they will be requeued by batch manager
            for idx in missing_indices:
                translations[idx] = None
        
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
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. You translate text accurately "
                        "while preserving meaning, tone, and formatting. You always respond "
                        "with valid JSON containing the translations."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.model,
            temperature=0.3,
            max_tokens=8192,
        )
    
    def _split_into_batches(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches respecting token limits."""
        batches = []
        current_batch = []
        current_tokens = 200  # Base overhead
        
        for text in texts:
            text_tokens = self._estimate_tokens(text) + 20  # Per-item overhead
            
            # Check if adding this text would exceed limits
            if (len(current_batch) >= self.items_per_batch or 
                current_tokens + text_tokens > MAX_INPUT_TOKENS_PER_REQUEST):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_tokens = 200 + text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
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
        """Translate a dictionary of items."""
        if not items:
            return {}
        
        keys = list(items.keys())
        values = [items[k] for k in keys]
        
        # Filter only string values
        string_indices = [i for i, v in enumerate(values) if isinstance(v, str)]
        string_values = [values[i] for i in string_indices]
        
        if not string_values:
            return items
        
        # Translate strings
        translations = await self.translate_batch(string_values)
        
        # Reassemble result
        result = dict(items)
        for idx, trans in zip(string_indices, translations):
            result[keys[idx]] = trans
        
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
            'rate_limit_waits': self._stats.rate_limit_waits,
            'avg_response_time': f"{self._stats.avg_response_time:.3f}s",
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        await super().cleanup()
        
        if self._stats.total_requests > 0:
            logger.info(
                f"Cerebras stats: {self._stats.successful_items}/{self._stats.total_items} items "
                f"({(self._stats.successful_items / max(1, self._stats.total_items)):.1%} success), "
                f"tokens: {self._stats.total_input_tokens}+{self._stats.total_output_tokens}, "
                f"rate waits: {self._stats.rate_limit_waits}"
            )


