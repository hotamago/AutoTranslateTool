"""LM Studio translation service implementation with batch and concurrent optimization."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .base import BaseTranslator

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


class LMStudioTranslatorService(BaseTranslator):
    """
    LM Studio translation service with batch and concurrent optimization.
    
    Features:
    - Batch multiple texts in single API request for efficiency
    - Concurrent requests with semaphore control
    - Token-aware batching to respect API limits
    - Automatic retry with exponential backoff
    - Reuses HTTP session for better performance
    """
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None,
        api_url: str = "http://127.0.0.1:1234/v1/chat/completions",
        max_retries: int = 3,
        items_per_batch: int = 50,  # Items to translate per API call
    ):
        super().__init__(source_lang, target_lang, concurrency, api_key)
        self.api_url = api_url
        self.max_retries = max_retries
        self.items_per_batch = min(items_per_batch, MAX_ITEMS_PER_BATCH)
        
        # Stats tracking
        self._stats = TranslationStats()
    
    async def initialize(self):
        """Initialize LM Studio translator."""
        await self._init_http_session()
        logger.info(
            f"LM Studio translator initialized: api_url={self.api_url}, "
            f"concurrency={self.concurrency}, items_per_batch={self.items_per_batch}"
        )
    
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
    
    def _create_dict_prompt(self, items: Dict) -> str:
        """Create a prompt for dictionary translation."""
        prompt = f"""Translate the following JSON content from {self.source_lang} to {self.target_lang}.

IMPORTANT RULES:
1. Return ONLY a valid JSON object with the same keys
2. Preserve the exact meaning and tone
3. Keep special characters, punctuation, and formatting
4. Do not add explanations or comments
5. If a text cannot be translated, return it unchanged
6. Maintain the exact structure and keys

Input JSON:
{json.dumps(items, ensure_ascii=False, indent=2)}

Output the translated JSON:"""
        return prompt
    
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
    
    def _parse_dict_response(self, content: str, original_items: Dict) -> Dict:
        """Parse the dictionary translation response."""
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
        if missing_keys:
            raise ValueError(f"Missing translations for keys: {missing_keys}")
        
        return result
    
    async def _translate_batch_request(
        self, 
        texts: List[str], 
        retry_count: int = 0
    ) -> Tuple[List[str], bool]:
        """Make a single batch translation request."""
        if not texts:
            return [], True
        
        start_time = time.time()
        
        try:
            prompt = self._create_batch_prompt(texts)
            estimated_tokens = self._estimate_batch_tokens(texts)
            
            payload = {
                "messages": [
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
                "temperature": 0.3,
                "max_tokens": -1,
                "stream": False
            }
            
            session = await self._init_http_session()
            async with session.post(self.api_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    if not content:
                        raise RuntimeError("Empty response from LM Studio API")
                    
                    # Estimate tokens from usage if available
                    usage = result.get('usage', {})
                    input_tokens = usage.get('prompt_tokens', estimated_tokens)
                    output_tokens = usage.get('completion_tokens', 0)
                    
                    # Parse response
                    translations = self._parse_batch_response(content, texts)
                    
                    elapsed = time.time() - start_time
                    self._stats.record_success(len(texts), input_tokens, output_tokens, elapsed)
                    
                    logger.debug(
                        f"Batch of {len(texts)} translated in {elapsed:.2f}s "
                        f"(tokens: {input_tokens}+{output_tokens})"
                    )
                    
                    return translations, True
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"LM Studio API error {response.status}: {error_text}")
            
        except Exception as e:
            # Retry on errors indefinitely
            wait_time = min((2 ** retry_count) * 2, 120)  # Cap at 2 minutes
            self._stats.record_retry()
            logger.warning(f"Request failed (attempt {retry_count + 1}): {e}, retrying in {wait_time}s")
            await asyncio.sleep(wait_time)
            return await self._translate_batch_request(texts, retry_count + 1)
    
    async def _translate_dict_request(
        self, 
        items: Dict, 
        retry_count: int = 0
    ) -> Tuple[Dict, bool]:
        """Make a single dictionary translation request."""
        if not items:
            return {}, True
        
        start_time = time.time()
        
        try:
            prompt = self._create_dict_prompt(items)
            estimated_tokens = self._estimate_batch_tokens([str(v) for v in items.values() if isinstance(v, str)])
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator. You translate JSON files accurately "
                            "while preserving meaning, tone, formatting, and structure. You always respond "
                            "with valid JSON containing the translations with the same keys."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": -1,
                "stream": False
            }
            
            session = await self._init_http_session()
            async with session.post(self.api_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    if not content:
                        raise RuntimeError("Empty response from LM Studio API")
                    
                    # Estimate tokens from usage if available
                    usage = result.get('usage', {})
                    input_tokens = usage.get('prompt_tokens', estimated_tokens)
                    output_tokens = usage.get('completion_tokens', 0)
                    
                    # Parse response
                    translated_dict = self._parse_dict_response(content, items)
                    
                    elapsed = time.time() - start_time
                    item_count = len([v for v in items.values() if isinstance(v, str)])
                    self._stats.record_success(item_count, input_tokens, output_tokens, elapsed)
                    
                    logger.debug(
                        f"Dict of {len(items)} items translated in {elapsed:.2f}s "
                        f"(tokens: {input_tokens}+{output_tokens})"
                    )
                    
                    return translated_dict, True
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"LM Studio API error {response.status}: {error_text}")
            
        except Exception as e:
            # Retry on errors indefinitely
            wait_time = min((2 ** retry_count) * 2, 120)  # Cap at 2 minutes
            self._stats.record_retry()
            logger.warning(f"Request failed (attempt {retry_count + 1}): {e}, retrying in {wait_time}s")
            await asyncio.sleep(wait_time)
            return await self._translate_dict_request(items, retry_count + 1)
    
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
    
    def _split_dict_into_batches(self, items: Dict) -> List[Dict]:
        """Split dictionary into batches respecting token limits."""
        batches = []
        current_batch = {}
        current_tokens = 200  # Base overhead
        
        for key, value in items.items():
            if not isinstance(value, str):
                # Non-string values go directly to current batch
                current_batch[key] = value
                continue
            
            text_tokens = self._estimate_tokens(value) + 20  # Per-item overhead
            
            # Check if adding this item would exceed limits
            if (len([v for v in current_batch.values() if isinstance(v, str)]) >= self.items_per_batch or 
                current_tokens + text_tokens > MAX_INPUT_TOKENS_PER_REQUEST):
                if current_batch:
                    batches.append(current_batch)
                current_batch = {key: value}
                current_tokens = 200 + text_tokens
            else:
                current_batch[key] = value
                current_tokens += text_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate multiple texts with batch optimization and concurrency.
        
        Splits texts into optimal batches and processes them concurrently.
        """
        if not texts:
            return []
        
        # Split into batches
        batches = self._split_into_batches(texts)
        
        # Create a mapping to reassemble results
        results = [None] * len(texts)
        batch_ranges = []
        
        start_idx = 0
        for batch in batches:
            batch_ranges.append((start_idx, start_idx + len(batch)))
            start_idx += len(batch)
        
        # Process batches with concurrency control
        async def process_batch(batch_idx: int):
            async with self.semaphore:
                batch = batches[batch_idx]
                start, end = batch_ranges[batch_idx]
                
                # Retry until successful
                while True:
                    translations, success = await self._translate_batch_request(batch)
                    if success:
                        for i, translation in enumerate(translations):
                            results[start + i] = translation
                        break
                    # If not successful, retry (shouldn't happen with new retry logic, but keep as safety)
                    logger.warning(f"Batch {batch_idx} failed, retrying...")
                    await asyncio.sleep(1)
        
        # Run all batches concurrently (semaphore limits actual concurrency)
        tasks = [process_batch(i) for i in range(len(batches))]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all results are present (retry any missing ones)
        missing_indices = [i for i, r in enumerate(results) if r is None]
        if missing_indices:
            logger.warning(f"Retrying {len(missing_indices)} missing translations...")
            missing_texts = [texts[i] for i in missing_indices]
            missing_translations = await self.translate_batch(missing_texts)
            for idx, trans in zip(missing_indices, missing_translations):
                results[idx] = trans
        
        return results
    
    async def translate_batch_dict(self, items: Dict) -> Dict:
        """
        Translate a dictionary of items with batch optimization and concurrency.
        
        Splits large dictionaries into optimal batches and processes them concurrently.
        """
        if not items:
            return {}
        
        # Filter only string values for translation
        string_items = {k: v for k, v in items.items() if isinstance(v, str)}
        non_string_items = {k: v for k, v in items.items() if not isinstance(v, str)}
        
        if not string_items:
            return items
        
        # Split into batches
        batches = self._split_dict_into_batches(string_items)
        
        # Process batches with concurrency control
        async def process_batch(batch: Dict):
            async with self.semaphore:
                translated_dict, success = await self._translate_dict_request(batch)
                return translated_dict
        
        # Run all batches concurrently (semaphore limits actual concurrency)
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge all batch results and retry failed batches
        result = dict(non_string_items)
        failed_batches = []
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, dict):
                result.update(batch_result)
            else:
                # On exception, retry the batch
                logger.error(f"Batch translation failed: {batch_result}, will retry")
                failed_batches.append(i)
        
        # Retry failed batches
        while failed_batches:
            logger.info(f"Retrying {len(failed_batches)} failed batches...")
            retry_failed = []
            for batch_idx in failed_batches:
                batch = batches[batch_idx]
                try:
                    translated_dict, success = await self._translate_dict_request(batch)
                    if success:
                        result.update(translated_dict)
                    else:
                        retry_failed.append(batch_idx)
                except Exception as e:
                    logger.error(f"Batch {batch_idx} retry failed: {e}")
                    retry_failed.append(batch_idx)
            failed_batches = retry_failed
        
        # Verify all original keys are present (retry any missing ones)
        missing_keys = [key for key in items.keys() if key not in result]
        if missing_keys:
            logger.warning(f"Retrying {len(missing_keys)} missing keys...")
            missing_items = {k: items[k] for k in missing_keys}
            missing_translated = await self.translate_batch_dict(missing_items)
            result.update(missing_translated)
        
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
                f"LM Studio stats: {self._stats.successful_items}/{self._stats.total_items} items "
                f"({(self._stats.successful_items / max(1, self._stats.total_items)):.1%} success), "
                f"tokens: {self._stats.total_input_tokens}+{self._stats.total_output_tokens}, "
                f"retries: {self._stats.retry_count}"
            )
