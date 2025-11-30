"""Google Translate service implementation with high-performance optimizations."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from aiohttp import TCPConnector, ClientTimeout, ClientSession

from deep_translator import GoogleTranslator

from .base import BaseTranslator

logger = logging.getLogger(__name__)

# Multiple Google Translate endpoints for load balancing
GOOGLE_TRANSLATE_ENDPOINTS = [
    "https://translate.googleapis.com/translate_a/single",
    "https://translate.google.com/translate_a/single",
    "https://translate.google.com.vn/translate_a/single",
    "https://translate.google.co.uk/translate_a/single",
]

# User agents rotation for avoiding detection
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


@dataclass
class TranslationStats:
    """Track translation statistics for adaptive optimization."""
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    rate_limited: int = 0
    avg_response_time: float = 0.0
    
    def record_success(self, response_time: float):
        self.total_requests += 1
        self.successful += 1
        # Exponential moving average
        self.avg_response_time = 0.9 * self.avg_response_time + 0.1 * response_time
    
    def record_failure(self):
        self.total_requests += 1
        self.failed += 1
    
    def record_rate_limit(self):
        self.total_requests += 1
        self.rate_limited += 1
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful / self.total_requests


class GoogleTranslatorService(BaseTranslator):
    """High-performance Google Translate service with advanced optimizations."""
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 100,  # Higher default concurrency
        api_key: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 0.3,
        adaptive_throttle: bool = True,
    ):
        super().__init__(source_lang, target_lang, concurrency, api_key)
        self.google_translator: Optional[GoogleTranslator] = None
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.adaptive_throttle = adaptive_throttle
        self._endpoint_index = 0
        self._user_agent_index = 0
        self._lock = asyncio.Lock()
        self._stats = TranslationStats()
        self._sessions: List[ClientSession] = []
        self._session_pool_size = min(4, max(1, concurrency // 25))
        self._current_session_idx = 0
        
        # Adaptive throttling
        self._current_delay = 0.0
        self._min_delay = 0.0
        self._max_delay = 2.0
    
    async def _create_optimized_session(self) -> ClientSession:
        """Create an optimized HTTP session for high throughput."""
        connector = TCPConnector(
            limit=self.concurrency,
            limit_per_host=self.concurrency // 2,
            ttl_dns_cache=600,  # Cache DNS for 10 minutes
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
        )
        
        timeout = ClientTimeout(
            total=45,
            connect=8,
            sock_read=25,
            sock_connect=8,
        )
        
        return ClientSession(
            connector=connector,
            timeout=timeout,
            raise_for_status=False,
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with rotating user agent."""
        self._user_agent_index = (self._user_agent_index + 1) % len(USER_AGENTS)
        return {
            'User-Agent': USER_AGENTS[self._user_agent_index],
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
        }
    
    async def initialize(self):
        """Initialize connection pool with multiple sessions."""
        if not self._sessions:
            # Create multiple sessions for better throughput
            for _ in range(self._session_pool_size):
                session = await self._create_optimized_session()
                self._sessions.append(session)
            self.http_session = self._sessions[0]
            
            # Warm up connections by making test requests
            await self._warmup_connections()
            
        logger.info(
            f"Google Translator initialized: concurrency={self.concurrency}, "
            f"sessions={self._session_pool_size}"
        )
    
    async def _warmup_connections(self):
        """Warm up connection pool with test requests."""
        try:
            warmup_tasks = []
            for session in self._sessions[:2]:
                warmup_tasks.append(
                    self._make_test_request(session)
                )
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
            logger.debug("Connection pool warmed up")
        except Exception as e:
            logger.debug(f"Warmup failed (non-critical): {e}")
    
    async def _make_test_request(self, session: ClientSession):
        """Make a test request to warm up connection."""
        url = GOOGLE_TRANSLATE_ENDPOINTS[0]
        params = {
            'client': 'gtx',
            'sl': self.source_lang,
            'tl': self.target_lang,
            'dt': 't',
            'q': 'test'
        }
        async with session.get(url, params=params, headers=self._get_headers()) as resp:
            await resp.read()
    
    def _get_session(self) -> ClientSession:
        """Round-robin session selection."""
        self._current_session_idx = (self._current_session_idx + 1) % len(self._sessions)
        return self._sessions[self._current_session_idx]
    
    def _get_endpoint(self) -> str:
        """Round-robin endpoint selection with random jitter."""
        self._endpoint_index = (self._endpoint_index + 1) % len(GOOGLE_TRANSLATE_ENDPOINTS)
        return GOOGLE_TRANSLATE_ENDPOINTS[self._endpoint_index]
    
    def _adjust_throttle(self):
        """Adaptively adjust request delay based on success rate."""
        if not self.adaptive_throttle:
            return
        
        if self._stats.total_requests < 10:
            return
        
        success_rate = self._stats.success_rate
        
        if success_rate < 0.8:
            # Increase delay if too many failures
            self._current_delay = min(self._max_delay, self._current_delay + 0.1)
        elif success_rate > 0.95 and self._current_delay > self._min_delay:
            # Decrease delay if mostly successful
            self._current_delay = max(self._min_delay, self._current_delay - 0.05)
    
    async def _translate_with_retry(
        self, 
        session: ClientSession, 
        text: str,
        retry_count: int = 0
    ) -> Tuple[str, bool]:
        """Translate with exponential backoff retry."""
        if not text or not text.strip():
            return text, True
        
        # Adaptive throttling delay
        if self._current_delay > 0:
            await asyncio.sleep(self._current_delay + random.uniform(0, 0.1))
        
        start_time = time.monotonic()
        
        try:
            url = self._get_endpoint()
            params = {
                'client': 'gtx',
                'sl': self.source_lang,
                'tl': self.target_lang,
                'dt': 't',
                'q': text
            }
            
            async with session.get(url, params=params, headers=self._get_headers()) as response:
                if response.status == 200:
                    data = await response.json(content_type=None)
                    elapsed = time.monotonic() - start_time
                    self._stats.record_success(elapsed)
                    self._adjust_throttle()
                    
                    if data and len(data) > 0 and data[0]:
                        translated_text = ''.join([
                            item[0] if item and len(item) > 0 else ''
                            for item in data[0] if item
                        ])
                        return translated_text if translated_text else text, True
                    return text, True
                
                elif response.status == 429:
                    self._stats.record_rate_limit()
                    self._adjust_throttle()
                    
                    if retry_count < self.max_retries:
                        delay = self.base_delay * (2 ** retry_count) + random.uniform(0.5, 1.5)
                        logger.debug(f"Rate limited, waiting {delay:.2f}s (retry {retry_count + 1})")
                        await asyncio.sleep(delay)
                        # Try different endpoint on retry
                        return await self._translate_with_retry(session, text, retry_count + 1)
                    
                    logger.warning(f"Rate limit exceeded after {self.max_retries} retries")
                    return text, False
                
                elif response.status >= 500:
                    self._stats.record_failure()
                    if retry_count < self.max_retries:
                        delay = self.base_delay * (2 ** retry_count)
                        await asyncio.sleep(delay)
                        return await self._translate_with_retry(session, text, retry_count + 1)
                    return text, False
                
                else:
                    self._stats.record_failure()
                    logger.warning(f"Google API returned status {response.status}")
                    return text, False
                    
        except asyncio.TimeoutError:
            self._stats.record_failure()
            if retry_count < self.max_retries:
                delay = self.base_delay * (2 ** retry_count)
                await asyncio.sleep(delay)
                return await self._translate_with_retry(session, text, retry_count + 1)
            logger.warning(f"Timeout after {self.max_retries} retries for '{text[:30]}...'")
            return text, False
            
        except Exception as e:
            self._stats.record_failure()
            if retry_count < self.max_retries:
                delay = self.base_delay * (2 ** retry_count)
                await asyncio.sleep(delay)
                return await self._translate_with_retry(session, text, retry_count + 1)
            logger.error(f"Translation error for '{text[:30]}...': {e}")
            return text, False
    
    async def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate multiple texts with high concurrency.
        Uses semaphore for rate limiting and session pooling.
        """
        if not texts:
            return []
        
        if not self._sessions:
            await self.initialize()
        
        results = [None] * len(texts)
        failed_indices = []
        
        async def translate_single(index: int, text: str):
            """Translate with session rotation."""
            async with self.semaphore:
                session = self._get_session()
                translated, success = await self._translate_with_retry(session, text)
                results[index] = translated
                if not success:
                    failed_indices.append(index)
        
        # Process all texts concurrently
        tasks = [translate_single(i, text) for i, text in enumerate(texts)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle failed translations with fallback (in batches)
        if failed_indices:
            logger.info(f"Using fallback for {len(failed_indices)} failed translations")
            fallback_semaphore = asyncio.Semaphore(10)  # Limit fallback concurrency
            
            async def fallback_translate(idx: int):
                async with fallback_semaphore:
                    results[idx] = await self._fallback_translate(texts[idx])
            
            fallback_tasks = [fallback_translate(idx) for idx in failed_indices]
            await asyncio.gather(*fallback_tasks, return_exceptions=True)
        
        return [r if r is not None else texts[i] for i, r in enumerate(results)]
    
    async def _fallback_translate(self, text: str) -> str:
        """Fallback translation using deep_translator."""
        try:
            if self.google_translator is None:
                self.google_translator = GoogleTranslator(
                    source=self.source_lang,
                    target=self.target_lang
                )
            loop = asyncio.get_running_loop()
            translated = await loop.run_in_executor(
                self.executor,
                self.google_translator.translate,
                text
            )
            return translated if translated else text
        except Exception as e:
            logger.error(f"Fallback translation failed for '{text[:30]}...': {e}")
            return text
    
    async def translate_single(self, text: str) -> str:
        """Translate a single text."""
        if not text or not text.strip():
            return text
        
        if not self._sessions:
            await self.initialize()
        
        async with self.semaphore:
            session = self._get_session()
            translated, success = await self._translate_with_retry(session, text)
            if not success:
                return await self._fallback_translate(text)
            return translated
    
    async def translate_batch_parallel(
        self, 
        texts: List[str],
        chunk_size: int = 500
    ) -> List[str]:
        """
        Ultra-fast parallel translation for very large batches.
        Splits work across multiple chunks processed simultaneously.
        """
        if not texts:
            return []
        
        if len(texts) <= chunk_size:
            return await self.translate_batch(texts)
        
        if not self._sessions:
            await self.initialize()
        
        # Split into chunks
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        logger.info(f"Processing {len(texts)} texts in {len(chunks)} parallel chunks")
        
        # Process all chunks in parallel
        chunk_results = await asyncio.gather(
            *[self.translate_batch(chunk) for chunk in chunks],
            return_exceptions=True
        )
        
        # Flatten and handle errors
        results = []
        for i, chunk_result in enumerate(chunk_results):
            if isinstance(chunk_result, Exception):
                logger.error(f"Chunk {i} failed: {chunk_result}")
                results.extend(chunks[i])  # Return original texts on failure
            else:
                results.extend(chunk_result)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get translation statistics."""
        return {
            'total_requests': self._stats.total_requests,
            'successful': self._stats.successful,
            'failed': self._stats.failed,
            'rate_limited': self._stats.rate_limited,
            'success_rate': f"{self._stats.success_rate:.2%}",
            'avg_response_time': f"{self._stats.avg_response_time:.3f}s",
            'current_throttle_delay': f"{self._current_delay:.3f}s",
        }
    
    async def cleanup(self):
        """Cleanup all sessions and resources."""
        for session in self._sessions:
            if session and not session.closed:
                await session.close()
        self._sessions.clear()
        self.http_session = None
        
        if self.executor:
            self.executor.shutdown(wait=False)
        
        # Log final stats
        if self._stats.total_requests > 0:
            logger.info(
                f"Google Translate stats: {self._stats.successful}/{self._stats.total_requests} "
                f"({self._stats.success_rate:.1%} success), "
                f"{self._stats.rate_limited} rate limited, "
                f"avg response: {self._stats.avg_response_time:.3f}s"
            )
