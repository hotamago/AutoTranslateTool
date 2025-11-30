"""Base translator class."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import aiohttp


class BaseTranslator(ABC):
    """Base class for all translation services."""
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.api_key = api_key
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.executor = ThreadPoolExecutor(max_workers=concurrency)
        self.http_session: Optional[aiohttp.ClientSession] = None
    
    async def _init_http_session(self):
        """Initialize HTTP session for async requests."""
        if self.http_session is None:
            connector = aiohttp.TCPConnector(
                limit=self.concurrency,
                limit_per_host=self.concurrency
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self.http_session
    
    async def _close_http_session(self):
        """Close HTTP session."""
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
    
    @abstractmethod
    async def initialize(self):
        """Initialize the translator (load models, etc.)."""
        pass
    
    @abstractmethod
    async def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts."""
        pass
    
    async def translate_single(self, text: str) -> str:
        """Translate a single text (default implementation uses batch)."""
        if not text:
            return ""
        results = await self.translate_batch([text])
        return results[0] if results else text
    
    async def cleanup(self):
        """Cleanup resources."""
        await self._close_http_session()

