"""Bing (Microsoft) Translate service implementation."""

import asyncio
import logging
from typing import Optional, List

from deep_translator import MicrosoftTranslator

from .base import BaseTranslator

logger = logging.getLogger(__name__)


class BingTranslatorService(BaseTranslator):
    """Bing (Microsoft) Translate service."""
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None
    ):
        super().__init__(source_lang, target_lang, concurrency, api_key)
        self.bing_translator: Optional[MicrosoftTranslator] = None
    
    async def initialize(self):
        """Initialize Bing translator."""
        if not self.api_key:
            logger.warning(
                "Bing (Microsoft) translator requires an API key. "
                "Please provide one with --api_key."
            )
        if self.bing_translator is None:
            self.bing_translator = MicrosoftTranslator(
                source=self.source_lang,
                target=self.target_lang,
                api_key=self.api_key
            )
    
    async def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple texts using Bing Translate. Retries indefinitely until success."""
        if not texts:
            return []
        
        async def translate_single(text: str, retry_count: int = 0) -> str:
            """Translate a single text. Retries indefinitely until success."""
            if not text or not text.strip():
                return text
            
            async with self.semaphore:
                try:
                    if self.bing_translator is None:
                        await self.initialize()
                    loop = asyncio.get_running_loop()
                    translated = await loop.run_in_executor(
                        self.executor,
                        self.bing_translator.translate,
                        text
                    )
                    if translated and translated.strip():
                        return translated
                    # Empty translation, retry
                    delay = min(0.5 * (2 ** retry_count), 60)
                    logger.warning(f"Empty translation received (attempt {retry_count + 1}), retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                    return await translate_single(text, retry_count + 1)
                except Exception as e:
                    delay = min(0.5 * (2 ** retry_count), 120)
                    logger.warning(f"Failed to translate '{text[:30]}...' (attempt {retry_count + 1}): {e}, retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                    return await translate_single(text, retry_count + 1)
        
        tasks = [translate_single(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return list(results)

