"""Bing (Microsoft) Translate service implementation."""

import asyncio
import logging
from typing import Optional

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
        """Translate multiple texts using Bing Translate."""
        if not texts:
            return []
        
        async def translate_single(text: str) -> str:
            """Translate a single text."""
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
                    return translated
                except Exception as e:
                    logger.error(f"Failed to translate '{text[:20]}...': {e}")
                    return text
        
        tasks = [translate_single(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return list(results)

