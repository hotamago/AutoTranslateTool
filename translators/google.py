"""Google Translate service implementation."""

import asyncio
import logging
from typing import List, Optional

from deep_translator import GoogleTranslator

from .base import BaseTranslator

logger = logging.getLogger(__name__)


class GoogleTranslatorService(BaseTranslator):
    """Google Translate service."""
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None
    ):
        super().__init__(source_lang, target_lang, concurrency, api_key)
        self.google_translator: Optional[GoogleTranslator] = None
    
    async def initialize(self):
        """Initialize Google translator."""
        # No special initialization needed
        pass
    
    async def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple texts using Google Translate API."""
        if not texts:
            return []
        
        session = await self._init_http_session()
        
        async def translate_single(text: str) -> str:
            """Translate a single text using Google Translate API."""
            async with self.semaphore:
                try:
                    # Use Google Translate web API directly
                    url = "https://translate.googleapis.com/translate_a/single"
                    params = {
                        'client': 'gtx',
                        'sl': self.source_lang,
                        'tl': self.target_lang,
                        'dt': 't',
                        'q': text
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            # Parse response - Google returns array of arrays
                            if data and len(data) > 0 and data[0]:
                                translated_text = ''.join([
                                    item[0] if item and len(item) > 0 else ''
                                    for item in data[0] if item
                                ])
                                return translated_text if translated_text else text
                            return text
                        else:
                            # Fallback to deep_translator if API fails
                            logger.warning(
                                f"Google API returned status {response.status} "
                                f"for '{text[:20]}...', using fallback"
                            )
                            return await self._fallback_translate(text)
                except Exception as e:
                    logger.error(
                        f"Translation failed for '{text[:20]}...': {e}, using fallback"
                    )
                    return await self._fallback_translate(text)
        
        # Process all texts concurrently
        tasks = [translate_single(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return list(results)
    
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
            return translated
        except Exception as e:
            logger.error(f"Fallback translation also failed for '{text[:20]}...': {e}")
            return text
    
    async def translate_single(self, text: str) -> str:
        """Translate a single text using Google Translate."""
        if not text:
            return ""
        
        async with self.semaphore:
            return await self._fallback_translate(text)

