"""LM Studio translation service implementation."""

import json
import logging
from typing import Dict, List, Optional

import aiohttp

from .base import BaseTranslator

logger = logging.getLogger(__name__)


class LMStudioTranslatorService(BaseTranslator):
    """LM Studio translation service."""
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None,
        api_url: str = "http://127.0.0.1:1234/v1/chat/completions"
    ):
        super().__init__(source_lang, target_lang, concurrency, api_key)
        self.api_url = api_url
    
    async def initialize(self):
        """Initialize LM Studio translator."""
        # No special initialization needed
        pass
    
    async def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts using LM Studio."""
        # LM Studio works with dicts, not lists
        # This is a placeholder - actual implementation uses translate_batch_dict
        raise NotImplementedError(
            "LM Studio uses translate_batch_dict for dict-based translation"
        )
    
    async def translate_batch_dict(self, items: Dict) -> Dict:
        """Translate a dictionary of items using LM Studio."""
        if not items:
            return {}
        
        prompt = f"""
You are a professional translator. Translate the following JSON content from {self.source_lang} to {self.target_lang}.
Maintain the keys and structure exactly. Output ONLY the translated JSON.
JSON to translate:
{json.dumps(items, ensure_ascii=False)}
"""
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that translates JSON files."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": -1,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        # Parse JSON from content
                        try:
                            if "```json" in content:
                                content = content.split("```json")[1].split("```")[0].strip()
                            elif "```" in content:
                                content = content.split("```")[1].split("```")[0].strip()
                            return json.loads(content)
                        except json.JSONDecodeError:
                            # Try to find JSON object in text
                            start = content.find('{')
                            end = content.rfind('}')
                            if start != -1 and end != -1:
                                try:
                                    return json.loads(content[start:end+1])
                                except Exception:
                                    pass
                            logger.error("Failed to parse JSON from LM Studio response")
                            return {}
                    else:
                        logger.error(f"LM Studio error: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"LM Studio request failed: {e}")
            return {}

