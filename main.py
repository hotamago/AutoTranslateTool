import argparse
import asyncio
import json
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Set, Dict

import aiofiles
from deep_translator import GoogleTranslator, MicrosoftTranslator
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(self, service: str, source_lang: str, target_lang: str, concurrency: int = 10, api_key: str = None):
        self.service = service.lower()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(concurrency)
        self.executor = ThreadPoolExecutor(max_workers=concurrency)

    def get_translator(self):
        if self.service == 'google':
            return GoogleTranslator(source=self.source_lang, target=self.target_lang)
        elif self.service == 'bing':
            if not self.api_key:
                logger.warning("Bing (Microsoft) translator requires an API key. Please provide one with --api_key.")
            return MicrosoftTranslator(source=self.source_lang, target=self.target_lang, api_key=self.api_key)
        else:
            if self.service not in ['google', 'bing']:
                logger.warning(f"Service '{self.service}' not explicitly supported. Defaulting to Google.")
                return GoogleTranslator(source=self.source_lang, target=self.target_lang)
            return GoogleTranslator(source=self.source_lang, target=self.target_lang)

    async def translate_text(self, text: str) -> str:
        if not text:
            return ""
        
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            try:
                translator = self.get_translator()
                translated = await loop.run_in_executor(
                    self.executor, 
                    translator.translate, 
                    text
                )
                return translated
            except Exception as e:
                logger.error(f"Failed to translate '{text[:20]}...': {e}")
                return text

    async def load_cache(self, cache_file: str) -> Set[str]:
        completed_keys = set()
        if not os.path.exists(cache_file):
            return completed_keys
        
        logger.info(f"Loading cache from {cache_file}...")
        try:
            async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                async for line in f:
                    try:
                        entry = json.loads(line)
                        # Expecting {"key": "original_key", "value": "translated_value"}
                        # Or just {"original_key": "translated_value"}?
                        # The user asked for "mỗi dòng là một cặp key:value".
                        # Let's stick to {"key": "k", "value": "v"} or {"k": "v"}.
                        # {"k": "v"} is simpler but if key has special chars it's fine.
                        # However, if we just dump {"k": "v"}, we need to iterate keys to find which one it is.
                        # Ideally we want to know WHICH keys are done.
                        # Let's assume the line is a dict with 1 key-value pair.
                        if isinstance(entry, dict):
                            completed_keys.update(entry.keys())
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading cache file: {e}")
        
        logger.info(f"Loaded {len(completed_keys)} entries from cache.")
        return completed_keys

    async def append_to_cache(self, cache_file: str, key: str, value: str):
        try:
            entry = {key: value}
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            async with aiofiles.open(cache_file, 'a', encoding='utf-8') as f:
                await f.write(line)
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    async def process_file(self, input_file: str, output_file: str, cache_file: str):
        logger.info(f"Loading input file: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            return

        if not isinstance(data, dict):
            logger.error("Input file must be a JSON object (key-value map).")
            return

        # Load cache
        completed_keys = await self.load_cache(cache_file)
        
        # Identify pending items
        pending_items = {k: v for k, v in data.items() if k not in completed_keys}
        logger.info(f"Total items: {len(data)}. Completed: {len(completed_keys)}. Pending: {len(pending_items)}.")

        if not pending_items:
            logger.info("All items already translated.")
            await self.finalize_output(data, cache_file, output_file)
            return

        logger.info(f"Starting translation from {self.source_lang} to {self.target_lang} using {self.service}...")
        
        with tqdm(total=len(pending_items), desc="Translating", unit="key") as pbar:
            async def process_item(k, v):
                if isinstance(v, str):
                    translated_val = await self.translate_text(v)
                    await self.append_to_cache(cache_file, k, translated_val)
                    pbar.update(1)
                else:
                    # Non-string value, just write as is or skip?
                    # Let's write as is to cache so it's marked done.
                    await self.append_to_cache(cache_file, k, v)
                    pbar.update(1)

            tasks = [process_item(k, v) for k, v in pending_items.items()]
            # Use gather to run concurrently
            await asyncio.gather(*tasks)

        await self.finalize_output(data, cache_file, output_file)

    async def finalize_output(self, original_data: Dict, cache_file: str, output_file: str):
        logger.info("Finalizing output file...")
        # Reconstruct full dictionary from cache
        # We need to respect the order of original_data if possible, or just dump.
        # Python 3.7+ dicts preserve insertion order.
        # We'll read cache into a dict.
        
        translated_map = {}
        if os.path.exists(cache_file):
            try:
                async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                    async for line in f:
                        try:
                            entry = json.loads(line)
                            translated_map.update(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading cache for finalization: {e}")

        # Merge with original data (in case some keys were missing in cache or we want to keep original order)
        # Actually, we want the translated values.
        # Let's create the final dict in original order.
        final_output = {}
        for k in original_data.keys():
            if k in translated_map:
                final_output[k] = translated_map[k]
            else:
                # Should not happen if all done, but if so, keep original?
                # Or maybe it failed.
                final_output[k] = original_data[k]

        logger.info(f"Saving final output to: {output_file}")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            logger.info("Translation completed successfully.")
        except Exception as e:
            logger.error(f"Error writing output file: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Auto Translation Tool")
    parser.add_argument("-s", "--source_lang", required=True, help="Source language code (e.g., 'en')")
    parser.add_argument("-t", "--target_lang", required=True, help="Target language code (e.g., 'vi')")
    parser.add_argument("-i", "--input_file", required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output_file", required=True, help="Path to output JSON file")
    parser.add_argument("-m", "--service", default="google", help="Translation service (google, bing, ...)")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrency level (default: 10)")
    parser.add_argument("-k", "--api_key", help="API key for services that require it (e.g., bing)")
    parser.add_argument("--cache_file", help="Path to cache file (JSONL). Defaults to <output_file>.cache")

    args = parser.parse_args()

    cache_file = args.cache_file
    if not cache_file:
        cache_file = args.output_file + ".cache"

    manager = TranslationManager(
        service=args.service,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        concurrency=args.concurrency,
        api_key=args.api_key
    )

    await manager.process_file(args.input_file, args.output_file, cache_file)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
