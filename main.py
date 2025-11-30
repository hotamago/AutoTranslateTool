"""Main entry point for Auto Translation Tool."""

import argparse
import asyncio
import logging
import sys

from manager import TranslationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Auto Translation Tool")
    parser.add_argument(
        "-s", "--source_lang", required=True,
        help="Source language code (e.g., 'en')"
    )
    parser.add_argument(
        "-t", "--target_lang", required=True,
        help="Target language code (e.g., 'vi')"
    )
    parser.add_argument(
        "-i", "--input_file", required=True,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "-o", "--output_file", required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "-m", "--service", default="google",
        help="Translation service (google, bing, nllb, opus-mt, lmstudio)"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=10,
        help="Concurrency level (default: 10)"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=10,
        help="Batch size for NLLB/opus-mt/LM Studio (default: 10)"
    )
    parser.add_argument(
        "-k", "--api_key",
        help="API key for services that require it (e.g., bing)"
    )
    parser.add_argument(
        "--model_url",
        help="Model URL/identifier for opus-mt service (e.g., Helsinki-NLP/opus-mt-ja-vi)"
    )
    parser.add_argument(
        "--cache_file",
        help="Path to cache file (JSONL). Defaults to <output_file>.cache"
    )
    parser.add_argument(
        "--ignore_regex", action="append",
        help="Regex pattern to ignore during translation (can be specified multiple times). "
             "Example: '^[0-9]+$' to ignore pure numbers"
    )
    
    args = parser.parse_args()
    
    cache_file = args.cache_file
    if not cache_file:
        cache_file = args.output_file + ".cache"
    
    manager = TranslationManager(
        service=args.service,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        concurrency=args.concurrency,
        api_key=args.api_key,
        batch_size=args.batch_size,
        ignore_patterns=args.ignore_regex,
        model_url=args.model_url
    )
    
    await manager.process_file(args.input_file, args.output_file, cache_file)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
