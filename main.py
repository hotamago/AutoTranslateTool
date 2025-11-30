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
        help="Translation service(s) - single service or comma-separated mix (e.g., 'google,cerebras,lmstudio,nvidia')"
    )
    parser.add_argument(
        "-c", "--concurrency", type=str, default="100",
        help="Concurrency level(s) - single value or comma-separated per service (e.g., '100,4' for google,cerebras). Default: 100"
    )
    parser.add_argument(
        "-b", "--batch_size", type=str, default="200",
        help="Batch size(s) - single value or comma-separated per service (e.g., '1,200' for google,cerebras). Default: 200"
    )
    parser.add_argument(
        "-k", "--api_key",
        help="API key for services that require it (e.g., bing)"
    )
    parser.add_argument(
        "--api_key_cerebras",
        help="API key for Cerebras service"
    )
    parser.add_argument(
        "--api_key_nvidia",
        help="API key for NVIDIA service"
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
    
    # Parse comma-separated services
    services = [s.strip().lower() for s in args.service.split(',') if s.strip()]
    if not services:
        logger.error("No valid services specified")
        return
    
    # Parse comma-separated concurrency values
    concurrency_values = [c.strip() for c in args.concurrency.split(',') if c.strip()]
    concurrency_ints = []
    for c in concurrency_values:
        try:
            concurrency_ints.append(int(c))
        except ValueError:
            logger.warning(f"Invalid concurrency value '{c}', using default 100")
            concurrency_ints.append(100)
    
    # Parse comma-separated batch_size values
    batch_size_values = [b.strip() for b in args.batch_size.split(',') if b.strip()]
    batch_size_ints = []
    for b in batch_size_values:
        try:
            batch_size_ints.append(int(b))
        except ValueError:
            logger.warning(f"Invalid batch_size value '{b}', using default 200")
            batch_size_ints.append(200)
    
    # Match concurrency and batch_size to services
    # If fewer values than services, use last value for remaining services
    # If more values than services, ignore extra values
    service_concurrency = {}
    service_batch_size = {}
    
    for i, service in enumerate(services):
        if i < len(concurrency_ints):
            service_concurrency[service] = concurrency_ints[i]
        elif concurrency_ints:
            service_concurrency[service] = concurrency_ints[-1]  # Use last value
        else:
            service_concurrency[service] = 100  # Default
        
        if i < len(batch_size_ints):
            service_batch_size[service] = batch_size_ints[i]
        elif batch_size_ints:
            service_batch_size[service] = batch_size_ints[-1]  # Use last value
        else:
            service_batch_size[service] = 200  # Default
    
    config_str = ", ".join([
        f"{s}: concurrency={service_concurrency[s]}, batch_size={service_batch_size[s]}" 
        for s in services
    ])
    logger.info(f"Service configurations: {config_str}")
    
    manager = TranslationManager(
        services=services,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        service_concurrency=service_concurrency,
        api_key=args.api_key,
        api_key_cerebras=args.api_key_cerebras,
        api_key_nvidia=args.api_key_nvidia,
        service_batch_size=service_batch_size,
        ignore_patterns=args.ignore_regex
    )
    
    await manager.process_file(args.input_file, args.output_file, cache_file)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
