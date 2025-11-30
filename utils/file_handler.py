"""File I/O utilities."""

import json
import logging
from typing import Dict

from .cache import load_translated_map

logger = logging.getLogger(__name__)


def load_json_file(input_file: str) -> Dict:
    """Load JSON file and return as dictionary."""
    logger.info(f"Loading input file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        raise


def save_json_file(output_file: str, data: Dict):
    """Save dictionary to JSON file."""
    logger.info(f"Saving final output to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Translation completed successfully.")
    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        raise


async def finalize_output(original_data: Dict, cache_file: str, output_file: str):
    """Reconstruct final output from cache and save to output file."""
    logger.info("Finalizing output file...")
    
    # Load all translated entries from cache
    translated_map = await load_translated_map(cache_file)
    
    # Create final dict in original order
    final_output = {}
    for k in original_data.keys():
        if k in translated_map:
            final_output[k] = translated_map[k]
        else:
            # Should not happen if all done, but if so, keep original
            final_output[k] = original_data[k]
    
    save_json_file(output_file, final_output)

