"""Validation utilities for translation."""

import re
from typing import List, Pattern


def compile_ignore_patterns(ignore_patterns: List[str]) -> List[Pattern]:
    """Compile regex patterns for ignoring text during translation."""
    compiled_patterns = []
    if ignore_patterns:
        for pattern in ignore_patterns:
            try:
                compiled = re.compile(pattern)
                compiled_patterns.append(compiled)
            except re.error as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Invalid regex pattern '{pattern}': {e}. Ignoring.")
    return compiled_patterns


def should_ignore(text: str, ignore_patterns: List[Pattern]) -> bool:
    """Check if text matches any ignore pattern."""
    if not isinstance(text, str) or not text:
        return False
    for pattern in ignore_patterns:
        if pattern.search(text):
            return True
    return False


def has_duplicate_token_error(translated_text: str, original_text: str = "") -> bool:
    """Detect if translation has duplicate token errors (repetitive patterns)."""
    if not isinstance(translated_text, str) or not translated_text:
        return False
    
    # Remove leading/trailing whitespace and common prefixes like "- "
    text = translated_text.strip()
    if text.startswith("- "):
        text = text[2:].strip()
    
    # Split by common separators (comma, space, period)
    # Check for excessive repetition of the same word/phrase
    words = text.replace(",", " ").replace(".", " ").split()
    
    if len(words) < 3:
        # Very short translations might be legitimate, but check for repetition
        if len(words) >= 2 and len(set(words)) == 1:
            return True
        return False
    
    # Check if the same word appears too many times (more than 50% of words)
    word_counts = {}
    for word in words:
        word_lower = word.lower()
        word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
    
    total_words = len(words)
    max_count = max(word_counts.values()) if word_counts else 0
    
    # If a single word appears more than 50% of the time, it's likely an error
    if max_count > total_words * 0.5:
        return True
    
    # Check for repetitive patterns (same 2-3 word sequence repeating)
    # Look for sequences that repeat more than 3 times
    for seq_len in [2, 3]:
        for i in range(len(words) - seq_len * 3):
            seq = tuple(words[i:i+seq_len])
            count = 1
            j = i + seq_len
            while j + seq_len <= len(words) and tuple(words[j:j+seq_len]) == seq:
                count += 1
                j += seq_len
            if count >= 4:  # Same sequence repeated 4+ times
                return True
    
    return False

