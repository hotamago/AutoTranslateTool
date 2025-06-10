#!/usr/bin/env python3
"""
Auto Translation Tool
Translates JSON files using Google Translate with progress tracking and resume functionality.
"""

import json
import argparse
import os
import sys
from typing import Dict, Any, List, Tuple, Set
from pathlib import Path
import time

from googletrans import Translator
from tqdm import tqdm


class TranslationTracker:
    """Handles progress tracking and resume functionality."""
    
    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.completed_keys = set()
        self.load_progress()
    
    def load_progress(self):
        """Load previously completed translations."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.completed_keys = set(data.get('completed_keys', []))
            except (json.JSONDecodeError, FileNotFoundError):
                self.completed_keys = set()
    
    def save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump({'completed_keys': list(self.completed_keys)}, f)
    
    def mark_completed(self, key_path: str):
        """Mark a key path as completed."""
        self.completed_keys.add(key_path)
        self.save_progress()
    
    def mark_completed_batch(self, key_paths: List[str]):
        """Mark multiple key paths as completed."""
        self.completed_keys.update(key_paths)
        self.save_progress()
    
    def is_completed(self, key_path: str) -> bool:
        """Check if a key path is already completed."""
        return key_path in self.completed_keys
    
    def cleanup(self):
        """Remove progress file after successful completion."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)


class JSONTranslator:
    """Main translation class."""
    
    def __init__(self, src_lang: str = 'auto', dest_lang: str = 'en', output_format: str = 'preserve', batch_size: int = 1):
        self.translator = Translator()
        self.src_lang = src_lang
        self.dest_lang = dest_lang
        self.output_format = output_format  # 'preserve' or 'mapping'
        self.batch_size = max(1, batch_size)  # Ensure batch_size is at least 1
        self.translation_cache = {}
    
    def translate_text(self, text: str) -> str:
        """Translate a single text string with caching."""
        if not text or not text.strip():
            return text
        
        # Check cache first
        cache_key = f"{self.src_lang}->{self.dest_lang}:{text}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Add delay to avoid rate limiting
            time.sleep(0.1)
            result = self.translator.translate(text, src=self.src_lang, dest=self.dest_lang)
            translated = result.text
            
            # Cache the result
            self.translation_cache[cache_key] = translated
            return translated
            
        except Exception as e:
            print(f"Translation error for '{text}': {e}")
            return text  # Return original text on error
    
    def translate_batch(self, texts: List[str]) -> Dict[str, str]:
        """Translate multiple texts in a single request."""
        if not texts:
            return {}
        
        # Filter out empty texts and check cache
        texts_to_translate = []
        results = {}
        
        for text in texts:
            if not text or not text.strip():
                results[text] = text
                continue
            
            cache_key = f"{self.src_lang}->{self.dest_lang}:{text}"
            if cache_key in self.translation_cache:
                results[text] = self.translation_cache[cache_key]
            else:
                texts_to_translate.append(text)
        
        # Translate remaining texts
        if texts_to_translate:
            try:
                # Add delay to avoid rate limiting
                time.sleep(0.2)
                
                if len(texts_to_translate) == 1:
                    # Single text translation
                    result = self.translator.translate(texts_to_translate[0], src=self.src_lang, dest=self.dest_lang)
                    translated = result.text
                    results[texts_to_translate[0]] = translated
                    self.translation_cache[f"{self.src_lang}->{self.dest_lang}:{texts_to_translate[0]}"] = translated
                else:
                    # Batch translation
                    batch_results = self.translator.translate(texts_to_translate, src=self.src_lang, dest=self.dest_lang)
                    
                    # Handle both single result and list of results
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]
                    
                    for i, original_text in enumerate(texts_to_translate):
                        if i < len(batch_results):
                            translated = batch_results[i].text
                            results[original_text] = translated
                            self.translation_cache[f"{self.src_lang}->{self.dest_lang}:{original_text}"] = translated
                        else:
                            # Fallback if batch result is incomplete
                            results[original_text] = original_text
                
            except Exception as e:
                print(f"Batch translation error: {e}")
                # Fallback to individual translation
                for text in texts_to_translate:
                    try:
                        individual_result = self.translate_text(text)
                        results[text] = individual_result
                    except:
                        results[text] = text  # Use original text as fallback
        
        return results
    
    def chunk_list(self, lst: List, chunk_size: int) -> List[List]:
        """Split a list into chunks of specified size."""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
    def get_all_text_values(self, data: Any) -> Set[str]:
        """Get all unique text values from nested JSON structure."""
        texts = set()
        
        if isinstance(data, str) and data.strip():
            texts.add(data)
        elif isinstance(data, dict):
            for value in data.values():
                texts.update(self.get_all_text_values(value))
        elif isinstance(data, list):
            for item in data:
                texts.update(self.get_all_text_values(item))
        
        return texts
    
    def get_all_text_paths(self, data: Dict[str, Any], parent_path: str = "") -> List[Tuple[str, str]]:
        """Get all text values and their paths from nested JSON."""
        paths = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{parent_path}.{key}" if parent_path else key
                if isinstance(value, str):
                    paths.append((current_path, value))
                elif isinstance(value, (dict, list)):
                    paths.extend(self.get_all_text_paths(value, current_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{parent_path}[{i}]"
                if isinstance(item, str):
                    paths.append((current_path, item))
                elif isinstance(item, (dict, list)):
                    paths.extend(self.get_all_text_paths(item, current_path))
        
        return paths
    
    def set_value_by_path(self, data: Dict[str, Any], path: str, value: str):
        """Set a value in nested JSON by path."""
        parts = path.split('.')
        current = data
        
        for part in parts[:-1]:
            if '[' in part and part.endswith(']'):
                # Handle array access
                key, index_str = part.split('[')
                index = int(index_str[:-1])
                if key:
                    current = current[key]
                current = current[index]
            else:
                current = current[part]
        
        final_part = parts[-1]
        if '[' in final_part and final_part.endswith(']'):
            key, index_str = final_part.split('[')
            index = int(index_str[:-1])
            if key:
                current = current[key]
            current[index] = value
        else:
            current[final_part] = value
    
    def translate_json_file(self, input_file: str, output_file: str, progress_file: str = None):
        """Translate entire JSON file with progress tracking."""
        # Setup progress tracking
        if progress_file is None:
            progress_file = f"{input_file}.progress"
        
        tracker = TranslationTracker(progress_file)
        
        # Load input JSON
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading input file '{input_file}': {e}")
            return False
        
        if self.output_format == 'mapping':
            return self._translate_to_mapping_format(data, output_file, tracker)
        else:
            return self._translate_preserve_structure(data, output_file, tracker)
    
    def _translate_to_mapping_format(self, data: Any, output_file: str, tracker: TranslationTracker):
        """Translate to simple key-value mapping format."""
        # Get all unique text values
        unique_texts = self.get_all_text_values(data)
        unique_texts = {text for text in unique_texts if text.strip()}  # Remove empty strings
        
        if not unique_texts:
            print("No text content found to translate.")
            return False
        
        # Load existing translations if output file exists
        translated_mapping = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    translated_mapping = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Filter out already completed translations
        remaining_texts = [text for text in unique_texts 
                          if not tracker.is_completed(text) and text not in translated_mapping]
        
        if not remaining_texts:
            print("All translations already completed!")
            # Ensure all unique texts are in the output (in case of partial completion)
            for text in unique_texts:
                if text not in translated_mapping:
                    translated_mapping[text] = text  # Fallback to original
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_mapping, f, ensure_ascii=False, indent=2)
            tracker.cleanup()
            return True
        
        print(f"Found {len(unique_texts)} unique text entries")
        print(f"Remaining to translate: {len(remaining_texts)}")
        print(f"Already completed: {len(unique_texts) - len(remaining_texts)}")
        print(f"Translating from '{self.src_lang}' to '{self.dest_lang}'")
        print(f"Output format: Key-Value mapping")
        print(f"Batch size: {self.batch_size}")
        
        # Split texts into batches
        text_batches = self.chunk_list(remaining_texts, self.batch_size)
        total_batches = len(text_batches)
        successful_translations = 0
        
        with tqdm(total=len(remaining_texts), 
                 desc="Translating", 
                 unit="entries",
                 bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Batch: {postfix}") as pbar:
            
            for batch_idx, text_batch in enumerate(text_batches):
                try:
                    # Translate the batch
                    batch_results = self.translate_batch(text_batch)
                    
                    # Update results and tracking
                    for original_text in text_batch:
                        if original_text in batch_results:
                            translated_mapping[original_text] = batch_results[original_text]
                            tracker.mark_completed(original_text)
                            successful_translations += 1
                        else:
                            # Fallback to original text
                            translated_mapping[original_text] = original_text
                            tracker.mark_completed(original_text)
                    
                    # Save progress periodically (every 5 batches or 50 items)
                    if (batch_idx + 1) % 5 == 0 or successful_translations % 50 == 0:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(translated_mapping, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    print(f"\nError translating batch {batch_idx + 1}: {e}")
                    # Process batch items individually as fallback
                    for text in text_batch:
                        try:
                            translated_text = self.translate_text(text)
                            translated_mapping[text] = translated_text
                            tracker.mark_completed(text)
                            successful_translations += 1
                        except:
                            translated_mapping[text] = text  # Use original as fallback
                            tracker.mark_completed(text)
                
                # Update progress bar
                pbar.update(len(text_batch))
                current_text = text_batch[0] if text_batch else ""
                pbar.set_postfix_str(f"{batch_idx + 1}/{total_batches} | {current_text[:25]}{'...' if len(current_text) > 25 else ''}")
        
        # Save final result
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_mapping, f, ensure_ascii=False, indent=2)
            
            print(f"\nTranslation completed!")
            print(f"Successfully translated: {successful_translations}/{len(remaining_texts)} entries")
            print(f"Total unique texts: {len(translated_mapping)}")
            print(f"Processed {total_batches} batches of size {self.batch_size}")
            print(f"Output saved to: {output_file}")
            
            # Cleanup progress file on successful completion
            if successful_translations >= len(remaining_texts):
                tracker.cleanup()
                print("Progress file cleaned up.")
            else:
                print(f"Progress saved to: {tracker.progress_file}")
                print("You can resume translation later by running the same command.")
            
            return True
            
        except Exception as e:
            print(f"Error saving output file: {e}")
            return False
    
    def _translate_preserve_structure(self, data: Any, output_file: str, tracker: TranslationTracker):
        """Translate while preserving original JSON structure."""
        # Create a copy for translation
        translated_data = json.loads(json.dumps(data))
        
        # Get all text paths
        text_paths = self.get_all_text_paths(data)
        
        if not text_paths:
            print("No text content found to translate.")
            return False
        
        # Filter out already completed translations
        remaining_paths = [(path, text) for path, text in text_paths 
                          if not tracker.is_completed(path)]
        
        if not remaining_paths:
            print("All translations already completed!")
            # Still save the output file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            tracker.cleanup()
            return True
        
        print(f"Found {len(text_paths)} total text entries")
        print(f"Remaining to translate: {len(remaining_paths)}")
        print(f"Already completed: {len(text_paths) - len(remaining_paths)}")
        print(f"Translating from '{self.src_lang}' to '{self.dest_lang}'")
        print(f"Output format: Preserve structure")
        print(f"Batch size: {self.batch_size}")
        
        # Load existing partial results if output file exists
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    translated_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Split paths into batches
        path_batches = self.chunk_list(remaining_paths, self.batch_size)
        total_batches = len(path_batches)
        successful_translations = 0
        
        with tqdm(total=len(remaining_paths), 
                 desc="Translating", 
                 unit="entries",
                 bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Batch: {postfix}") as pbar:
            
            for batch_idx, path_batch in enumerate(path_batches):
                try:
                    # Extract texts for batch translation
                    batch_texts = [text for _, text in path_batch]
                    batch_results = self.translate_batch(batch_texts)
                    
                    # Apply translated results to the data structure
                    batch_paths_completed = []
                    for (path, original_text) in path_batch:
                        if original_text in batch_results:
                            translated_text = batch_results[original_text]
                            self.set_value_by_path(translated_data, path, translated_text)
                            batch_paths_completed.append(path)
                            successful_translations += 1
                        else:
                            # Keep original text if translation failed
                            batch_paths_completed.append(path)
                    
                    # Mark batch as completed
                    tracker.mark_completed_batch(batch_paths_completed)
                    
                    # Save progress periodically (every 5 batches or 50 items)
                    if (batch_idx + 1) % 5 == 0 or successful_translations % 50 == 0:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(translated_data, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    print(f"\nError translating batch {batch_idx + 1}: {e}")
                    # Process batch items individually as fallback
                    for path, original_text in path_batch:
                        try:
                            translated_text = self.translate_text(original_text)
                            self.set_value_by_path(translated_data, path, translated_text)
                            tracker.mark_completed(path)
                            successful_translations += 1
                        except Exception as individual_e:
                            print(f"\nError translating path '{path}': {individual_e}")
                            tracker.mark_completed(path)  # Mark as completed to avoid retrying
                
                # Update progress bar
                pbar.update(len(path_batch))
                current_text = path_batch[0][1] if path_batch else ""
                pbar.set_postfix_str(f"{batch_idx + 1}/{total_batches} | {current_text[:25]}{'...' if len(current_text) > 25 else ''}")
        
        # Save final result
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            
            print(f"\nTranslation completed!")
            print(f"Successfully translated: {successful_translations}/{len(remaining_paths)} entries")
            print(f"Processed {total_batches} batches of size {self.batch_size}")
            print(f"Output saved to: {output_file}")
            
            # Cleanup progress file on successful completion
            if successful_translations >= len(remaining_paths):
                tracker.cleanup()
                print("Progress file cleaned up.")
            else:
                print(f"Progress saved to: {tracker.progress_file}")
                print("You can resume translation later by running the same command.")
            
            return True
            
        except Exception as e:
            print(f"Error saving output file: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Translate JSON files using Google Translate with batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Key-value mapping format with default batch size (1)
  python main.py input.json output.json --format mapping
  
  # Use batch processing with size 5
  python main.py input.json output.json --format mapping --batch-size 5
  
  # Preserve original structure with batch processing
  python main.py input.json output.json --format preserve --batch-size 10
  
  # Translate to Spanish with batch size 3
  python main.py input.json output.json --src-lang en --dest-lang es --batch-size 3
        """
    )
    
    parser.add_argument('input_file', help='Input JSON file to translate')
    parser.add_argument('output_file', help='Output JSON file for translated content')
    parser.add_argument('--src-lang', default='auto', 
                       help='Source language code (default: auto-detect)')
    parser.add_argument('--dest-lang', default='en', 
                       help='Target language code (default: en)')
    parser.add_argument('--format', choices=['mapping', 'preserve'], default='mapping',
                       help='Output format: "mapping" for key-value pairs, "preserve" to keep structure (default: mapping)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of texts to translate in each batch (default: 1, max recommended: 10)')
    parser.add_argument('--progress-file', 
                       help='Custom progress file path (default: input_file.progress)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    # Validate batch size
    if args.batch_size < 1:
        print("Error: Batch size must be at least 1.")
        sys.exit(1)
    elif args.batch_size > 20:
        print("Warning: Large batch sizes (>20) may cause API errors. Consider using smaller batches.")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize translator
    translator = JSONTranslator(
        src_lang=args.src_lang, 
        dest_lang=args.dest_lang, 
        output_format=args.format,
        batch_size=args.batch_size
    )
    
    # Start translation
    print(f"Starting translation...")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    
    success = translator.translate_json_file(
        args.input_file, 
        args.output_file, 
        args.progress_file
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
