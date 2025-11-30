import argparse
import asyncio
import json
import logging
import sys
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Set, Dict, List, Optional

import aiofiles
import aiohttp
from deep_translator import GoogleTranslator, MicrosoftTranslator
from tqdm import tqdm

NLLB_LANG_MAP = {
    'en': 'eng_Latn',
    'vi': 'vie_Latn',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'zh': 'zho_Hans',
    'fr': 'fra_Latn',
    'es': 'spa_Latn',
    'de': 'deu_Latn',
    'ru': 'rus_Cyrl',
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(self, service: str, source_lang: str, target_lang: str, concurrency: int = 10, api_key: str = None, batch_size: int = 10, ignore_patterns: Optional[List[str]] = None, model_url: Optional[str] = None):
        self.service = service.lower()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.api_key = api_key
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.model_url = model_url
        self.semaphore = asyncio.Semaphore(concurrency)
        self.executor = ThreadPoolExecutor(max_workers=concurrency)
        
        # Compile ignore regex patterns
        self.ignore_patterns = []
        if ignore_patterns:
            for pattern in ignore_patterns:
                try:
                    compiled = re.compile(pattern)
                    self.ignore_patterns.append(compiled)
                    logger.info(f"Added ignore pattern: {pattern}")
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}. Ignoring.")
        
        # Reusable translator instances (optimization)
        self.google_translator = None
        self.bing_translator = None
        self.http_session = None  # For async HTTP requests
        
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.opus_model = None
        self.opus_tokenizer = None
        self.device = None

    def should_ignore(self, text: str) -> bool:
        """Check if text matches any ignore pattern."""
        if not isinstance(text, str) or not text:
            return False
        for pattern in self.ignore_patterns:
            if pattern.search(text):
                return True
        return False

    def has_duplicate_token_error(self, translated_text: str, original_text: str = "") -> bool:
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

    async def _init_http_session(self):
        """Initialize HTTP session for async requests."""
        if self.http_session is None:
            connector = aiohttp.TCPConnector(limit=self.concurrency, limit_per_host=self.concurrency)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.http_session

    async def _close_http_session(self):
        """Close HTTP session."""
        if self.http_session:
            await self.http_session.close()
            self.http_session = None

    async def translate_batch_google(self, texts: List[str]) -> List[str]:
        """Translate multiple texts using Google Translate API directly via aiohttp (faster)."""
        if not texts:
            return []
        
        session = await self._init_http_session()
        results = []
        
        # Process texts in parallel batches for better performance
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
                                translated_text = ''.join([item[0] if item and len(item) > 0 else '' for item in data[0] if item])
                                return translated_text if translated_text else text
                            return text
                        else:
                            # Fallback to deep_translator if API fails
                            logger.warning(f"Google API returned status {response.status} for '{text[:20]}...', using fallback")
                            if self.google_translator is None:
                                self.google_translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
                            loop = asyncio.get_running_loop()
                            translated = await loop.run_in_executor(
                                self.executor,
                                self.google_translator.translate,
                                text
                            )
                            return translated
                except Exception as e:
                    logger.error(f"Translation failed for '{text[:20]}...': {e}, using fallback")
                    # Fallback to deep_translator
                    try:
                        if self.google_translator is None:
                            self.google_translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
                        loop = asyncio.get_running_loop()
                        translated = await loop.run_in_executor(
                            self.executor,
                            self.google_translator.translate,
                            text
                        )
                        return translated
                    except Exception as e2:
                        logger.error(f"Fallback translation also failed for '{text[:20]}...': {e2}")
                        return text
        
        # Process all texts concurrently (up to concurrency limit)
        tasks = [translate_single(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return list(results)

    async def translate_with_google_fallback(self, text: str) -> str:
        """Translate text using Google Translate as fallback."""
        if not text:
            return ""
        
        async with self.semaphore:
            try:
                if self.google_translator is None:
                    self.google_translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
                loop = asyncio.get_running_loop()
                translated = await loop.run_in_executor(
                    self.executor, 
                    self.google_translator.translate, 
                    text
                )
                return translated
            except Exception as e:
                logger.error(f"Failed to translate with Google fallback '{text[:20]}...': {e}")
                return text

    def get_translator(self):
        """Get or create translator instance (reused for efficiency)."""
        if self.service == 'google':
            if self.google_translator is None:
                self.google_translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
            return self.google_translator
        elif self.service == 'bing':
            if not self.api_key:
                logger.warning("Bing (Microsoft) translator requires an API key. Please provide one with --api_key.")
            if self.bing_translator is None:
                self.bing_translator = MicrosoftTranslator(source=self.source_lang, target=self.target_lang, api_key=self.api_key)
            return self.bing_translator
        else:
            if self.service not in ['google', 'bing', 'nllb', 'opus-mt', 'lmstudio']:
                logger.warning(f"Service '{self.service}' not explicitly supported. Defaulting to Google.")
            if self.google_translator is None:
                self.google_translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
            return self.google_translator

    async def translate_text(self, text: str) -> str:
        """Translate single text (optimized with reusable translator)."""
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

    async def init_nllb(self):
        if self.service == 'nllb' and not self.nllb_model:
            logger.info("Loading NLLB model...")
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import torch
                
                # Enhanced GPU detection and diagnostics
                logger.info("Checking GPU availability...")
                logger.info(f"PyTorch version: {torch.__version__}")
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA version: {torch.version.cuda}")
                    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                        logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                else:
                    logger.warning("CUDA is not available. PyTorch may have been installed without CUDA support.")
                    logger.warning("To use GPU, install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
                
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logger.info(f"Using device: {self.device} (GPU)")
                else:
                    self.device = torch.device("cpu")
                    logger.warning(f"Using device: {self.device} (CPU)")
                
                # Create models directory if it doesn't exist
                models_dir = "./_models"
                os.makedirs(models_dir, exist_ok=True)
                
                model_name = "facebook/nllb-200-distilled-1.3B"
                
                # Check if model is already cached by looking for the model file
                # The model should be in: _models/models--facebook--nllb-200-distilled-1.3B/snapshots/*/pytorch_model.bin
                model_cached = False
                model_cache_path = os.path.join(models_dir, "models--facebook--nllb-200-distilled-1.3B")
                if os.path.exists(model_cache_path):
                    # Look for pytorch_model.bin in any snapshot directory
                    snapshots_dir = os.path.join(model_cache_path, "snapshots")
                    if os.path.exists(snapshots_dir):
                        for snapshot in os.listdir(snapshots_dir):
                            snapshot_path = os.path.join(snapshots_dir, snapshot)
                            model_file = os.path.join(snapshot_path, "pytorch_model.bin")
                            if os.path.exists(model_file):
                                model_cached = True
                                logger.info(f"Found cached model in {models_dir}")
                                break
                
                # Clean up any incomplete download files
                if os.path.exists(model_cache_path):
                    blobs_dir = os.path.join(model_cache_path, "blobs")
                    if os.path.exists(blobs_dir):
                        for blob_file in os.listdir(blobs_dir):
                            if blob_file.endswith(".incomplete"):
                                incomplete_path = os.path.join(blobs_dir, blob_file)
                                try:
                                    os.remove(incomplete_path)
                                    logger.info(f"Removed incomplete download: {blob_file}")
                                except Exception as e:
                                    logger.warning(f"Could not remove incomplete file {blob_file}: {e}")
                
                # Use local_files_only if model is cached to prevent re-downloading
                if model_cached:
                    logger.info(f"Loading from cache: {models_dir}")
                    self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=models_dir, local_files_only=True)
                    self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=models_dir, local_files_only=True)
                else:
                    logger.info(f"Downloading and caching model to: {models_dir}")
                    self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=models_dir)
                    self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=models_dir)
                
                # Move model to device after loading
                self.nllb_model = self.nllb_model.to(self.device)
                
                # Convert to float16 for faster inference if CUDA is available
                if isinstance(self.device, torch.device) and self.device.type == "cuda":
                    logger.info("Converting model to float16 for faster inference...")
                    self.nllb_model = self.nllb_model.half()  # Convert to float16
                    logger.info("✓ Model converted to float16")
                
                self.nllb_model.eval()  # Set to evaluation mode
                
                # Verify model is on the correct device
                if isinstance(self.device, torch.device) and self.device.type == "cuda":
                    model_device = next(self.nllb_model.parameters()).device
                    logger.info(f"Model loaded on device: {model_device}")
                    if model_device.type != "cuda":
                        logger.error(f"WARNING: Model is on {model_device} but expected CUDA!")
                    else:
                        logger.info("✓ Model successfully loaded on GPU")
                
                logger.info("NLLB model loaded.")
            except ImportError:
                logger.error("Transformers or Torch not installed. Please install them to use NLLB.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to load NLLB model: {e}")
                sys.exit(1)

    async def translate_batch_nllb(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        
        src = NLLB_LANG_MAP.get(self.source_lang, self.source_lang)
        tgt = NLLB_LANG_MAP.get(self.target_lang, self.target_lang)
        
        try:
            import torch
            self.nllb_tokenizer.src_lang = src
            inputs = self.nllb_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            
            # Move all input tensors to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get the target language token ID using convert_tokens_to_ids
            # (lang_code_to_id was removed in recent transformers versions)
            forced_bos_token_id = self.nllb_tokenizer.convert_tokens_to_ids(tgt)
            
            # Use torch.no_grad() for inference to save memory
            # Use mixed precision (autocast) for faster inference on GPU
            with torch.no_grad():
                if isinstance(self.device, torch.device) and self.device.type == "cuda":
                    # Use autocast for mixed precision inference (faster on modern GPUs)
                    with torch.amp.autocast('cuda'):
                        translated_tokens = self.nllb_model.generate(
                            **inputs, forced_bos_token_id=forced_bos_token_id, max_length=512
                        )
                else:
                    # CPU inference without autocast
                    translated_tokens = self.nllb_model.generate(
                        **inputs, forced_bos_token_id=forced_bos_token_id, max_length=512
                    )
            
            # Move tokens back to CPU for decoding
            translated_tokens = translated_tokens.cpu()
            return self.nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"NLLB batch translation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return texts # Return original on failure

    async def init_opus(self):
        if self.service == 'opus-mt' and not self.opus_model:
            if not self.model_url:
                logger.error("opus-mt service requires --model_url parameter (e.g., Helsinki-NLP/opus-mt-ja-vi)")
                sys.exit(1)
            
            logger.info(f"Loading opus-mt model: {self.model_url}")
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import torch
                
                # Enhanced GPU detection and diagnostics
                logger.info("Checking GPU availability...")
                logger.info(f"PyTorch version: {torch.__version__}")
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA version: {torch.version.cuda}")
                    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                        logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                else:
                    logger.warning("CUDA is not available. PyTorch may have been installed without CUDA support.")
                    logger.warning("To use GPU, install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
                
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    logger.info(f"Using device: {self.device} (GPU)")
                else:
                    self.device = torch.device("cpu")
                    logger.warning(f"Using device: {self.device} (CPU)")
                
                # Create models directory if it doesn't exist
                models_dir = "./_models"
                os.makedirs(models_dir, exist_ok=True)
                
                # Convert model URL to cache path format (e.g., Helsinki-NLP/opus-mt-ja-vi -> models--Helsinki-NLP--opus-mt-ja-vi)
                model_cache_name = self.model_url.replace("/", "--")
                model_cache_path = os.path.join(models_dir, f"models--{model_cache_name}")
                
                # Check if model is already cached
                model_cached = False
                if os.path.exists(model_cache_path):
                    snapshots_dir = os.path.join(model_cache_path, "snapshots")
                    if os.path.exists(snapshots_dir):
                        for snapshot in os.listdir(snapshots_dir):
                            snapshot_path = os.path.join(snapshots_dir, snapshot)
                            # Check for model files (could be pytorch_model.bin, model.safetensors, etc.)
                            model_files = [f for f in os.listdir(snapshot_path) if f.endswith(('.bin', '.safetensors'))]
                            if model_files:
                                model_cached = True
                                logger.info(f"Found cached model in {models_dir}")
                                break
                
                # Clean up any incomplete download files
                if os.path.exists(model_cache_path):
                    blobs_dir = os.path.join(model_cache_path, "blobs")
                    if os.path.exists(blobs_dir):
                        for blob_file in os.listdir(blobs_dir):
                            if blob_file.endswith(".incomplete"):
                                incomplete_path = os.path.join(blobs_dir, blob_file)
                                try:
                                    os.remove(incomplete_path)
                                    logger.info(f"Removed incomplete download: {blob_file}")
                                except Exception as e:
                                    logger.warning(f"Could not remove incomplete file {blob_file}: {e}")
                
                # Use local_files_only if model is cached to prevent re-downloading
                if model_cached:
                    logger.info(f"Loading from cache: {models_dir}")
                    self.opus_tokenizer = AutoTokenizer.from_pretrained(self.model_url, cache_dir=models_dir, local_files_only=True)
                    self.opus_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_url, cache_dir=models_dir, local_files_only=True)
                else:
                    logger.info(f"Downloading and caching model to: {models_dir}")
                    self.opus_tokenizer = AutoTokenizer.from_pretrained(self.model_url, cache_dir=models_dir)
                    self.opus_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_url, cache_dir=models_dir)
                
                # Move model to device after loading
                self.opus_model = self.opus_model.to(self.device)
                
                # Convert to float16 for faster inference if CUDA is available
                if isinstance(self.device, torch.device) and self.device.type == "cuda":
                    logger.info("Converting model to float16 for faster inference...")
                    self.opus_model = self.opus_model.half()  # Convert to float16
                    logger.info("✓ Model converted to float16")
                
                self.opus_model.eval()  # Set to evaluation mode
                
                # Verify model is on the correct device
                if isinstance(self.device, torch.device) and self.device.type == "cuda":
                    model_device = next(self.opus_model.parameters()).device
                    logger.info(f"Model loaded on device: {model_device}")
                    if model_device.type != "cuda":
                        logger.error(f"WARNING: Model is on {model_device} but expected CUDA!")
                    else:
                        logger.info("✓ Model successfully loaded on GPU")
                
                logger.info("opus-mt model loaded.")
            except ImportError:
                logger.error("Transformers or Torch not installed. Please install them to use opus-mt.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to load opus-mt model: {e}")
                sys.exit(1)

    async def translate_batch_opus(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        
        try:
            import torch
            inputs = self.opus_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            
            # Move all input tensors to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use torch.no_grad() for inference to save memory
            # Use mixed precision (autocast) for faster inference on GPU
            with torch.no_grad():
                if isinstance(self.device, torch.device) and self.device.type == "cuda":
                    # Use autocast for mixed precision inference (faster on modern GPUs)
                    with torch.amp.autocast('cuda'):
                        translated_tokens = self.opus_model.generate(
                            **inputs, max_length=512
                        )
                else:
                    # CPU inference without autocast
                    translated_tokens = self.opus_model.generate(
                        **inputs, max_length=512
                    )
            
            # Move tokens back to CPU for decoding
            translated_tokens = translated_tokens.cpu()
            return self.opus_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"opus-mt batch translation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return texts # Return original on failure

    async def translate_batch_lmstudio(self, items: dict) -> dict:
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
                { "role": "system", "content": "You are a helpful assistant that translates JSON files." },
                { "role": "user", "content": prompt }
            ],
            "temperature": 0.3,
            "max_tokens": -1,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("http://127.0.0.1:1234/v1/chat/completions", json=payload) as response:
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

    async def load_cache(self, cache_file: str) -> Set[str]:
        completed_keys = set()
        if not os.path.exists(cache_file):
            return completed_keys
        
        logger.info(f"Loading cache from {cache_file}...")
        try:
            # Use errors='replace' to handle invalid UTF-8 bytes gracefully
            # Invalid bytes will be replaced with the Unicode replacement character ()
            async with aiofiles.open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
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
            await self._close_http_session()
            return

        logger.info(f"Starting translation from {self.source_lang} to {self.target_lang} using {self.service}...")
        
        try:
            if self.service == 'nllb':
                await self.init_nllb()
                
                keys = list(pending_items.keys())
                with tqdm(total=len(keys), desc="Translating (NLLB Batch)", unit="item") as pbar:
                    for i in range(0, len(keys), self.batch_size):
                        batch_keys = keys[i:i+self.batch_size]
                        batch_values = [pending_items[k] for k in batch_keys]
                        
                        # Filter out non-string values and ignored patterns for translation
                        to_translate_indices = []
                        to_translate_values = []
                        for idx, v in enumerate(batch_values):
                            if isinstance(v, str) and not self.should_ignore(v):
                                to_translate_indices.append(idx)
                                to_translate_values.append(v)
                        
                        if to_translate_values:
                            translated_values = await self.translate_batch_nllb(to_translate_values)
                        else:
                            translated_values = []
                            
                        # Reconstruct batch result and check for duplicate token errors
                        trans_idx = 0
                        for idx, key in enumerate(batch_keys):
                            val = batch_values[idx]
                            if isinstance(val, str):
                                if self.should_ignore(val):
                                    # Skip translation, use original value
                                    await self.append_to_cache(cache_file, key, val)
                                elif trans_idx < len(translated_values):
                                    translated_val = translated_values[trans_idx]
                                    
                                    # Check for duplicate token errors
                                    if self.has_duplicate_token_error(translated_val, val):
                                        logger.warning(f"NLLB duplicate token error detected for key '{key[:50]}...'. Falling back to Google Translate.")
                                        # Fallback to Google Translate
                                        try:
                                            translated_val = await self.translate_with_google_fallback(val)
                                            logger.info(f"Google Translate fallback successful for key '{key[:50]}...'")
                                        except Exception as e:
                                            logger.error(f"Google Translate fallback failed for key '{key[:50]}...': {e}")
                                            # Use original value if fallback also fails
                                            translated_val = val
                                    
                                    await self.append_to_cache(cache_file, key, translated_val)
                                    trans_idx += 1
                                else:
                                    # Should not happen unless translation returned fewer items
                                    logger.error(f"Mismatch in translation count for key {key}")
                            else:
                                await self.append_to_cache(cache_file, key, val)
                        
                        pbar.update(len(batch_keys))
                        
            elif self.service == 'opus-mt':
                await self.init_opus()
                
                keys = list(pending_items.keys())
                with tqdm(total=len(keys), desc="Translating (opus-mt Batch)", unit="item") as pbar:
                    for i in range(0, len(keys), self.batch_size):
                        batch_keys = keys[i:i+self.batch_size]
                        batch_values = [pending_items[k] for k in batch_keys]
                        
                        # Filter out non-string values and ignored patterns for translation
                        to_translate_indices = []
                        to_translate_values = []
                        for idx, v in enumerate(batch_values):
                            if isinstance(v, str) and not self.should_ignore(v):
                                to_translate_indices.append(idx)
                                to_translate_values.append(v)
                        
                        if to_translate_values:
                            translated_values = await self.translate_batch_opus(to_translate_values)
                        else:
                            translated_values = []
                            
                        # Reconstruct batch result and check for duplicate token errors
                        trans_idx = 0
                        for idx, key in enumerate(batch_keys):
                            val = batch_values[idx]
                            if isinstance(val, str):
                                if self.should_ignore(val):
                                    # Skip translation, use original value
                                    await self.append_to_cache(cache_file, key, val)
                                elif trans_idx < len(translated_values):
                                    translated_val = translated_values[trans_idx]
                                    
                                    # Check for duplicate token errors
                                    if self.has_duplicate_token_error(translated_val, val):
                                        logger.warning(f"opus-mt duplicate token error detected for key '{key[:50]}...'. Falling back to Google Translate.")
                                        # Fallback to Google Translate
                                        try:
                                            translated_val = await self.translate_with_google_fallback(val)
                                            logger.info(f"Google Translate fallback successful for key '{key[:50]}...'")
                                        except Exception as e:
                                            logger.error(f"Google Translate fallback failed for key '{key[:50]}...': {e}")
                                            # Use original value if fallback also fails
                                            translated_val = val
                                    
                                    await self.append_to_cache(cache_file, key, translated_val)
                                    trans_idx += 1
                                else:
                                    # Should not happen unless translation returned fewer items
                                    logger.error(f"Mismatch in translation count for key {key}")
                            else:
                                await self.append_to_cache(cache_file, key, val)
                        
                        pbar.update(len(batch_keys))
                        
            elif self.service == 'lmstudio':
                keys = list(pending_items.keys())
                with tqdm(total=len(keys), desc="Translating (LM Studio Batch)", unit="item") as pbar:
                    for i in range(0, len(keys), self.batch_size):
                        batch_keys = keys[i:i+self.batch_size]
                        batch_items = {k: pending_items[k] for k in batch_keys}
                        
                        # Separate ignored items from items to translate
                        items_to_translate = {}
                        ignored_items = {}
                        for k, v in batch_items.items():
                            if isinstance(v, str) and self.should_ignore(v):
                                ignored_items[k] = v
                            else:
                                items_to_translate[k] = v
                        
                        # Translate only non-ignored items
                        translated_batch = {}
                        if items_to_translate:
                            translated_batch = await self.translate_batch_lmstudio(items_to_translate)
                        
                        # If translation failed entirely, translated_batch might be empty
                        # We should probably retry or skip. For now, we skip updating cache so it can be retried later.
                        if not translated_batch and items_to_translate:
                            logger.warning(f"Batch {i//self.batch_size} failed or returned empty.")
                            # Optionally write failures? No, keep them pending.
                        else:
                            # Write translated items
                            for k, v in translated_batch.items():
                                await self.append_to_cache(cache_file, k, v)
                            # Write ignored items (original values)
                            for k, v in ignored_items.items():
                                await self.append_to_cache(cache_file, k, v)
                                
                        pbar.update(len(batch_keys))

            else:
                # Use batch processing for Google Translate for better performance
                if self.service == 'google':
                    keys = list(pending_items.keys())
                    with tqdm(total=len(keys), desc="Translating (Google Batch)", unit="item") as pbar:
                        for i in range(0, len(keys), self.batch_size):
                            batch_keys = keys[i:i+self.batch_size]
                            batch_values = [pending_items[k] for k in batch_keys]
                            
                            # Filter out non-string values and ignored patterns for translation
                            to_translate_indices = []
                            to_translate_values = []
                            for idx, v in enumerate(batch_values):
                                if isinstance(v, str) and not self.should_ignore(v):
                                    to_translate_indices.append(idx)
                                    to_translate_values.append(v)
                            
                            if to_translate_values:
                                # Use optimized batch translation
                                translated_values = await self.translate_batch_google(to_translate_values)
                            else:
                                translated_values = []
                                
                            # Reconstruct batch result
                            trans_idx = 0
                            for idx, key in enumerate(batch_keys):
                                val = batch_values[idx]
                                if isinstance(val, str):
                                    if self.should_ignore(val):
                                        # Skip translation, use original value
                                        await self.append_to_cache(cache_file, key, val)
                                    elif trans_idx < len(translated_values):
                                        translated_val = translated_values[trans_idx]
                                        await self.append_to_cache(cache_file, key, translated_val)
                                        trans_idx += 1
                                    else:
                                        logger.error(f"Mismatch in translation count for key {key}")
                                else:
                                    await self.append_to_cache(cache_file, key, val)
                            
                            pbar.update(len(batch_keys))
                else:
                    # For other services (Bing, etc.), use individual translation
                    with tqdm(total=len(pending_items), desc="Translating", unit="key") as pbar:
                        async def process_item(k, v):
                            if isinstance(v, str):
                                if self.should_ignore(v):
                                    # Skip translation, use original value
                                    await self.append_to_cache(cache_file, k, v)
                                else:
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
        finally:
            # Always close HTTP session if it was opened
            await self._close_http_session()

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
                # Use errors='replace' to handle invalid UTF-8 bytes gracefully
                # Invalid bytes will be replaced with the Unicode replacement character ()
                async with aiofiles.open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
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
    parser.add_argument("-m", "--service", default="google", help="Translation service (google, bing, nllb, opus-mt, lmstudio)")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrency level (default: 10)")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Batch size for NLLB/opus-mt/LM Studio (default: 10)")
    parser.add_argument("-k", "--api_key", help="API key for services that require it (e.g., bing)")
    parser.add_argument("--model_url", help="Model URL/identifier for opus-mt service (e.g., Helsinki-NLP/opus-mt-ja-vi)")
    parser.add_argument("--cache_file", help="Path to cache file (JSONL). Defaults to <output_file>.cache")
    parser.add_argument("--ignore_regex", action="append", help="Regex pattern to ignore during translation (can be specified multiple times). Example: '^[0-9]+$' to ignore pure numbers")

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
