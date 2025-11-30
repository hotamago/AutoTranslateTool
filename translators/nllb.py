"""NLLB (No Language Left Behind) translation service implementation."""

import os
import sys
import logging
from typing import List, Optional

import torch

from .base import BaseTranslator
from config.constants import NLLB_LANG_MAP

logger = logging.getLogger(__name__)


class NLLBTranslatorService(BaseTranslator):
    """NLLB translation service."""
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None
    ):
        super().__init__(source_lang, target_lang, concurrency, api_key)
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.device = None
    
    async def initialize(self):
        """Initialize NLLB model."""
        if self.nllb_model is not None:
            return
        
        logger.info("Loading NLLB model...")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Enhanced GPU detection and diagnostics
            logger.info("Checking GPU availability...")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    logger.info(
                        f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                    )
            else:
                logger.warning(
                    "CUDA is not available. PyTorch may have been installed without CUDA support."
                )
                logger.warning(
                    "To use GPU, install PyTorch with CUDA: "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu121"
                )
            
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
            
            # Check if model is already cached
            model_cached = False
            model_cache_path = os.path.join(
                models_dir, "models--facebook--nllb-200-distilled-1.3B"
            )
            if os.path.exists(model_cache_path):
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
                                logger.warning(
                                    f"Could not remove incomplete file {blob_file}: {e}"
                                )
            
            # Load model
            if model_cached:
                logger.info(f"Loading from cache: {models_dir}")
                self.nllb_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=models_dir, local_files_only=True
                )
                self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, cache_dir=models_dir, local_files_only=True
                )
            else:
                logger.info(f"Downloading and caching model to: {models_dir}")
                self.nllb_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=models_dir
                )
                self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, cache_dir=models_dir
                )
            
            # Move model to device after loading
            self.nllb_model = self.nllb_model.to(self.device)
            
            # Convert to float16 for faster inference if CUDA is available
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                logger.info("Converting model to float16 for faster inference...")
                self.nllb_model = self.nllb_model.half()
                logger.info("✓ Model converted to float16")
            
            self.nllb_model.eval()
            
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
    
    async def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts using NLLB."""
        if not texts:
            return []
        
        src = NLLB_LANG_MAP.get(self.source_lang, self.source_lang)
        tgt = NLLB_LANG_MAP.get(self.target_lang, self.target_lang)
        
        try:
            self.nllb_tokenizer.src_lang = src
            inputs = self.nllb_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            
            # Move all input tensors to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get the target language token ID
            forced_bos_token_id = self.nllb_tokenizer.convert_tokens_to_ids(tgt)
            
            # Use torch.no_grad() for inference to save memory
            with torch.no_grad():
                if isinstance(self.device, torch.device) and self.device.type == "cuda":
                    # Use autocast for mixed precision inference
                    with torch.amp.autocast('cuda'):
                        translated_tokens = self.nllb_model.generate(
                            **inputs,
                            forced_bos_token_id=forced_bos_token_id,
                            max_length=512
                        )
                else:
                    # CPU inference without autocast
                    translated_tokens = self.nllb_model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=512
                    )
            
            # Move tokens back to CPU for decoding
            translated_tokens = translated_tokens.cpu()
            return self.nllb_tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )
        except Exception as e:
            logger.error(f"NLLB batch translation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return texts  # Return original on failure

