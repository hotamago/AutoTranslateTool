"""Opus-MT translation service implementation."""

import os
import sys
import logging
from typing import List, Optional

import torch

from .base import BaseTranslator

logger = logging.getLogger(__name__)


class OpusMTTranslatorService(BaseTranslator):
    """Opus-MT translation service."""
    
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        concurrency: int = 10,
        api_key: Optional[str] = None,
        model_url: Optional[str] = None
    ):
        super().__init__(source_lang, target_lang, concurrency, api_key)
        self.model_url = model_url
        self.opus_model = None
        self.opus_tokenizer = None
        self.device = None
    
    async def initialize(self):
        """Initialize Opus-MT model."""
        if self.opus_model is not None:
            return
        
        if not self.model_url:
            logger.error(
                "opus-mt service requires --model_url parameter "
                "(e.g., Helsinki-NLP/opus-mt-ja-vi)"
            )
            sys.exit(1)
        
        logger.info(f"Loading opus-mt model: {self.model_url}")
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
            
            # Convert model URL to cache path format
            model_cache_name = self.model_url.replace("/", "--")
            model_cache_path = os.path.join(models_dir, f"models--{model_cache_name}")
            
            # Check if model is already cached
            model_cached = False
            if os.path.exists(model_cache_path):
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    for snapshot in os.listdir(snapshots_dir):
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        # Check for model files
                        model_files = [
                            f for f in os.listdir(snapshot_path)
                            if f.endswith(('.bin', '.safetensors'))
                        ]
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
                                logger.warning(
                                    f"Could not remove incomplete file {blob_file}: {e}"
                                )
            
            # Load model
            if model_cached:
                logger.info(f"Loading from cache: {models_dir}")
                self.opus_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_url, cache_dir=models_dir, local_files_only=True
                )
                self.opus_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_url, cache_dir=models_dir, local_files_only=True
                )
            else:
                logger.info(f"Downloading and caching model to: {models_dir}")
                self.opus_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_url, cache_dir=models_dir
                )
                self.opus_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_url, cache_dir=models_dir
                )
            
            # Move model to device after loading
            self.opus_model = self.opus_model.to(self.device)
            
            # Convert to float16 for faster inference if CUDA is available
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                logger.info("Converting model to float16 for faster inference...")
                self.opus_model = self.opus_model.half()
                logger.info("✓ Model converted to float16")
            
            self.opus_model.eval()
            
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
    
    async def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts using Opus-MT."""
        if not texts:
            return []
        
        try:
            inputs = self.opus_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            
            # Move all input tensors to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use torch.no_grad() for inference to save memory
            with torch.no_grad():
                if isinstance(self.device, torch.device) and self.device.type == "cuda":
                    # Use autocast for mixed precision inference
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
            return self.opus_tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )
        except Exception as e:
            logger.error(f"opus-mt batch translation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return texts  # Return original on failure

