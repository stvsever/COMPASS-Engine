"""
COMPASS Local LLM Handler

Handles integration with Open Source LLMs via vLLM (preferred) or HuggingFace Transformers (fallback).
Ensures safe sequential execution and efficient resource management.
"""

import logging
import threading
import time
from typing import List, Dict, Optional, Any
import warnings

# Attempt optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..config.settings import get_settings

logger = logging.getLogger("compass.local_llm")

class LocalLLMHandler:
    """
    Singleton handler for Local LLM inference.
    Manages loading the model once and handling generation requests.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.models.local_model_name
        self.backend_type = "unknown"
        self._lock = threading.Lock() # Crucial: Local inference is not thread-safe for VRAM
        
        self.llm_engine = None
        self.tokenizer = None
        self.hf_model = None
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model using vLLM if available, else Transformers."""
        logger.info(f"Initializing Local LLM: {self.model_name}")
        
        # 1. Try vLLM (Preferred for speed, but often CUDA only)
        if VLLM_AVAILABLE:
            try:
                logger.info(f"Attempting to load {self.model_name} with vLLM...")
                # vLLM is heavy, parameters usually need tuning for specific GPUs
                # We use defaults here but this might fail on Mac (MPS)
                self.llm_engine = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    dtype="auto"
                )
                self.backend_type = "vllm"
                logger.info("✓ vLLM loaded successfully")
                return
            except Exception as e:
                logger.warning(f"vLLM initialization failed: {e}. Falling back to Transformers.")
        
        # 2. Fallback to Transformers (Better Mac/CPU support)
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Attempting to load {self.model_name} with Transformers (MPS/CPU)...")
                
                # Determine device
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
                
                logger.info(f"Using device: {self.device}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    trust_remote_code=True,
                    device_map="auto" if self.device == "cuda" else None 
                )
                
                if self.device != "cuda": # "auto" handles cuda move
                    self.hf_model.to(self.device)
                
                self.backend_type = "transformers"
                logger.info("✓ Transformers loaded successfully")
                return
            except Exception as e:
                logger.error(f"Transformers initialization failed: {e}")
                raise RuntimeError(f"Could not load Local LLM {self.model_name}. Ensure `transformers` and `torch` are installed.")
        else:
            raise RuntimeError("Neither vLLM nor Transformers is available. Please install dependencies.")

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text completion.
        Returns interface compatible with OpenAI-like responses.
        """
        with self._lock: # Enforce sequential access
            start_time = time.time()
            prompt = self._apply_chat_template(messages)
            
            output_text = ""
            if self.backend_type == "vllm":
                output_text = self._generate_vllm(prompt, max_tokens, temperature)
            elif self.backend_type == "transformers":
                output_text = self._generate_transformers(prompt, max_tokens, temperature)
                
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Estimate tokens roughly (or use tokenizer if available)
            # This is an approximation for logging/budgeting
            est_tokens = len(output_text.split()) * 1.3 
            
            return {
                "content": output_text,
                "model": self.model_name,
                "prompt_tokens": len(prompt.split()) // 3, # Crude approx
                "completion_tokens": int(est_tokens),
                "total_tokens": int(est_tokens) + (len(prompt.split()) // 3),
                "finish_reason": "stop",
                "latency_ms": latency_ms
            }

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages list to a prompt string."""
        # Check if tokenizer has chat template
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                pass # Fallback to manual construction
        
        # Manual construction (standard Qwen/ChatML format)
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    def _generate_vllm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using vLLM."""
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|im_end|>"]
        )
        outputs = self.llm_engine.generate([prompt], params)
        return outputs[0].outputs[0].text

    def _generate_transformers(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Transformers."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Safety for very small models/devices
        if temperature < 0.01: temperature = 0.01
        
        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.1),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# Singleton
_local_llm_instance = None

def get_local_llm() -> LocalLLMHandler:
    global _local_llm_instance
    if _local_llm_instance is None:
        _local_llm_instance = LocalLLMHandler()
    return _local_llm_instance
