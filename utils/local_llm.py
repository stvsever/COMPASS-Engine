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
        backend_pref = (self.settings.models.local_backend_type or "auto").lower()
        if VLLM_AVAILABLE and backend_pref in ["auto", "vllm"]:
            try:
                logger.info(f"Attempting to load {self.model_name} with vLLM...")
                # vLLM is heavy, parameters usually need tuning for specific GPUs
                # We use defaults here but this might fail on Mac (MPS)
                dtype = (self.settings.models.local_dtype or "auto").lower()
                kv_cache_dtype = self.settings.models.local_kv_cache_dtype
                if dtype == "fp8":
                    # vLLM does not always accept dtype=fp8; prefer FP8 KV cache.
                    dtype = "auto"
                    if not kv_cache_dtype:
                        kv_cache_dtype = "fp8_e4m3"
                quant = self.settings.models.local_quantization
                max_len = self.settings.models.local_max_model_len or self.settings.models.local_max_tokens
                llm_kwargs = {
                    "model": self.model_name,
                    "trust_remote_code": bool(self.settings.models.local_trust_remote_code),
                    "dtype": dtype,
                }
                if quant:
                    llm_kwargs["quantization"] = str(quant)
                if self.settings.models.local_tensor_parallel_size and self.settings.models.local_tensor_parallel_size > 1:
                    llm_kwargs["tensor_parallel_size"] = int(self.settings.models.local_tensor_parallel_size)
                if self.settings.models.local_pipeline_parallel_size and self.settings.models.local_pipeline_parallel_size > 1:
                    llm_kwargs["pipeline_parallel_size"] = int(self.settings.models.local_pipeline_parallel_size)
                if self.settings.models.local_gpu_memory_utilization:
                    llm_kwargs["gpu_memory_utilization"] = float(self.settings.models.local_gpu_memory_utilization)
                if max_len and int(max_len) > 0:
                    llm_kwargs["max_model_len"] = int(max_len)
                if kv_cache_dtype:
                    llm_kwargs["kv_cache_dtype"] = str(kv_cache_dtype)
                if self.settings.models.local_enforce_eager:
                    llm_kwargs["enforce_eager"] = True

                self.llm_engine = LLM(**llm_kwargs)
                self.backend_type = "vllm"
                logger.info("✓ vLLM loaded successfully")
                return
            except Exception as e:
                logger.warning(f"vLLM initialization failed: {e}. Falling back to Transformers.")
        elif backend_pref == "vllm" and not VLLM_AVAILABLE:
            raise RuntimeError("vLLM requested but not installed. Install vLLM or switch to transformers.")
        
        # 2. Fallback to Transformers (Better Mac/CPU support)
        if TRANSFORMERS_AVAILABLE and backend_pref in ["auto", "transformers"]:
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
                    trust_remote_code=bool(self.settings.models.local_trust_remote_code)
                )

                torch_dtype = None
                dtype = (self.settings.models.local_dtype or "auto").lower()
                if dtype in ["float16", "fp16", "half"]:
                    torch_dtype = torch.float16
                elif dtype in ["bfloat16", "bf16"]:
                    torch_dtype = torch.bfloat16
                elif dtype in ["float32", "fp32"]:
                    torch_dtype = torch.float32
                elif dtype == "fp8":
                    logger.warning("FP8 dtype is not supported in Transformers; falling back to float16.")
                    torch_dtype = torch.float16

                quant = (self.settings.models.local_quantization or "").lower()
                quant_kwargs = {}
                if quant in ["4bit", "int4", "bnb_4bit"]:
                    quant_kwargs["load_in_4bit"] = True
                elif quant in ["8bit", "int8", "bnb_8bit"]:
                    quant_kwargs["load_in_8bit"] = True
                elif quant in ["awq", "gptq", "fp8"]:
                    logger.warning(f"Quantization '{quant}' is not supported in Transformers path; use vLLM.")

                attn_impl = (self.settings.models.local_attn_implementation or "auto").lower()
                attn_kwargs = {}
                if attn_impl != "auto":
                    attn_kwargs["attn_implementation"] = attn_impl
                
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype or (torch.float16 if self.device != "cpu" else torch.float32),
                    trust_remote_code=bool(self.settings.models.local_trust_remote_code),
                    device_map="auto" if self.device == "cuda" else None,
                    **quant_kwargs,
                    **attn_kwargs,
                )
                
                if self.device != "cuda": # "auto" handles cuda move
                    self.hf_model.to(self.device)
                
                self.backend_type = "transformers"
                logger.info("✓ Transformers loaded successfully")
                return
            except Exception as e:
                logger.error(f"Transformers initialization failed: {e}")
                raise RuntimeError(f"Could not load Local LLM {self.model_name}. Ensure `transformers` and `torch` are installed.")
        elif backend_pref == "transformers" and not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers requested but not installed. Install transformers/torch or switch to vLLM.")
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
