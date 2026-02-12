"""
COMPASS Local LLM Handler

Handles integration with Open Source LLMs via vLLM (preferred) or HuggingFace Transformers (fallback).
Ensures safe sequential execution and efficient resource management.

Notes:
- vLLM path is preferred on HPC (CUDA).
- Transformers fallback supports MPS/CPU/CUDA, with optional streaming for UI via TextIteratorStreamer.
"""

import logging
import threading
import time
from typing import List, Dict, Optional, Any, Iterator
import warnings

# Attempt optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

from ..config.settings import get_settings

logger = logging.getLogger("compass.local_llm")


class LocalLLMHandler:
    """
    Singleton handler for Local LLM inference.
    Manages loading the model once and handling generation requests.
    """
    device: str = "cpu"

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.models.local_model_name
        self.backend_type = "unknown"
        self._lock = threading.Lock()  # Local inference is not thread-safe for VRAM

        # Always define these so later code never AttributeErrors
        self.device: str = "cpu"  # "cuda" | "mps" | "cpu"
        self.llm_engine = None
        self.tokenizer = None
        self.hf_model = None

        self._initialize_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────────
    def _initialize_model(self):
        """Initialize the model using vLLM if available, else Transformers."""
        logger.info(f"Initializing Local LLM: {self.model_name}")

        backend_pref = (self.settings.models.local_backend_type or "auto").lower()

        # Try to load a tokenizer early (useful for correct chat templates even for vLLM).
        # If transformers isn't installed, we keep manual template fallback.
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=bool(self.settings.models.local_trust_remote_code),
                )
            except Exception as e:
                logger.warning(f"Tokenizer preload failed (will use manual template fallback): {e}")
                self.tokenizer = None

        # 1) Try vLLM (preferred)
        if VLLM_AVAILABLE and backend_pref in ["auto", "vllm"]:
            try:
                logger.info(f"Attempting to load {self.model_name} with vLLM...")
                print(f"[LocalLLM] vLLM init start: model={self.model_name} backend_pref={backend_pref}")

                dtype = (self.settings.models.local_dtype or "auto").lower()
                kv_cache_dtype = self.settings.models.local_kv_cache_dtype

                # Prefer FP8 KV cache; keep model dtype auto
                if dtype == "fp8":
                    dtype = "auto"
                    if not kv_cache_dtype:
                        # vLLM versions vary; "fp8" is commonly accepted.
                        kv_cache_dtype = "fp8"

                quant = (self.settings.models.local_quantization or "").strip().lower() or None
                max_len = self.settings.models.local_max_model_len or self.settings.models.local_max_tokens

                # Clamp max_len to tokenizer/config if possible (prevents 64k on 32k-native models)
                clamped_max_len = self._clamp_max_len(max_len)

                base_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "trust_remote_code": bool(self.settings.models.local_trust_remote_code),
                    "dtype": dtype,
                }

                if self.settings.models.local_tensor_parallel_size and self.settings.models.local_tensor_parallel_size > 1:
                    base_kwargs["tensor_parallel_size"] = int(self.settings.models.local_tensor_parallel_size)

                if self.settings.models.local_pipeline_parallel_size and self.settings.models.local_pipeline_parallel_size > 1:
                    base_kwargs["pipeline_parallel_size"] = int(self.settings.models.local_pipeline_parallel_size)

                if self.settings.models.local_gpu_memory_utilization:
                    base_kwargs["gpu_memory_utilization"] = float(self.settings.models.local_gpu_memory_utilization)

                if clamped_max_len and int(clamped_max_len) > 0:
                    base_kwargs["max_model_len"] = int(clamped_max_len)

                if self.settings.models.local_enforce_eager:
                    base_kwargs["enforce_eager"] = True

                # vLLM on HPC implies CUDA usage
                self.device = "cuda"

                # Build robust candidates:
                # - Prefer awq_marlin when user selected awq (faster on supported GPUs)
                # - Retry without kv_cache_dtype if a build rejects it
                quant_candidates: List[Optional[str]] = []
                if quant == "awq":
                    quant_candidates = ["awq_marlin", "awq", None]
                elif quant:
                    quant_candidates = [quant, None]
                else:
                    quant_candidates = [None]

                kv_candidates: List[Optional[str]] = []
                if kv_cache_dtype:
                    norm_kv = self._normalize_kv_cache_dtype(str(kv_cache_dtype))
                    kv_candidates = [norm_kv, "auto", None]
                else:
                    kv_candidates = [None]

                attempts: List[Dict[str, Any]] = []
                seen: set[tuple] = set()
                for q in quant_candidates:
                    for kv in kv_candidates:
                        cand = dict(base_kwargs)
                        if q:
                            cand["quantization"] = q
                        if kv:
                            cand["kv_cache_dtype"] = kv
                        key = tuple(sorted((k, str(v)) for k, v in cand.items()))
                        if key in seen:
                            continue
                        seen.add(key)
                        attempts.append(cand)

                last_exc: Optional[Exception] = None
                for idx, cand in enumerate(attempts, 1):
                    label = (
                        f"quant={cand.get('quantization', 'auto')}, "
                        f"kv_cache_dtype={cand.get('kv_cache_dtype', 'default')}"
                    )
                    print(f"[LocalLLM] vLLM init attempt {idx}/{len(attempts)} ({label})")
                    logger.info("vLLM init attempt %s/%s with %s", idx, len(attempts), label)
                    try:
                        self.llm_engine = LLM(**cand)
                        break
                    except Exception as e:
                        last_exc = e
                        logger.warning("vLLM init attempt %s failed: %s", idx, e)
                        print(f"[LocalLLM] vLLM init attempt {idx} failed: {e}")
                if self.llm_engine is None:
                    if last_exc is not None:
                        raise last_exc
                    raise RuntimeError("vLLM initialization failed after all fallback attempts.")

                self.backend_type = "vllm"
                logger.info("✓ vLLM loaded successfully")
                print("[LocalLLM] ✓ vLLM loaded successfully")
                return

            except Exception as e:
                import traceback
                logger.error(f"vLLM initialization detailed error: {str(e)}")
                logger.error(traceback.format_exc())
                print(f"[LocalLLM] ✗ vLLM init failed: {e}")
                logger.warning("vLLM initialization failed. Falling back to Transformers (if allowed).")
                backend_pref = "auto"

        elif backend_pref == "vllm" and not VLLM_AVAILABLE:
            raise RuntimeError("vLLM requested but not installed. Install vLLM or switch to transformers.")

        # 2) Fallback to Transformers
        if TRANSFORMERS_AVAILABLE and backend_pref in ["auto", "transformers"]:
            try:
                logger.info(f"Attempting to load {self.model_name} with Transformers...")
                print(f"[LocalLLM] Transformers init start: model={self.model_name}")

                # Determine device
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"

                logger.info(f"Using device: {self.device}")

                # Ensure tokenizer exists
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=bool(self.settings.models.local_trust_remote_code),
                    )

                # Dtype selection
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
                else:
                    torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

                # Quantization handling
                quant = (self.settings.models.local_quantization or "").lower()
                quant_kwargs: Dict[str, Any] = {}

                # bitsandbytes-style quantization
                if quant in ["4bit", "int4", "bnb_4bit"]:
                    quant_kwargs["load_in_4bit"] = True
                elif quant in ["8bit", "int8", "bnb_8bit"]:
                    quant_kwargs["load_in_8bit"] = True
                elif quant in ["awq", "gptq"]:
                    # Important: Transformers fallback for AWQ is not generally safe unless you have a supported path.
                    # Keep flexibility: allow it only if user explicitly requests it.
                    allow_awq_fallback = bool(getattr(self.settings.models, "allow_awq_transformers_fallback", False))
                    if not allow_awq_fallback:
                        raise RuntimeError(
                            f"Quantization '{quant}' is not supported in Transformers fallback by default. "
                            f"Use vLLM for {quant} models (or set settings.models.allow_awq_transformers_fallback=True "
                            f"if you have a validated Transformers AWQ path)."
                        )
                    logger.warning(f"Proceeding with Transformers fallback for '{quant}' because allow_awq_transformers_fallback=True.")

                # Attention implementation override
                attn_impl = (self.settings.models.local_attn_implementation or "auto").lower()
                attn_kwargs: Dict[str, Any] = {}
                if attn_impl != "auto":
                    attn_kwargs["attn_implementation"] = attn_impl

                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=bool(self.settings.models.local_trust_remote_code),
                    device_map="auto" if self.device == "cuda" else None,
                    **quant_kwargs,
                    **attn_kwargs,
                )

                if self.device != "cuda":  # device_map="auto" handles CUDA move
                    self.hf_model.to(self.device)

                self.backend_type = "transformers"
                logger.info("✓ Transformers loaded successfully")
                print("[LocalLLM] ✓ Transformers loaded successfully")
                return

            except Exception as e:
                import traceback
                logger.error(f"Transformers initialization failed: {e}")
                logger.error(traceback.format_exc())
                print(f"[LocalLLM] ✗ Transformers init failed: {e}")
                raise RuntimeError(
                    f"Could not load Local LLM {self.model_name}. "
                    f"See logs above for the full traceback."
                )

        elif backend_pref == "transformers" and not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers requested but not installed. Install transformers/torch or switch to vLLM.")

        raise RuntimeError("Neither vLLM nor Transformers is available. Please install dependencies.")

    # ──────────────────────────────────────────────────────────────────────────
    # Public APIs
    # ──────────────────────────────────────────────────────────────────────────
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text completion (non-streaming).
        Returns interface compatible with OpenAI-like responses.
        """
        with self._lock:
            # Aggressive memory cleanup before generation (only if torch exists & CUDA)
            if self.backend_type == "transformers" and TRANSFORMERS_AVAILABLE and self.device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            start_time = time.time()
            prompt = self._apply_chat_template(messages)

            if self.backend_type == "vllm":
                output_text = self._generate_vllm(prompt, max_tokens, temperature)
            elif self.backend_type == "transformers":
                output_text = self._generate_transformers(prompt, max_tokens, temperature)
            else:
                raise RuntimeError(f"Unknown backend_type: {self.backend_type}")

            latency_ms = int((time.time() - start_time) * 1000)

            # Better token counting when tokenizer is available
            prompt_tokens = self._count_tokens(prompt) if self.tokenizer else (len(prompt.split()) // 3)
            completion_tokens = self._count_tokens(output_text) if self.tokenizer else int(len(output_text.split()) * 1.3)

            return {
                "content": output_text,
                "model": self.model_name,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens) + int(completion_tokens),
                "finish_reason": "stop",
                "latency_ms": latency_ms,
            }

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Iterator[str]:
        """
        Streaming generation for UI mode.

        - Transformers: true streaming using TextIteratorStreamer.
        - vLLM: falls back to non-streaming (returns one chunk) unless you later move to AsyncLLMEngine.
        """
        with self._lock:
            prompt = self._apply_chat_template(messages)

            if self.backend_type == "transformers":
                if not TRANSFORMERS_AVAILABLE:
                    raise RuntimeError("Transformers backend selected but transformers/torch are not available.")
                yield from self._generate_transformers_stream(prompt, max_tokens, temperature)
                return

            if self.backend_type == "vllm":
                # vLLM python API LLM.generate isn't a true token streamer.
                # Keep UI compatible: yield once.
                text = self._generate_vllm(prompt, max_tokens, temperature)
                yield text
                return

            raise RuntimeError(f"Unknown backend_type: {self.backend_type}")

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt formatting
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages list to a prompt string."""
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass  # fallback below

        # Manual construction (ChatML-like)
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            else:
                # unknown role: treat as user
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    # ──────────────────────────────────────────────────────────────────────────
    # vLLM backend
    # ──────────────────────────────────────────────────────────────────────────
    def _generate_vllm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using vLLM."""
        if not self.llm_engine:
            raise RuntimeError("vLLM backend requested but llm_engine is not initialized.")

        # Allow deterministic greedy decoding for strict JSON calls.
        if temperature <= 0:
            temperature = 0.0
        elif temperature < 0.01:
            temperature = 0.01

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["<|im_end|>"]
        )
        outputs = self.llm_engine.generate([prompt], params)
        return outputs[0].outputs[0].text

    # ──────────────────────────────────────────────────────────────────────────
    # Transformers backend (non-stream + stream)
    # ──────────────────────────────────────────────────────────────────────────
    def _generate_transformers(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Transformers."""
        if not (self.hf_model and self.tokenizer):
            raise RuntimeError("Transformers backend requested but model/tokenizer is not initialized.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Safety for small temps
        if temperature <= 0:
            temperature = 0.0
        elif temperature < 0.01:
            temperature = 0.01

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

    def _generate_transformers_stream(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
        """Streaming generation using Transformers + TextIteratorStreamer."""
        if not (self.hf_model and self.tokenizer):
            raise RuntimeError("Transformers backend requested but model/tokenizer is not initialized.")

        # Safety for small temps
        if temperature <= 0:
            temperature = 0.0
        elif temperature < 0.01:
            temperature = 0.01

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.1),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run generation in a background thread while yielding tokens
        t = threading.Thread(target=self.hf_model.generate, kwargs=gen_kwargs, daemon=True)
        t.start()

        for token_text in streamer:
            yield token_text

        t.join(timeout=0.1)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _count_tokens(self, text: str) -> int:
        try:
            return int(len(self.tokenizer(text, add_special_tokens=False).input_ids))  # type: ignore
        except Exception:
            return int(len(text.split()) // 3)

    def _clamp_max_len(self, requested: Optional[int]) -> Optional[int]:
        """
        Clamp requested max length against tokenizer/model config when possible.
        Prevents common vLLM init failures when user requests > native context.
        """
        if not requested:
            return None

        req = int(requested)
        if req <= 0:
            return None

        detected: Optional[int] = None

        # Tokenizer-based detection
        if self.tokenizer is not None:
            try:
                tmax = getattr(self.tokenizer, "model_max_length", None)
                if isinstance(tmax, int) and tmax > 0 and tmax < 10**9:
                    detected = tmax
            except Exception:
                pass

        # Config-based detection (if tokenizer didn’t provide)
        if TRANSFORMERS_AVAILABLE and detected is None:
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=bool(self.settings.models.local_trust_remote_code))
                for k in ("max_position_embeddings", "max_sequence_length", "max_seq_len", "seq_length"):
                    v = getattr(cfg, k, None)
                    if isinstance(v, int) and v > 0:
                        detected = max(detected or 0, v)
            except Exception:
                pass

        # If we detected something meaningful and the request exceeds it, clamp.
        if detected and detected > 0 and req > detected:
            logger.warning(f"Requested max_len={req} exceeds detected limit={detected}. Clamping to {detected}.")
            return detected

        return req

    def _normalize_kv_cache_dtype(self, v: str) -> str:
        """
        Normalize kv_cache dtype strings across common variants.
        vLLM versions differ; keep minimal normalization.
        """
        vv = v.strip().lower()
        aliases = {
            "fp8_e4m3": "fp8",
            "fp8-e4m3": "fp8",
            "fp8e4m3": "fp8",
            "e4m3": "fp8",
        }
        return aliases.get(vv, v)


# Singleton
_local_llm_instance: Optional[LocalLLMHandler] = None


def get_local_llm() -> LocalLLMHandler:
    global _local_llm_instance
    if _local_llm_instance is None:
        _local_llm_instance = LocalLLMHandler()
    return _local_llm_instance


def reset_local_llm():
    global _local_llm_instance
    _local_llm_instance = None
