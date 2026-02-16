"""
COMPASS LLM Client

Wrapper for OpenRouter API with retry logic, token tracking, and logging.
"""

import threading
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from ..config.settings import get_settings, LLMBackend
except ImportError:
    from config.settings import get_settings, LLMBackend

logger = logging.getLogger("compass.llm_client")


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    latency_ms: int
    
    @property
    def successful(self) -> bool:
        return self.finish_reason == "stop"


@dataclass
class TokenTracker:
    """Tracks token usage across calls."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    calls: List[Dict[str, int]] = field(default_factory=list)
    
    def add(self, prompt: int, completion: int, model: str):
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.calls.append({
            "model": model,
            "prompt": prompt,
            "completion": completion,
            "timestamp": time.time()
        })
    
    @property
    def total(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens
    
    def summary(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "call_count": len(self.calls)
        }


class LLMClient:
    """
    OpenAI API client for COMPASS.
    
    Handles all LLM interactions with:
    - Automatic retries with exponential backoff
    - Token usage tracking
    - Detailed logging
    - Error handling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.settings = settings
        self.token_tracker = TokenTracker()
        self.backend = self.settings.models.backend
        self.api_key = api_key or settings.openai_api_key
        
        self.local_llm = None
        self.client = None
        self.embedding_client = None

        if self.backend == LLMBackend.LOCAL:
            from .local_llm import get_local_llm
            self.local_llm = get_local_llm()
            logger.info("LLM Client initialized (LOCAL Backend)")
        elif self.backend == LLMBackend.OPENROUTER:
            if not self.settings.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not provided for OpenRouter backend")
            self.client = self._build_openrouter_client()
            self.embedding_client = self.client
            logger.info("LLM Client initialized (OpenRouter Backend)")
        elif self.backend == LLMBackend.OPENAI:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not provided for OpenAI backend")
            self.client = OpenAI(api_key=self.settings.openai_api_key)
            self.embedding_client = self.client
            logger.info("LLM Client initialized (OpenAI Backend)")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _build_openrouter_client(self) -> OpenAI:
        headers: Dict[str, str] = {}
        referer = (self.settings.openrouter_site_url or "").strip()
        title = (self.settings.openrouter_app_name or "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        kwargs: Dict[str, Any] = {
            "api_key": self.settings.openrouter_api_key,
            "base_url": self.settings.openrouter_base_url,
        }
        if headers:
            kwargs["default_headers"] = headers
        return OpenAI(**kwargs)

    def _provider_label(self) -> str:
        if self.backend == LLMBackend.OPENROUTER:
            return "OpenRouter"
        if self.backend == LLMBackend.OPENAI:
            return "OpenAI"
        if self.backend == LLMBackend.LOCAL:
            return "Local"
        return str(self.backend)

    def _resolve_model_name(self, model: str) -> str:
        resolved = str(model or "").strip()
        if self.backend == LLMBackend.OPENROUTER and resolved and "/" not in resolved:
            if resolved.startswith(("gpt-", "o1", "o3", "o4", "text-embedding-")):
                return f"openai/{resolved}"
        return resolved

    @staticmethod
    def _strip_provider_prefix(model: str) -> str:
        text = str(model or "").strip()
        if not text:
            return ""
        if "/" in text:
            return text.split("/", 1)[1].strip()
        return text

    @staticmethod
    def _is_network_transport_error(error: Exception) -> bool:
        msg = str(error).lower()
        network_markers = (
            "ssl",
            "certificate",
            "timed out",
            "timeout",
            "name or service not known",
            "temporary failure",
            "connection error",
            "max retries exceeded",
        )
        return any(marker in msg for marker in network_markers)

    @staticmethod
    def _is_openai_model_name(model: str) -> bool:
        name = str(model or "").strip().lower()
        if not name:
            return False
        return name.startswith(("gpt-", "o1", "o3", "o4", "text-embedding-"))

    def _to_openai_model_name(self, model: str) -> str:
        text = str(model or "").strip()
        if not text:
            return ""
        if "/" in text:
            provider, name = text.split("/", 1)
            if provider.strip().lower() != "openai":
                return ""
            candidate = name.strip()
            return candidate if self._is_openai_model_name(candidate) else ""
        return text if self._is_openai_model_name(text) else ""

    def _can_fallback_to_openai(self, error: Exception, requested_model: str = "") -> bool:
        if self.backend != LLMBackend.OPENROUTER:
            return False
        if not self.settings.openai_api_key:
            return False
        if not self._is_network_transport_error(error):
            return False
        candidate = self._to_openai_model_name(requested_model or self.settings.models.public_model_name)
        return bool(candidate)

    @staticmethod
    def _is_transient_public_api_error(error: Exception) -> bool:
        msg = str(error).lower()
        transient_markers = (
            "ssl",
            "certificate",
            "timed out",
            "timeout",
            "name or service not known",
            "temporary failure",
            "connection error",
            "max retries exceeded",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "error code: 504",
            "internal server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "failed to authenticate request with clerk",
            "clerk",
        )
        return any(marker in msg for marker in transient_markers)

    def _openrouter_model_fallback_candidate(self, model: str, error: Exception) -> str:
        if self.backend != LLMBackend.OPENROUTER:
            return ""
        if not self._is_transient_public_api_error(error):
            return ""
        requested = self._resolve_model_name(model)
        # If an OpenRouter-routed provider model fails transiently (e.g., x-ai auth proxy),
        # retry once with a broadly available OpenRouter model.
        if requested.startswith("openai/"):
            return ""
        fallback_model = "openai/gpt-5-nano"
        if requested == fallback_model:
            return ""
        return fallback_model

    def _switch_to_openai_fallback(self, reason: str) -> None:
        role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
        self.backend = LLMBackend.OPENAI
        self.settings.models.backend = LLMBackend.OPENAI
        public_default = self._to_openai_model_name(self.settings.models.public_model_name) or "gpt-5-nano"
        self.settings.models.public_model_name = public_default
        for role in role_names:
            attr = f"{role}_model"
            current = getattr(self.settings.models, attr, "")
            setattr(
                self.settings.models,
                attr,
                self._to_openai_model_name(current) or public_default,
            )
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.embedding_client = self.client
        logger.warning("OpenRouter transport failure. Falling back to OpenAI backend: %s", reason)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def call(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict] = None,
    ) -> LLMResponse:
        """
        Make an LLM API call.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to tool model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            response_format: Optional response format (e.g., {"type": "json_object"})
        
        Returns:
            LLMResponse with content and metadata
        """
        model = model or self.settings.models.tool_model
        model = self._resolve_model_name(model)
        if max_tokens is None:
            max_tokens = self.settings.models.tool_max_tokens
        if temperature is None:
            temperature = self.settings.models.tool_temperature
        
        # --- LOCAL BACKEND ROUTING ---
        backend_value = getattr(self.settings.models.backend, "value", self.settings.models.backend)
        backend_is_local = (
            self.settings.models.backend == LLMBackend.LOCAL
            or self.backend == LLMBackend.LOCAL
            or str(backend_value).lower() == "local"
        )
        if self.local_llm and backend_is_local:
            try:
                if not hasattr(self.local_llm, "device"):
                    setattr(self.local_llm, "device", "cpu")
                    logger.warning("LocalLLMHandler missing 'device' attribute; defaulting to 'cpu'.")
                json_mode = (
                    isinstance(response_format, dict)
                    and str(response_format.get("type", "")).lower() == "json_object"
                )
                local_messages = list(messages or [])
                local_temperature = float(temperature)
                if json_mode:
                    local_temperature = min(local_temperature, 0.15)
                    local_temperature = max(local_temperature, 0.0)
                    json_contract = (
                        "\n\nSTRICT JSON OUTPUT:\n"
                        "- Return ONLY one valid JSON object.\n"
                        "- No markdown fences.\n"
                        "- No commentary, analysis, or <think> blocks.\n"
                        "- Start with '{' and end with '}'."
                    )
                    if local_messages:
                        last = dict(local_messages[-1])
                        if str(last.get("role", "")) == "user":
                            last["content"] = f"{last.get('content', '')}{json_contract}"
                            local_messages[-1] = last
                        else:
                            local_messages.append({"role": "user", "content": json_contract.strip()})
                    else:
                        local_messages = [{"role": "user", "content": json_contract.strip()}]

                local_call_start = time.time()
                print(
                    f"[LLMClient][LOCAL] generate start "
                    f"model={self.settings.models.local_model_name} "
                    f"messages={len(local_messages)} max_tokens={int(max_tokens)} "
                    f"temp={float(local_temperature):.2f} json_mode={json_mode}"
                )
                # Run local generation in a thread so we can emit heartbeats on long calls.
                resp_data: Dict[str, Any] = {}
                local_error: List[Exception] = []

                def _run_local_generate() -> None:
                    try:
                        result = self.local_llm.generate(
                            messages=local_messages,
                            max_tokens=int(max_tokens),
                            temperature=float(local_temperature),
                        )
                        resp_data.update(result or {})
                    except Exception as exc:
                        local_error.append(exc)

                worker = threading.Thread(target=_run_local_generate, daemon=True)
                worker.start()
                while worker.is_alive():
                    worker.join(timeout=30)
                    if worker.is_alive():
                        waited_s = int(time.time() - local_call_start)
                        print(
                            f"[LLMClient][LOCAL] still generating... waited={waited_s}s "
                            f"model={self.settings.models.local_model_name}"
                        )
                if local_error:
                    raise local_error[0]
                
                # Track usage
                self.token_tracker.add(
                    resp_data["prompt_tokens"], 
                    resp_data["completion_tokens"], 
                    resp_data["model"]
                )
                print(
                    f"[LLMClient][LOCAL] generate done "
                    f"latency_ms={int(resp_data.get('latency_ms', 0))} "
                    f"prompt_tokens={int(resp_data.get('prompt_tokens', 0))} "
                    f"completion_tokens={int(resp_data.get('completion_tokens', 0))}"
                )
                
                return LLMResponse(
                    content=str(resp_data.get("content", "")),
                    model=str(resp_data.get("model", self.settings.models.local_model_name)),
                    prompt_tokens=int(resp_data.get("prompt_tokens", 0)),
                    completion_tokens=int(resp_data.get("completion_tokens", 0)),
                    total_tokens=int(
                        resp_data.get(
                            "total_tokens",
                            int(resp_data.get("prompt_tokens", 0)) + int(resp_data.get("completion_tokens", 0)),
                        )
                    ),
                    finish_reason=str(resp_data.get("finish_reason", "stop")),
                    latency_ms=int(resp_data.get("latency_ms", 0)),
                )
            except Exception as e:
                logger.error(f"Local LLM call failed: {str(e)}")
                logger.exception("Local LLM traceback")
                raise

        # --- Public API backends (OpenRouter/OpenAI) ---
        if self.client is None:
            raise RuntimeError(
                f"Public API client is not initialized (backend={self.settings.models.backend}). "
                "If you intended local inference, ensure backend is LOCAL."
            )
        
        
        # Log call details for debugging
        total_prompt_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"LLM Call: model={model}, messages={len(messages)}, prompt_chars={total_prompt_chars}")
        
        start_time = time.time()
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": int(max_tokens),
            }
            if not str(model).lower().startswith("gpt-5"):
                kwargs["temperature"] = float(temperature)
            
            if response_format:
                kwargs["response_format"] = response_format
            
            print(f"[LLMClient] Sending request to {model} (max_completion_tokens={max_tokens})...")
            response = None
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as first_error:
                last_error = first_error
                retried = False

                openrouter_fallback_model = self._openrouter_model_fallback_candidate(model, first_error)
                if openrouter_fallback_model:
                    retried = True
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs["model"] = openrouter_fallback_model
                    if str(openrouter_fallback_model).lower().startswith(("gpt-5", "openai/gpt-5")):
                        fallback_kwargs.pop("temperature", None)
                    print(
                        f"[LLMClient] OpenRouter request failed; retrying with fallback model "
                        f"{openrouter_fallback_model}..."
                    )
                    try:
                        response = self.client.chat.completions.create(**fallback_kwargs)
                        model = openrouter_fallback_model
                    except Exception as openrouter_fallback_error:
                        last_error = openrouter_fallback_error

                if response is None and self._can_fallback_to_openai(last_error, requested_model=model):
                    retried = True
                    self._switch_to_openai_fallback(str(last_error))
                    fallback_model = (
                        self._to_openai_model_name(model)
                        or self._to_openai_model_name(self.settings.models.public_model_name)
                        or "gpt-5-nano"
                    )
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs["model"] = fallback_model
                    if str(fallback_model).lower().startswith("gpt-5"):
                        fallback_kwargs.pop("temperature", None)
                    print(f"[LLMClient] Retrying on OpenAI fallback using model {fallback_model}...")
                    response = self.client.chat.completions.create(**fallback_kwargs)
                    model = fallback_model

                if response is None:
                    if retried:
                        raise last_error
                    raise
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract content - handle None case
            content = response.choices[0].message.content if response.choices else None
            
            # Check for empty response and raise explicit error
            if not content or not content.strip():
                finish_reason = response.choices[0].finish_reason if response.choices else "unknown"
                logger.error(f"Empty response from {model}. Finish reason: {finish_reason}")
                raise ValueError(f"Model {model} returned empty response (finish_reason={finish_reason})")
            
            # Extract usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            # Track usage
            self.token_tracker.add(prompt_tokens, completion_tokens, model)
            
            result = LLMResponse(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency_ms
            )
            
            logger.info(
                f"LLM call completed: {model}, "
                f"{result.total_tokens} tokens, {latency_ms}ms, {len(content)} chars"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def call_orchestrator(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Call with orchestrator model (GPT-5)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.call(
            messages=messages,
            model=self.settings.models.orchestrator_model,
            max_tokens=self.settings.models.orchestrator_max_tokens,
            temperature=self.settings.models.orchestrator_temperature,
            response_format={"type": "json_object"},
            **kwargs
        )
    
    def call_critic(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Call with critic model (GPT-5)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.call(
            messages=messages,
            model=self.settings.models.critic_model,
            max_tokens=self.settings.models.critic_max_tokens,
            temperature=self.settings.models.critic_temperature,
            response_format={"type": "json_object"},
            **kwargs
        )
    
    def call_predictor(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Call with predictor model (GPT-5)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.call(
            messages=messages,
            model=self.settings.models.predictor_model,
            max_tokens=self.settings.models.predictor_max_tokens,
            temperature=self.settings.models.predictor_temperature,
            response_format={"type": "json_object"},
            **kwargs
        )
    
    def call_tool(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Call with tool model (GPT-5-nano)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.call(
            messages=messages,
            model=self.settings.models.tool_model,
            max_tokens=self.settings.models.tool_max_tokens,
            temperature=self.settings.models.tool_temperature,
            response_format={"type": "json_object"},
            **kwargs
        )
    
    def ping(self, model: Optional[str] = None, timeout_s: int = 10) -> bool:
        """
        Lightweight connectivity check for configured backend.
        """
        if self.backend == LLMBackend.LOCAL:
            return True
        if not self.client:
            if self.backend == LLMBackend.OPENROUTER:
                self.client = self._build_openrouter_client()
            elif self.backend == LLMBackend.OPENAI:
                self.client = OpenAI(api_key=self.settings.openai_api_key)
            else:
                raise ValueError(f"Unsupported backend during ping: {self.backend}")
        try:
            self.client.models.list()
            return True
        except Exception:
            model = model or self.settings.models.tool_model
            model = self._resolve_model_name(model)
            try:
                for max_tokens in (128, 256, 512):
                    try:
                        self.client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "ping"}],
                            max_completion_tokens=max_tokens,
                            timeout=timeout_s,
                        )
                        return True
                    except Exception as inner:
                        if max_tokens == 512:
                            raise inner
            except Exception as inner:
                provider = self._provider_label()
                logger.error(f"{provider} connectivity check failed: {str(inner)}")
                raise RuntimeError(f"{provider} connectivity check failed: {inner}") from inner

    def get_token_usage(self) -> Dict[str, Any]:
        """Get current token usage summary."""
        return self.token_tracker.summary()
    
    def reset_token_tracker(self):
        """Reset token tracking for new participant."""
        self.token_tracker = TokenTracker()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _embedding_backend(self) -> str:
        if self.backend == LLMBackend.LOCAL:
            return "local"
        if self.backend == LLMBackend.OPENROUTER:
            return "openrouter"
        if self.backend == LLMBackend.OPENAI:
            return "openai"
        if self.settings.openrouter_api_key:
            return "openrouter"
        if self.settings.openai_api_key:
            return "openai"
        raise ValueError("No embedding provider configured (OPENROUTER_API_KEY or OPENAI_API_KEY required)")

    def _ensure_embedding_client(self, backend: str) -> OpenAI:
        if backend == "openrouter":
            if not self.settings.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not provided for embeddings")
            if self.embedding_client is None or self.backend != LLMBackend.OPENROUTER:
                self.embedding_client = self._build_openrouter_client()
            return self.embedding_client
        if backend == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not provided for embeddings")
            if self.embedding_client is None or self.backend != LLMBackend.OPENAI:
                self.embedding_client = OpenAI(api_key=self.settings.openai_api_key)
            return self.embedding_client
        raise ValueError(f"Unsupported embedding backend: {backend}")

    def get_embedding(self, text: str, model: str = "") -> List[float]:
        """
        Get embedding for text using configured provider.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            if len(text) > 30000:
                text = text[:30000]
            
            backend = self._embedding_backend()
            
            # --- LOCAL EMBEDDINGS (No API) ---
            if backend == "local":
                from sentence_transformers import SentenceTransformer
                
                # Determine model path
                embed_model_name = (model or self.settings.models.embedding_model or "").strip()
                if not embed_model_name:
                    embed_model_name = "BAAI/bge-large-en-v1.5" # Default high-quality local model
                
                # Clean up if it's a full path
                # If running on HPC, we might want to point to the specific downloaded model dir
                # For now, we rely on the user having downloaded it or it being cached in HF_HOME
                
                # Log first load
                if not getattr(self, "_local_embed_model", None):
                    logger.info(f"Loading local embedding model to CPU: {embed_model_name}")
                    # Offload to CPU (16GB) to save all VRAM for gpt-oss-20b (~21GB FP8)
                    # Node has 1TB RAM, so this is very safe.
                    self._local_embed_model = SentenceTransformer(embed_model_name, device="cpu")
                
                # Generate
                embedding = self._local_embed_model.encode(text, convert_to_numpy=True).tolist()
                return embedding

            # --- API EMBEDDINGS ---
            client = self._ensure_embedding_client(backend)
            embed_model = (model or self.settings.models.embedding_model or "").strip()
            if not embed_model:
                embed_model = "text-embedding-3-large"
            embed_model = self._resolve_model_name(embed_model)
            response = client.embeddings.create(input=text, model=embed_model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise


# Singleton instance
_llm_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    global _llm_client_instance
    desired_backend = get_settings().models.backend
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    else:
        if getattr(_llm_client_instance, "backend", None) != desired_backend:
            _llm_client_instance = LLMClient()
        elif desired_backend == LLMBackend.LOCAL and not _llm_client_instance.local_llm:
            _llm_client_instance = LLMClient()
        elif desired_backend in (LLMBackend.OPENAI, LLMBackend.OPENROUTER) and not _llm_client_instance.client:
            _llm_client_instance = LLMClient()
    return _llm_client_instance


def reset_llm_client() -> None:
    global _llm_client_instance
    _llm_client_instance = None
    try:
        from .local_llm import reset_local_llm
        reset_local_llm()
    except Exception:
        pass
