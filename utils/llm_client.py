"""
COMPASS LLM Client

Wrapper for OpenRouter API with retry logic, token tracking, and logging.
"""

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

    def _can_fallback_to_openai(self, error: Exception) -> bool:
        if self.backend != LLMBackend.OPENROUTER:
            return False
        if not self.settings.openai_api_key:
            return False
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

    def _switch_to_openai_fallback(self, reason: str) -> None:
        role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
        self.backend = LLMBackend.OPENAI
        self.settings.models.backend = LLMBackend.OPENAI
        self.settings.models.public_model_name = (
            self._strip_provider_prefix(self.settings.models.public_model_name) or "gpt-5-nano"
        )
        for role in role_names:
            attr = f"{role}_model"
            current = getattr(self.settings.models, attr, "")
            setattr(
                self.settings.models,
                attr,
                self._strip_provider_prefix(current) or self.settings.models.public_model_name,
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
        max_tokens = max_tokens or self.settings.models.tool_max_tokens
        temperature = temperature or self.settings.models.tool_temperature
        
        # --- LOCAL BACKEND ROUTING ---
        if self.settings.models.backend == LLMBackend.LOCAL and self.local_llm:
            try:
                # Local LLM generate
                resp_data = self.local_llm.generate(
                    messages=messages,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature)
                )
                
                # Track usage
                self.token_tracker.add(
                    resp_data["prompt_tokens"], 
                    resp_data["completion_tokens"], 
                    resp_data["model"]
                )
                
                return LLMResponse(**resp_data)
            except Exception as e:
                logger.error(f"Local LLM call failed: {str(e)}")
                raise

        # --- Public API backends (OpenRouter/OpenAI) ---
        
        
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
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as first_error:
                if self._can_fallback_to_openai(first_error):
                    self._switch_to_openai_fallback(str(first_error))
                    fallback_model = self._strip_provider_prefix(model) or model
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs["model"] = fallback_model
                    print(f"[LLMClient] Retrying on OpenAI fallback using model {fallback_model}...")
                    response = self.client.chat.completions.create(**fallback_kwargs)
                    model = fallback_model
                else:
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
