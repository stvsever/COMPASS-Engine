"""
COMPASS LLM Client

Wrapper for OpenAI API with retry logic, token tracking, and logging.
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
        self.api_key = api_key or settings.openai_api_key
        
        self.settings = settings
        self.token_tracker = TokenTracker()
        
        self.local_llm = None
        if self.settings.models.backend == LLMBackend.LOCAL:
            from .local_llm import get_local_llm
            self.local_llm = get_local_llm()
            logger.info("LLM Client initialized (LOCAL Backend)")
        else:
            if not self.api_key:
                raise ValueError("OpenAI API key not provided for OpenAI Backend")
            self.client = OpenAI(api_key=self.api_key)
            logger.info("LLM Client initialized (OpenAI Backend)")
    
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

        # --- OPENAI BACKEND ---
        
        
        # Log call details for debugging
        total_prompt_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"LLM Call: model={model}, messages={len(messages)}, prompt_chars={total_prompt_chars}")
        
        start_time = time.time()
        
        try:
            # Use max_completion_tokens for newer models (GPT-5, etc.)
            # Note: GPT-5 only supports default temperature (1.0), so we omit it
            kwargs = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": int(max_tokens),
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = self.client.chat.completions.create(**kwargs)
            
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
    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        """
        Get embedding for text using OpenAI API.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Ensure text is not too long for embedding model (approx 8k token limit usually)
            # Truncate if necessary (naive truncation, better handled upstream but safety net here)
            if len(text) > 30000:
                text = text[:30000]
                
            response = self.client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise


# Singleton instance
_llm_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance
