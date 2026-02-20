"""
COMPASS Base Agent

Abstract base class for all agents in the system.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..config.settings import get_settings, LLMBackend
from ..utils.llm_client import LLMClient, get_llm_client
from ..utils.json_parser import parse_json_response
from ..utils.core.token_manager import TokenManager

logger = logging.getLogger("compass.agents")


class BaseAgent(ABC):
    """
    Abstract base class for COMPASS agents.
    
    Provides common functionality:
    - LLM client access
    - Prompt loading
    - Logging
    - Token tracking
    """
    
    # Agent name for logging
    AGENT_NAME: str = "BaseAgent"
    
    # Prompt file name (relative to prompts directory)
    PROMPT_FILE: str = ""
    
    # Default LLM parameters (subclasses should override)
    LLM_MODEL: Optional[str] = None
    LLM_MAX_TOKENS: Optional[int] = None
    LLM_TEMPERATURE: Optional[float] = None
    JSON_EXPECTED_KEYS: Optional[List[str]] = None
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        token_manager: Optional[TokenManager] = None
    ):
        self.settings = get_settings()
        self.llm_client = llm_client or get_llm_client()
        self.token_manager = token_manager
        self.runtime_instruction: str = ""
        
        # Load system prompt
        self.system_prompt = self._load_prompt()
        
        logger.info(f"{self.AGENT_NAME} initialized")
    
    def _load_prompt(self) -> str:
        """Load the agent's system prompt from file."""
        if not self.PROMPT_FILE:
            return ""
        
        prompt_path = self.settings.paths.agent_prompts_dir / self.PROMPT_FILE
        
        if not prompt_path.exists():
            logger.warning(f"Prompt file not found: {prompt_path}")
            return ""
        
        with open(prompt_path, 'r') as f:
            return f.read()
    
    def _record_tokens(self, prompt_tokens: int, completion_tokens: int):
        """Record token usage if token manager is available."""
        if self.token_manager:
            self.token_manager.record_usage(
                component=self.AGENT_NAME.lower(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
    
    def _log_start(self, context: str = ""):
        """Log start of agent operation."""
        print(f"\n{'='*60}")
        print(f"[{self.AGENT_NAME}] Starting {context}")
        print(f"{'='*60}")
        logger.info(f"{self.AGENT_NAME} starting: {context}")
    
    def _log_complete(self, summary: str = ""):
        """Log completion of agent operation."""
        print(f"[{self.AGENT_NAME}] ✓ Complete: {summary}")
        print(f"{'='*60}\n")
        logger.info(f"{self.AGENT_NAME} complete: {summary}")
    
    def _log_error(self, error: str):
        """Log error in agent operation."""
        print(f"[{self.AGENT_NAME}] ✗ Error: {error}")
        logger.error(f"{self.AGENT_NAME} error: {error}")

    def set_runtime_instruction(self, instruction: Optional[str]) -> None:
        """Attach optional user-provided runtime instruction for this agent."""
        self.runtime_instruction = str(instruction or "").strip()

    def _append_runtime_instruction(self, prompt: str, *, label: str = "Agent Runtime Instruction") -> str:
        """Append optional runtime instruction to a prompt payload."""
        base = str(prompt or "")
        instruction = str(getattr(self, "runtime_instruction", "") or "").strip()
        if not instruction:
            return base
        return (
            f"{base}\n\n## {label}\n"
            f"{instruction}\n"
            "Apply this instruction while preserving schema validity and no-hallucination constraints."
        )
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the agent's main function.
        
        Must be implemented by subclasses.
        """
        pass
    
    def _call_llm(
        self,
        user_prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        expect_json: bool = True,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Make an LLM call with auto-repair for parsing errors.
        """
        # Set defaults from class attributes if not provided
        model = model or self.LLM_MODEL or self.settings.models.tool_model
        max_tokens = max_tokens or self.LLM_MAX_TOKENS or self.settings.models.tool_max_tokens
        temperature = (temperature if temperature is not None 
                      else(self.LLM_TEMPERATURE if self.LLM_TEMPERATURE is not None 
                           else self.settings.models.tool_temperature))

        backend_value = getattr(self.settings.models.backend, "value", self.settings.models.backend)
        backend_is_local = (
            self.settings.models.backend == LLMBackend.LOCAL
            or str(backend_value).lower() == "local"
        )

        base_prompt = user_prompt
        if expect_json and backend_is_local:
            base_prompt = (
                f"{user_prompt}\n\n"
                "STRICT OUTPUT REQUIREMENT:\n"
                "- Return ONLY one valid JSON object.\n"
                "- No markdown fences, no prose, no explanations.\n"
                "- Use double quotes for all keys/strings.\n"
                "- Ensure all required keys are present."
            )

        effective_max_retries = int(max_retries)
        if expect_json and backend_is_local and effective_max_retries < 4:
            # Local small models often need a few extra retries for strict JSON.
            effective_max_retries = 4

        current_prompt = base_prompt
        last_error = None
        
        for attempt in range(effective_max_retries + 1):
            if attempt > 0:
                print(f"[{self.AGENT_NAME}] ⚠ Auto-repair attempt {attempt}/{effective_max_retries}...")
                # Add error feedback to prompt
                error_feedback = f"\n\n### PREVIOUS ERROR\nYour previous response failed validation with error: {last_error}\nPlease fix the JSON format and ensure all required fields are present."
                current_prompt = base_prompt + error_feedback

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": current_prompt}
            ]
            
            effective_temperature = float(temperature)
            if expect_json and backend_is_local:
                effective_temperature = min(effective_temperature, 0.2)

            kwargs = {"messages": messages}
            kwargs["model"] = model
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = effective_temperature
            
            if expect_json:
                kwargs["response_format"] = {"type": "json_object"}
            
            try:
                call_start = time.time()
                print(
                    f"[{self.AGENT_NAME}] → LLM attempt {attempt + 1}/{effective_max_retries + 1} "
                    f"backend={backend_value} model={model} max_tokens={max_tokens} temp={effective_temperature:.2f}"
                )
                # Use base call method
                response = self.llm_client.call(**kwargs)
                elapsed_ms = int((time.time() - call_start) * 1000)
                print(
                    f"[{self.AGENT_NAME}] ← LLM response in {elapsed_ms}ms "
                    f"(prompt={response.prompt_tokens}, completion={response.completion_tokens}, finish={response.finish_reason})"
                )
                
                # Record tokens
                self._record_tokens(response.prompt_tokens, response.completion_tokens)
                
                if expect_json:
                    try:
                        parsed = parse_json_response(
                            response.content,
                            expected_keys=self.JSON_EXPECTED_KEYS,
                        )
                    except Exception as parse_exc:
                        if backend_is_local:
                            # Cheap repair pass on malformed JSON to avoid a full expensive re-plan.
                            try:
                                print(f"[{self.AGENT_NAME}] JSON parse failed, trying lightweight JSON repair call...")
                                repair_resp = self.llm_client.call(
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": (
                                                "You repair malformed JSON. "
                                                "Return ONLY one valid JSON object, no markdown, no explanation."
                                            ),
                                        },
                                        {"role": "user", "content": response.content},
                                    ],
                                    model=model,
                                    max_tokens=min(
                                        int(max_tokens),
                                        max(
                                            1024,
                                            int(
                                                getattr(
                                                    self.settings.token_budget,
                                                    "max_agent_output_tokens",
                                                    16000,
                                                )
                                                or 16000
                                            ),
                                        ),
                                    ),
                                    temperature=0.0,
                                    response_format={"type": "json_object"},
                                )
                                self._record_tokens(repair_resp.prompt_tokens, repair_resp.completion_tokens)
                                parsed = parse_json_response(
                                    repair_resp.content,
                                    expected_keys=self.JSON_EXPECTED_KEYS,
                                )
                                print(f"[{self.AGENT_NAME}] ✓ JSON repair call succeeded")
                                return parsed
                            except Exception as repair_exc:
                                logger.warning(
                                    f"[{self.AGENT_NAME}] JSON repair attempt failed: {type(repair_exc).__name__}: {repair_exc}"
                                )
                        preview = str(response.content or "").replace("\n", " ")[:220]
                        raise ValueError(f"JSON parse failed: {parse_exc}; preview={preview}") from parse_exc
                    if isinstance(parsed, dict):
                        print(f"[{self.AGENT_NAME}] ✓ Parsed JSON keys: {list(parsed.keys())[:8]}")
                    else:
                        print(f"[{self.AGENT_NAME}] ✓ Parsed JSON type: {type(parsed).__name__}")
                    return parsed
                
                return {"content": response.content, "tokens": response.total_tokens}
                
            except Exception as e:
                root_error = e
                if hasattr(e, "last_attempt"):
                    try:
                        candidate = e.last_attempt.exception()
                        if candidate is not None:
                            root_error = candidate
                    except Exception:
                        pass
                last_error = f"{type(root_error).__name__}: {root_error}"
                logger.warning(f"[{self.AGENT_NAME}] Attempt {attempt} failed: {last_error}")
                if attempt == effective_max_retries:
                    self._log_error(f"Failed after {effective_max_retries} retries: {last_error}")
                    raise
        
        return {}
