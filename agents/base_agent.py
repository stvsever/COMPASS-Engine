"""
COMPASS Base Agent

Abstract base class for all agents in the system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from ..config.settings import get_settings
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
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        token_manager: Optional[TokenManager] = None
    ):
        self.settings = get_settings()
        self.llm_client = llm_client or get_llm_client()
        self.token_manager = token_manager
        
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

        current_prompt = user_prompt
        last_error = None
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"[{self.AGENT_NAME}] ⚠ Auto-repair attempt {attempt}/{max_retries}...")
                # Add error feedback to prompt
                error_feedback = f"\n\n### PREVIOUS ERROR\nYour previous response failed validation with error: {last_error}\nPlease fix the JSON format and ensure all required fields are present."
                current_prompt = user_prompt + error_feedback

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": current_prompt}
            ]
            
            kwargs = {"messages": messages}
            kwargs["model"] = model
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
            
            if expect_json:
                kwargs["response_format"] = {"type": "json_object"}
            
            try:
                # Use base call method
                response = self.llm_client.call(**kwargs)
                
                # Record tokens
                self._record_tokens(response.prompt_tokens, response.completion_tokens)
                
                if expect_json:
                    return parse_json_response(response.content)
                
                return {"content": response.content, "tokens": response.total_tokens}
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{self.AGENT_NAME}] Attempt {attempt} failed: {last_error}")
                if attempt == max_retries:
                    self._log_error(f"Failed after {max_retries} retries: {last_error}")
                    raise
        
        return {}


