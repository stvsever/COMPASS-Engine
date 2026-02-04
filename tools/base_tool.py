"""
COMPASS Base Tool

Abstract base class for all tools in the system.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from ..config.settings import get_settings
from ..utils.llm_client import LLMClient, get_llm_client
from ..utils.json_parser import parse_json_response

logger = logging.getLogger("compass.tools")


@dataclass
class ToolOutput:
    """Standard output structure for all tools."""
    tool_name: str
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    execution_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            **self.output,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "execution_time_ms": self.execution_time_ms
        }


class BaseTool(ABC):
    """
    Abstract base class for COMPASS tools.
    
    All tools use GPT-5-nano for processing and follow a standard
    input/output pattern with automatic error handling.
    """
    
    # Tool name for logging and registration
    TOOL_NAME: str = "BaseTool"
    
    # Prompt file name (relative to tool_prompts directory)
    PROMPT_FILE: str = ""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None
    ):
        self.settings = get_settings()
        self.llm_client = llm_client or get_llm_client()
        
        # Load system prompt
        self.system_prompt = self._load_prompt()
        
        logger.debug(f"Tool {self.TOOL_NAME} initialized")
    
    def _load_prompt(self) -> str:
        """Load the tool's system prompt from file."""
        if not self.PROMPT_FILE:
            return ""
        
        prompt_path = self.settings.paths.tool_prompts_dir / self.PROMPT_FILE
        
        if not prompt_path.exists():
            logger.warning(f"Tool prompt not found: {prompt_path}")
            return ""
        
        with open(prompt_path, 'r') as f:
            return f.read()
    
    def execute(self, input_data: Dict[str, Any]) -> ToolOutput:
        """
        Execute the tool with the given input.
        
        Args:
            input_data: Tool-specific input data
        
        Returns:
            ToolOutput with results or error
        """
        start_time = time.time()
        
        logger.debug(f"Executing tool: {self.TOOL_NAME}")
        print(f"  [Tool:{self.TOOL_NAME}] Starting execution...")
        
        try:
            # Validate input
            validation_error = self._validate_input(input_data)
            if validation_error:
                return ToolOutput(
                    tool_name=self.TOOL_NAME,
                    success=False,
                    error=f"Input validation failed: {validation_error}",
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Build user prompt
            user_prompt = self._build_prompt(input_data)
            
            # Call LLM
            response = self.llm_client.call_tool(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt
            )
            
            # Debug: Log raw response (handle empty)
            content = response.content or ""
            preview = content[:500] if len(content) > 500 else content
            print(f"  [Tool:{self.TOOL_NAME}] Raw response ({len(content)} chars): {repr(preview)}")
            
            # Check for empty response
            if not content or not content.strip():
                raise ValueError("LLM returned empty response")
            
            # Parse response
            output_data = parse_json_response(response.content)
            
            # Post-process output
            processed = self._process_output(output_data, input_data)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            print(f"  [Tool:{self.TOOL_NAME}] ✓ Complete ({response.total_tokens} tokens, {execution_time}ms)")
            
            return ToolOutput(
                tool_name=self.TOOL_NAME,
                success=True,
                output=processed,
                tokens_used=response.total_tokens,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.exception(f"Tool {self.TOOL_NAME} failed: {e}")
            print(f"  [Tool:{self.TOOL_NAME}] ✗ Error: {str(e)[:50]}")
            
            return ToolOutput(
                tool_name=self.TOOL_NAME,
                success=False,
                error=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """
        Validate input data.
        
        Override in subclasses for tool-specific validation.
        Returns error message if invalid, None if valid.
        """
        return None
    
    @abstractmethod
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        Build the user prompt for the LLM.
        
        Must be implemented by subclasses.
        """
        pass
    
    def _process_output(
        self,
        output_data: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Post-process the LLM output.
        
        Override in subclasses for tool-specific processing.
        """
        return output_data


# Tool registry and factory
_tool_instances: Dict[str, BaseTool] = {}


def get_tool(tool_name) -> Optional[BaseTool]:
    """
    Get or create a tool instance by name.
    
    Args:
        tool_name: Tool name string or ToolName enum
    
    Returns:
        Tool instance or None if not found
    """
    # Handle enum
    if hasattr(tool_name, 'value'):
        tool_name = tool_name.value
    
    # Check cache
    if tool_name in _tool_instances:
        return _tool_instances[tool_name]
    
    # Import tool classes
    from . import TOOL_REGISTRY
    
    if tool_name not in TOOL_REGISTRY:
        logger.error(f"Unknown tool: {tool_name}")
        return None
    
    # Create instance
    tool_class = TOOL_REGISTRY[tool_name]
    instance = tool_class()
    _tool_instances[tool_name] = instance
    
    return instance
