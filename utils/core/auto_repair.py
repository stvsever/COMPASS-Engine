"""
COMPASS Auto-Repair Module

Self-healing functionality for failed tool calls.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from ...config.settings import get_settings
from ...data.models.execution_plan import PlanStep, ToolName

logger = logging.getLogger("compass.auto_repair")


class ErrorType(str, Enum):
    """Categories of errors for repair strategies."""
    JSON_PARSE_ERROR = "json_parse_error"
    API_TIMEOUT = "api_timeout"
    TOKEN_LIMIT = "token_limit"
    MISSING_INPUT = "missing_input"
    INVALID_OUTPUT = "invalid_output"
    TOOL_NOT_FOUND = "tool_not_found"
    UNKNOWN = "unknown"


@dataclass
class RepairStrategy:
    """A strategy for repairing a specific error type."""
    error_type: ErrorType
    description: str
    modifications: Dict[str, Any]


class AutoRepair:
    """
    Automatic repair for failed tool calls.
    
    Analyzes errors and applies appropriate repair strategies:
    - Retry with modified parameters
    - Reduce input size for token limits
    - Simplify requests for parse errors
    - Skip optional steps if unrecoverable
    """
    
    def __init__(self, max_retries: int = 3):
        self.settings = get_settings()
        self.max_retries = max_retries
        
        # Define repair strategies per error type
        self.strategies = self._build_strategies()
        
        logger.info(f"AutoRepair initialized with {max_retries} max retries")
    
    def _build_strategies(self) -> Dict[ErrorType, RepairStrategy]:
        """Build repair strategies for each error type."""
        return {
            ErrorType.JSON_PARSE_ERROR: RepairStrategy(
                error_type=ErrorType.JSON_PARSE_ERROR,
                description="Retry with explicit JSON format instruction",
                modifications={
                    "add_json_instruction": True,
                    "lower_temperature": 0.2
                }
            ),
            ErrorType.API_TIMEOUT: RepairStrategy(
                error_type=ErrorType.API_TIMEOUT,
                description="Retry with extended timeout",
                modifications={
                    "extend_timeout": True,
                    "reduce_max_tokens": 0.8  # Reduce by 20%
                }
            ),
            ErrorType.TOKEN_LIMIT: RepairStrategy(
                error_type=ErrorType.TOKEN_LIMIT,
                description="Reduce input size and retry",
                modifications={
                    "compress_input": True,
                    "truncate_ratio": 0.5
                }
            ),
            ErrorType.MISSING_INPUT: RepairStrategy(
                error_type=ErrorType.MISSING_INPUT,
                description="Provide default values for missing inputs",
                modifications={
                    "use_defaults": True
                }
            ),
            ErrorType.INVALID_OUTPUT: RepairStrategy(
                error_type=ErrorType.INVALID_OUTPUT,
                description="Retry with stricter output format",
                modifications={
                    "enforce_schema": True,
                    "add_examples": True
                }
            ),
            ErrorType.UNKNOWN: RepairStrategy(
                error_type=ErrorType.UNKNOWN,
                description="Generic retry with lower temperature",
                modifications={
                    "lower_temperature": 0.1,
                    "add_clarification": True
                }
            )
        }
    
    def can_repair(self, step: PlanStep, error: Optional[str]) -> bool:
        """
        Check if a failed step can be repaired.
        
        Returns False if:
        - Max retries exceeded
        - Error type is unrecoverable
        - Step is marked as non-retryable
        """
        if step.retry_count >= self.max_retries:
            logger.info(f"Step {step.step_id} exceeded max retries ({self.max_retries})")
            print(f"[AutoRepair] Step {step.step_id} exceeded max retries")
            return False
        
        error_type = self._classify_error(error)
        
        if error_type == ErrorType.TOOL_NOT_FOUND:
            logger.info(f"Step {step.step_id} has unrecoverable error: tool not found")
            print(f"[AutoRepair] Step {step.step_id} unrecoverable: tool not found")
            return False
        
        return True
    
    def _classify_error(self, error: Optional[str]) -> ErrorType:
        """Classify error string into an error type."""
        if not error:
            return ErrorType.UNKNOWN
        
        error_lower = error.lower()
        
        if "json" in error_lower or "parse" in error_lower:
            return ErrorType.JSON_PARSE_ERROR
        
        if "timeout" in error_lower or "timed out" in error_lower:
            return ErrorType.API_TIMEOUT
        
        if "token" in error_lower or "context length" in error_lower:
            return ErrorType.TOKEN_LIMIT
        
        if "missing" in error_lower or "required" in error_lower:
            return ErrorType.MISSING_INPUT
        
        if "invalid" in error_lower or "schema" in error_lower:
            return ErrorType.INVALID_OUTPUT
        
        if "not found" in error_lower:
            return ErrorType.TOOL_NOT_FOUND
        
        return ErrorType.UNKNOWN
    
    def get_repair_context(
        self,
        step: PlanStep,
        error: Optional[str]
    ) -> Dict[str, Any]:
        """
        Get repair context for retrying a step.
        
        Returns modifications to apply to the retry attempt.
        """
        error_type = self._classify_error(error)
        strategy = self.strategies.get(error_type, self.strategies[ErrorType.UNKNOWN])
        
        repair_context = {
            "retry_number": step.retry_count + 1,
            "original_error": error,
            "error_type": error_type.value,
            "strategy": strategy.description,
            "modifications": strategy.modifications.copy()
        }
        
        logger.info(
            f"Repair context for step {step.step_id}: "
            f"{error_type.value} -> {strategy.description}"
        )
        print(f"[AutoRepair] Strategy: {strategy.description}")
        
        return repair_context
    
    def apply_modifications(
        self,
        tool_input: Dict[str, Any],
        repair_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply repair modifications to tool input.
        
        Returns modified input for retry.
        """
        modified = tool_input.copy()
        mods = repair_context.get("modifications", {})
        
        # Apply specific modifications
        if mods.get("add_json_instruction"):
            if "parameters" not in modified:
                modified["parameters"] = {}
            modified["parameters"]["force_json"] = True
        
        if mods.get("lower_temperature"):
            modified["temperature"] = mods["lower_temperature"]
        
        if mods.get("compress_input"):
            modified = self._compress_input(modified, mods.get("truncate_ratio", 0.5))
        
        if mods.get("use_defaults"):
            modified = self._fill_defaults(modified)
        
        if mods.get("enforce_schema"):
            if "parameters" not in modified:
                modified["parameters"] = {}
            modified["parameters"]["strict_schema"] = True
        
        # Add error context for LLM to learn from
        modified["previous_error"] = repair_context.get("original_error")
        modified["repair_attempt"] = repair_context.get("retry_number")
        
        return modified
    
    def _compress_input(
        self,
        tool_input: Dict[str, Any],
        ratio: float
    ) -> Dict[str, Any]:
        """Compress text inputs to fit token limits."""
        compressed = tool_input.copy()
        
        # Truncate large string fields
        for key, value in compressed.items():
            if isinstance(value, str) and len(value) > 1000:
                target_len = int(len(value) * ratio)
                compressed[key] = value[:target_len] + "... [truncated]"
            elif isinstance(value, dict):
                # Recursively compress nested dicts
                compressed[key] = self._compress_input(value, ratio)
        
        return compressed
    
    def _fill_defaults(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in default values for missing required fields."""
        filled = tool_input.copy()
        
        # Common defaults
        defaults = {
            "target_condition": "neuropsychiatric",
            "compression_ratio": 5,
            "confidence": "MEDIUM"
        }
        
        if "parameters" not in filled:
            filled["parameters"] = {}
        
        for key, default in defaults.items():
            if key not in filled["parameters"]:
                filled["parameters"][key] = default
        
        return filled
    
    def log_repair_attempt(
        self,
        step_id: int,
        attempt: int,
        success: bool,
        strategy: str
    ):
        """Log a repair attempt for tracking."""
        status = "SUCCESS" if success else "FAILED"
        logger.info(
            f"Repair attempt {attempt} for step {step_id}: {status} ({strategy})"
        )
        print(f"[AutoRepair] Attempt {attempt}: {status}")
