"""
COMPASS Execution Logger

Detailed step-by-step execution logging for transparency.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import json

from ..config.settings import get_settings


class ExecutionLogger:
    """
    Detailed execution logger for COMPASS pipeline.
    
    Provides:
    - Console output with formatting
    - File logging for persistence
    - Structured log entries
    """
    
    def __init__(
        self,
        participant_id: str,
        log_to_file: bool = True,
        verbose: bool = True
    ):
        self.participant_id = participant_id
        self.settings = get_settings()
        self.verbose = verbose
        
        # Setup Python logger
        self.logger = logging.getLogger(f"compass.{participant_id}")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            log_dir = self.settings.paths.logs_dir
            log_file = log_dir / f"{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
        else:
            self.log_file = None
        
        # Log entries for structured access
        self.entries = []
    
    def log_pipeline_start(self, target_condition: str):
        """Log pipeline start."""
        self._log_entry("PIPELINE_START", {
            "participant_id": self.participant_id,
            "target_condition": target_condition,
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.info("=" * 60)
        self.logger.info(f"COMPASS Pipeline Started")
        self.logger.info(f"Participant: {self.participant_id}")
        self.logger.info(f"Target: {target_condition}")
        self.logger.info("=" * 60)
    
    def log_pipeline_end(self, success: bool, summary: dict):
        """Log pipeline completion."""
        self._log_entry("PIPELINE_END", {
            "success": success,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.info("=" * 60)
        self.logger.info(f"COMPASS Pipeline {'Completed' if success else 'Failed'}")
        self.logger.info(f"Result: {summary.get('prediction', 'N/A')}")
        self.logger.info(f"Probability: {summary.get('probability', 'N/A')}")
        self.logger.info(f"Iterations: {summary.get('iterations', 1)}")
        self.logger.info("=" * 60)
    
    def log_orchestrator(self, plan_summary: dict):
        """Log orchestrator output."""
        self._log_entry("ORCHESTRATOR", plan_summary)
        
        self.logger.info(f"Orchestrator created plan with {plan_summary.get('total_steps', 0)} steps")
        self.logger.debug(f"Plan details: {json.dumps(plan_summary)}")
    
    def log_executor_step(self, step_id: int, tool_name: str, status: str, tokens: int):
        """Log executor step."""
        self._log_entry("EXECUTOR_STEP", {
            "step_id": step_id,
            "tool_name": tool_name,
            "status": status,
            "tokens": tokens
        })
        
        symbol = "✓" if status == "COMPLETED" else "✗"
        self.logger.info(f"  Step {step_id}: {tool_name} {symbol} ({tokens} tokens)")
    
    def log_tool_execution(self, tool_name: str, success: bool, tokens: int, time_ms: int):
        """Log tool execution."""
        self._log_entry("TOOL_EXECUTION", {
            "tool_name": tool_name,
            "success": success,
            "tokens": tokens,
            "time_ms": time_ms
        })
        
        if self.verbose:
            self.logger.debug(f"    Tool {tool_name}: {tokens} tokens, {time_ms}ms")
    
    def log_predictor(self, prediction: dict):
        """Log predictor output."""
        self._log_entry("PREDICTOR", prediction)
        
        self.logger.info(f"Predictor: {prediction.get('classification', 'N/A')} "
                        f"(p={prediction.get('probability', 0):.3f})")
    
    def log_critic(self, evaluation: dict):
        """Log critic evaluation."""
        self._log_entry("CRITIC", evaluation)
        
        verdict = evaluation.get("verdict", "UNKNOWN")
        self.logger.info(f"Critic verdict: {verdict}")
        
        if verdict == "UNSATISFACTORY":
            self.logger.warning("Re-orchestration triggered")
    
    def log_error(self, component: str, error: str):
        """Log an error."""
        self._log_entry("ERROR", {
            "component": component,
            "error": error
        })
        
        self.logger.error(f"[{component}] {error}")
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def _log_entry(self, entry_type: str, data: dict):
        """Add structured log entry."""
        entry = {
            "type": entry_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.entries.append(entry)
    
    def get_structured_log(self) -> list:
        """Get all log entries as structured data."""
        return self.entries
    
    def save_structured_log(self, output_path: Optional[Path] = None):
        """Save structured log to JSON file."""
        if output_path is None:
            output_path = self.settings.paths.logs_dir / f"{self.participant_id}_structured.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.entries, f, indent=2)
        
        self.logger.info(f"Structured log saved to: {output_path}")


# Global logger instance
_logger_instance: Optional[ExecutionLogger] = None


def get_logger(participant_id: Optional[str] = None) -> ExecutionLogger:
    """Get or create execution logger."""
    global _logger_instance
    
    if participant_id:
        _logger_instance = ExecutionLogger(participant_id)
    
    if _logger_instance is None:
        _logger_instance = ExecutionLogger("default")
    
    return _logger_instance
