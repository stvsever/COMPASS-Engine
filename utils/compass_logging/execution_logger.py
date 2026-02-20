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

from ...config.settings import get_settings


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
    
    def log_pipeline_start(
        self,
        target_condition: str,
        control_condition: str,
        prediction_task_mode: Optional[str] = None,
    ):
        """Log pipeline start."""
        mode = str(prediction_task_mode or "").strip()
        is_classification_mode = mode.endswith("_classification")
        self._log_entry("PIPELINE_START", {
            "participant_id": self.participant_id,
            "target_condition": target_condition,
            "control_condition": control_condition,
            "prediction_task_mode": mode,
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.info("=" * 60)
        self.logger.info(f"COMPASS Pipeline Started")
        self.logger.info(f"Participant: {self.participant_id}")
        self.logger.info(f"Target: {target_condition}")
        if mode:
            self.logger.info(f"Prediction Mode: {mode}")
        if is_classification_mode and str(control_condition or "").strip():
            self.logger.info(f"Control: {control_condition}")
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
        prob = summary.get("probability", None)
        if isinstance(prob, (int, float)):
            self.logger.info(f"Probability / Root Confidence: {float(prob):.3f}")
        else:
            self.logger.info("Probability / Root Confidence: N/A")
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

        prob = prediction.get("probability", None)
        mode = str(prediction.get("prediction_task_mode") or "").strip()
        label = prediction.get('classification', 'N/A')
        if isinstance(prob, (int, float)):
            prob_text = f"{float(prob):.3f}"
            self.logger.info(
                f"Predictor: {label} [{mode or 'unknown_mode'}] (confidence={prob_text})"
            )
        else:
            self.logger.info(
                f"Predictor: {label} [{mode or 'unknown_mode'}]"
            )
    
    def log_critic(self, evaluation: dict):
        """Log critic evaluation."""
        self._log_entry("CRITIC", evaluation)
        
        verdict = evaluation.get("verdict", "UNKNOWN")
        self.logger.info(f"Critic verdict: {verdict}")
        
        if verdict == "UNSATISFACTORY":
            self.logger.warning("Re-orchestration triggered")

    def log_dataflow_summary(self, summary: dict, iteration: Optional[int] = None):
        """Log dataflow integrity summary and assertions."""
        payload = {
            "iteration": iteration,
            **(summary or {})
        }
        self._log_entry("DATAFLOW_SUMMARY", payload)
        if self.verbose:
            missing = payload.get("coverage", {}).get("missing_feature_count")
            chunks = payload.get("chunking", {}).get("predictor_chunk_count")
            invariant_ok = payload.get("assertions", {}).get("invariant_ok")
            self.logger.info(
                f"Dataflow summary: missing={missing} chunks={chunks} invariant_ok={invariant_ok}"
            )

    def log_explainability(self, payload: dict):
        """Log explainability method execution summary."""
        self._log_entry("EXPLAINABILITY", payload or {})
        methods = (payload or {}).get("methods", {}) or {}
        method_names = sorted(methods.keys())
        success_count = sum(1 for name in method_names if (methods.get(name) or {}).get("status") == "success")
        self.logger.info(
            "Explainability: status=%s methods=%s success=%s/%s",
            (payload or {}).get("status"),
            ",".join(method_names) if method_names else "none",
            success_count,
            len(method_names),
        )
    
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
