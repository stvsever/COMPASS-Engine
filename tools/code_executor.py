"""
COMPASS Code Executor Tool

Executes Python code safely for custom analyses.
"""

import logging
from typing import Dict, Any, Optional

from .base_tool import BaseTool

logger = logging.getLogger("compass.tools.code_executor")


# Allowed imports for sandboxed execution
ALLOWED_IMPORTS = {
    "numpy": "numpy",
    "np": "numpy",
    "scipy.stats": "scipy.stats",
    "statistics": "statistics",
    "math": "math",
    "json": "json",
}


class CodeExecutor(BaseTool):
    """
    Executes Python code in a sandboxed environment.
    
    For statistical computations and custom analyses that
    cannot be performed by other tools.
    """
    
    TOOL_NAME = "CodeExecutor"
    PROMPT_FILE = "code_executor.txt"
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate input and check for dangerous operations."""
        parameters = input_data.get("parameters", {})
        code = parameters.get("code", "")
        
        if not code:
            return "No code provided to execute"
        
        # Safety checks
        dangerous_patterns = [
            "import os", "import sys", "import subprocess",
            "exec(", "eval(", "__import__",
            "open(", "file(", "write(",
            "requests.", "urllib.", "socket.",
            "os.system", "os.popen", "os.exec",
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                return f"Unsafe operation detected: {pattern}"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the code analysis/execution prompt."""
        parameters = input_data.get("parameters", {})
        code = parameters.get("code", "")
        expected_output = parameters.get("expected_output", "Analysis results")
        
        # Get input data for the code
        code_input_data = parameters.get("input_data", {})
        if not code_input_data:
            # Try to extract from hierarchical deviation
            code_input_data = self._extract_code_input(input_data)
        
        prompt_parts = [
            "## CODE TO ANALYZE AND EXECUTE",
            f"```python\n{code}\n```",
            
            f"\n## EXPECTED OUTPUT: {expected_output}",
            
            f"\n## INPUT DATA FOR CODE",
            f"```json\n{str(code_input_data)[:2000]}\n```",
            
            "\n## TASK",
            "1. Analyze the code for correctness and safety",
            "2. Execute the code with the provided input data",
            "3. Return the execution result",
            "",
            "Return JSON with:",
            "- execution_id: unique identifier",
            "- status: SUCCESS or ERROR",
            "- result: the output from the code execution",
            "- error: error message if failed"
        ]
        
        return "\n".join(prompt_parts)
    
    def _extract_code_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data to pass to code execution."""
        code_input = {}
        
        # Extract z-scores from hierarchical deviation
        deviation = input_data.get("hierarchical_deviation", {})
        if deviation:
            z_scores = []
            self._collect_z_scores(deviation.get("root", {}), z_scores)
            code_input["z_scores"] = z_scores[:100]  # Limit size
        
        return code_input
    
    def _collect_z_scores(self, node: Dict[str, Any], z_scores: list):
        """Recursively collect z-scores from deviation tree."""
        if "z_score" in node and node["z_score"] is not None:
            z_scores.append(node["z_score"])
        
        for child in node.get("children", []):
            self._collect_z_scores(child, z_scores)
    
    def _process_output(
        self,
        output_data: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mark as code execution output."""
        output_data["tool_type"] = "code_execution"
        return output_data
