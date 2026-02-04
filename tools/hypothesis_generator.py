"""
COMPASS Hypothesis Generator Tool

Generates biomedical hypotheses explaining observed deviations.
"""

import json
from typing import Dict, Any, Optional

from .base_tool import BaseTool


class HypothesisGenerator(BaseTool):
    """
    Generates plausible biomedical hypotheses for abnormalities.
    
    Links deviations to neuropsychiatric/neurologic conditions
    through established pathophysiological mechanisms.
    """
    
    TOOL_NAME = "HypothesisGenerator"
    PROMPT_FILE = "hypothesis_generator.txt"
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if "target_condition" not in input_data:
            return "Missing target_condition"
        
        if not input_data.get("hierarchical_deviation"):
            return "Missing hierarchical_deviation data"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the hypothesis generation prompt."""
        target = input_data.get("target_condition", "neuropsychiatric")
        
        # Get abnormality data from dependency outputs or hierarchical deviation
        dep_outputs = input_data.get("dependency_outputs", {})
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        
        # Collect abnormalities from various sources
        abnormalities = self._collect_abnormalities(dep_outputs, hierarchical_deviation)
        
        # Get patient context
        participant_id = input_data.get("participant_id", "unknown")
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
            f"## PARTICIPANT: {participant_id}",
            
            f"\n## DETECTED ABNORMALITIES",
            json.dumps(abnormalities, indent=2)[:3000],
            
            f"\n## HIERARCHICAL DEVIATION PROFILE",
            self._format_deviation_summary(hierarchical_deviation),
            
            "\n## TASK",
            f"Generate biomedical hypotheses explaining these abnormalities.",
            f"Link to {target} disorder mechanisms.",
            "Provide primary and alternative hypotheses.",
            "Include differential considerations."
        ]
        
        return "\n".join(prompt_parts)
    
    def _collect_abnormalities(
        self,
        dep_outputs: Dict[str, Any],
        hierarchical_deviation: Dict[str, Any]
    ) -> list:
        """Collect all abnormalities from various sources."""
        abnormalities = []
        
        # From dependency outputs
        for step_key, output in dep_outputs.items():
            if isinstance(output, dict):
                if "key_abnormalities" in output:
                    abnormalities.extend(output["key_abnormalities"][:5])
        
        # From domain summaries
        if "domain_summaries" in hierarchical_deviation:
            for domain, summary in hierarchical_deviation["domain_summaries"].items():
                if isinstance(summary, dict) and summary.get("severity") in ["SEVERE", "MODERATE"]:
                    abnormalities.append({
                        "domain": domain,
                        "severity": summary.get("severity"),
                        "description": f"{domain} shows {summary.get('severity')} abnormality"
                    })
        
        return abnormalities[:15]  # Limit to top 15
    
    def _format_deviation_summary(self, deviation: Dict[str, Any]) -> str:
        """Format deviation for prompt."""
        if not deviation:
            return "No deviation data available"
        
        parts = []
        if "domain_summaries" in deviation:
            for domain, summary in deviation["domain_summaries"].items():
                if isinstance(summary, dict):
                    parts.append(f"- {domain}: {summary.get('severity', 'UNKNOWN')}")
                else:
                    parts.append(f"- {domain}: {str(summary)[:100]}")
        
        return "\n".join(parts) if parts else "Deviation format not recognized"
