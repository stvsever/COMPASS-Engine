"""
COMPASS Multimodal Narrative Creator Tool

Creates integrated narratives across multiple data modalities.
"""

import json
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool


class MultimodalNarrativeCreator(BaseTool):
    """
    Creates integrated clinical narratives across 2+ modalities.
    
    Identifies cross-modal patterns and convergent/divergent evidence.
    """
    
    TOOL_NAME = "MultimodalNarrativeCreator"
    PROMPT_FILE = "multimodal_narrative.txt"
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        domains = input_data.get("input_domains", [])
        
        if len(domains) < 1:
            return "At least 1 domain required for narrative creation (multimodal or unimodal fusion)"
        
        if "target_condition" not in input_data:
            return "Missing target_condition"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the narrative creation prompt."""
        domains = input_data.get("input_domains", [])
        target = input_data.get("target_condition", "neuropsychiatric")
        
        # Get dependency outputs (from previous unimodal compression)
        dep_outputs = input_data.get("dependency_outputs", {})
        
        # Get hierarchical deviation and non-numerical data (always passed)
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        non_numerical_data = input_data.get("non_numerical_data", "")
        
        prompt_parts = [
            f"## DOMAINS TO INTEGRATE: {', '.join(domains)}",
            f"\n## TARGET CONDITION: {target}",
            
            f"\n## DOMAIN SUMMARIES FROM PREVIOUS STEPS"
        ]
        
        for step_key, output in dep_outputs.items():
            if isinstance(output, dict):
                domain = output.get("domain", step_key)
                narrative = output.get("clinical_narrative", "")
                abnormalities = output.get("key_abnormalities", [])
                
                prompt_parts.append(f"\n### {domain}")
                prompt_parts.append(f"Narrative: {narrative}")
                prompt_parts.append(f"Key abnormalities: {json.dumps(abnormalities)[:500]}")
        
        prompt_parts.extend([
            f"\n## HIERARCHICAL DEVIATION PROFILE (ALWAYS INCLUDED)",
            self._summarize_deviation(hierarchical_deviation),
            
            f"\n## NON-NUMERICAL DATA (ALWAYS INCLUDED)",
            non_numerical_data[:1500] if non_numerical_data else "No non-numerical data",
            
            "\n## TASK",
            f"Create an integrated narrative across {', '.join(domains)}.",
            "Identify cross-modal patterns (convergent, divergent, complementary).",
            f"Focus on relevance to {target} prediction."
        ])
        
        return "\n".join(prompt_parts)
    
    def _process_output(
        self,
        output_data: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add synthesized domain info to output."""
        # Handle inputs from previous steps (which might be fusion results themselves)
        domains = input_data.get("input_domains", [])
        dep_outputs = input_data.get("dependency_outputs", {})
        
        # Collect domain names from dependencies
        source_domains = []
        for out in dep_outputs.values():
            if isinstance(out, dict) and "domain" in out:
                source_domains.append(out["domain"])
            elif isinstance(out, dict) and "input_domains" in out:
                source_domains.extend(out["input_domains"])
        
        # If no explicit domain found, fallback to input_domains
        if not source_domains:
            source_domains = domains
            
        # Create a "Fusion" label
        output_data["domain"] = f"Fusion:{'+'.join(sorted(list(set(source_domains))))}"
        
        # ALIAS KEYS FOR RECURSIVE FUSION COMPATIBILITY
        # The tool consumes 'clinical_narrative' and 'key_abnormalities' from dependencies.
        # But the prompt outputs 'integrated_narrative' and 'key_insights'.
        # We must map them for the next consumer.
        if "integrated_narrative" in output_data:
            output_data["clinical_narrative"] = output_data["integrated_narrative"]
            
        if "key_insights" in output_data:
            output_data["key_abnormalities"] = output_data["key_insights"]
            
        return output_data

    def _summarize_deviation(self, deviation: Dict[str, Any]) -> str:
        """Create brief summary of hierarchical deviation."""
        if not deviation:
            return "No deviation data"
        
        parts = []
        if "domain_summaries" in deviation:
            for domain, summary in deviation["domain_summaries"].items():
                if isinstance(summary, dict):
                    severity = summary.get("severity", "UNKNOWN")
                    parts.append(f"- {domain}: {severity}")
        
        return "\n".join(parts) if parts else "Deviation data structure not summarizable"
