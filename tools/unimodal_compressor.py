"""
COMPASS Unimodal Compressor Tool

Compresses single-modality data into token-efficient clinical summaries.
"""

import json
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool
from multi_agent_system.utils.toon import json_to_toon


class UnimodalCompressor(BaseTool):
    """
    Compresses single-domain data into clinical summaries.
    
    Maximizes retention of clinically relevant information while
    reducing token count for efficient processing.
    """
    
    TOOL_NAME = "UnimodalCompressor"
    PROMPT_FILE = "unimodal_compressor.txt"
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        required = ["input_domains", "target_condition"]
        
        for key in required:
            if key not in input_data:
                return f"Missing required input: {key}"
        
        if not input_data.get("input_domains"):
            return "At least one input domain required"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the compression prompt."""
        domain = input_data.get("input_domains", ["UNKNOWN"])[0]
        target = input_data.get("target_condition", "neuropsychiatric")
        parameters = input_data.get("parameters", {})
        compression_ratio = parameters.get("compression_ratio", 5)
        node_paths = input_data.get("parameters", {}).get("node_paths", [])
        # Backward compatibility for single node_path
        if not node_paths and "node_path" in input_data.get("parameters", {}):
            single_path = input_data.get("parameters", {}).get("node_path")
            if single_path:
                node_paths = [single_path] if isinstance(single_path, str) else single_path

        domain_label = f"{domain}"
        if node_paths:
            paths_str = ", ".join(["_".join(p) if isinstance(p, list) else str(p) for p in node_paths])
            domain_label = f"{domain} (Focus: {paths_str})"
        
        # Get domain data
        domain_data = input_data.get("domain_data", {}).get(domain, {})
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        
        # Format hierarchical data for this domain (extracting specific subtrees)
        domain_deviation = self._extract_subtrees(hierarchical_deviation, domain, node_paths)
        
        # Helper to strict serialize Pydantic models
        def to_dict(obj):
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            if hasattr(obj, 'dict'):
                return obj.dict()
            return obj

        # Serialize domain data
        if isinstance(domain_data, list):
            serializable_domain_data = [to_dict(item) for item in domain_data]
        else:
            serializable_domain_data = domain_data
            
        # Serialize deviation data
        if hasattr(domain_deviation, 'model_dump'):
            serializable_deviation = domain_deviation.model_dump()
        elif hasattr(domain_deviation, 'dict'):
            serializable_deviation = domain_deviation.dict()
        else:
            serializable_deviation = domain_deviation

        # Convert to TOON format for token compression
        # We increase the character limit significantly as TOON is much more compact
        domain_data_toon = json_to_toon(serializable_domain_data)
        deviation_toon = json_to_toon(serializable_deviation)

        prompt_parts = [
            f"## DOMAIN TO COMPRESS: {domain_label}",
            f"\n## TARGET CONDITION: {target}",
            #f"\n## COMPRESSION RATIO: {compression_ratio}x",
            
            f"\n## DOMAIN DATA (TOON Format)",
            f"Note: Values are GAMLSS-normalized Z-scores (Mean=0, SD=1).",
            f"Interpretation: |Z| > 0.5 is Deviant. |Z| > 2.0 is Abnormal.",
            f"```text\n{domain_data_toon[:32000]}\n```",
            
            f"\n## HIERARCHICAL DEVIATION FOR THIS DOMAIN (TOON Format)",
            f"```text\n{deviation_toon[:32000]}\n```",
            
            "\n## TASK",
            f"Compress the {domain} domain data into a token-efficient clinical summary.",
            f"Focus on information relevant to {target} prediction. "
            f"Still highly detailled and variance-maximizing relevance for phenotypic prediction.",
            #f"Target compression ratio: {compression_ratio}x"
        ]
        
        return "\n".join(prompt_parts)
    
    def _extract_subtrees(
        self,
        hierarchical_deviation: Dict[str, Any],
        domain: str,
        node_paths: List[Any]
    ) -> Dict[str, Any]:
        """
        Extract specific subtrees matching the provided paths.
        Supports selecting multiple subtrees (e.g. ['BRAIN_MRI:structural', 'BRAIN_MRI:functional']).
        """
        if not hierarchical_deviation:
            return {}

        # 1. Start with the domain root
        domain_root = None
        if "root" in hierarchical_deviation:
            root = hierarchical_deviation["root"]
            # Find the domain node
            if root.get("node_name", "").upper() == domain.upper():
                domain_root = root
            else:
                for child in root.get("children", []):
                    if child.get("node_name", "").upper() == domain.upper():
                        domain_root = child
                        break
        
        if not domain_root:
            return {}

        # If no specific paths requested, return whole domain
        if not node_paths:
            return domain_root

        # 2. Filter children based on paths
        # We create a synthetic root containing only the requested branches
        synthetic_root = {
            "node_name": domain_root.get("node_name"),
            "z_score": domain_root.get("z_score"),
            "severity": domain_root.get("severity"),
            "children": []
        }

        def find_node_by_path(current_node, target_path_segments):
            """Recursive search for a node matching the path segments."""
            if not target_path_segments:
                return current_node
            
            next_segment = target_path_segments[0].upper()
            for child in current_node.get("children", []):
                if child.get("node_name", "").upper() == next_segment:
                    return find_node_by_path(child, target_path_segments[1:])
            return None

        # Process each requested path
        for path in node_paths:
            # Normalize path: 'DOMAIN:Sub:Leaf' -> ['Sub', 'Leaf'] (assuming domain is already handled)
            # The orchestrator usually passes full paths like ['BRAIN_MRI', 'structural']
            # We strip the domain prefix if present
            segments = path if isinstance(path, list) else str(path).split(":")
            
            # Remove domain name from start if present (e.g. BRAIN_MRI -> Structural)
            if segments and segments[0].upper() == domain.upper():
                segments = segments[1:]
            
            match = find_node_by_path(domain_root, segments)
            if match:
                synthetic_root["children"].append(match)

        return synthetic_root
    
    def _process_output(
        self,
        output_data: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add domain info to output."""
        domain = input_data.get("input_domains", ["UNKNOWN"])[0]
        
        # Check for node_paths to create a specific domain label
        parameters = input_data.get("parameters", {})
        node_paths = parameters.get("node_paths", [])
        # Backward compatibility check
        if not node_paths and "node_path" in parameters:
             single = parameters["node_path"]
             node_paths = [single] if isinstance(single, list) else [single]
             
        if node_paths:
            # Create label likes "BRAIN_MRI:Structural+Functional"
            # Extract last segment of each path
            labels = []
            for p in node_paths:
                segments = p if isinstance(p, list) else str(p).split(":")
                # If segment[0] is domain, skip it
                if segments and segments[0].upper() == domain.upper():
                    segments = segments[1:]
                if segments:
                    labels.append(segments[-1]) # Use leaf name
            
            if labels:
                domain = f"{domain}:{'+'.join(labels)}"

        output_data["domain"] = domain
        return output_data
