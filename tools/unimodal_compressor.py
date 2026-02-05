"""
COMPASS Unimodal Compressor Tool

Compresses single-modality data into token-efficient clinical summaries.
"""

import json
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool
from multi_agent_system.utils.toon import json_to_toon
from multi_agent_system.utils.path_utils import split_node_path, normalize_segment
from multi_agent_system.utils.token_packer import truncate_text_by_tokens


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
        node_paths = (parameters.get("node_paths") or []) or []
        # Backward compatibility for single node_path
        if not node_paths and "node_path" in parameters:
            single_path = parameters.get("node_path")
            if single_path:
                node_paths = [single_path] if isinstance(single_path, str) else single_path

        domain_label = f"{domain}"
        if node_paths:
            paths_str = ", ".join(
                ["|".join(split_node_path(p)) if not isinstance(p, list) else "|".join(split_node_path(p)) for p in node_paths]
            )
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

        # Token-aware packing (avoid blunt character slicing).
        from multi_agent_system.config.settings import LLMBackend
        if self.settings.models.backend == LLMBackend.LOCAL:
            ctx_max = int(self.settings.models.local_max_tokens or 2048)
            prompt_budget = max(512, int(ctx_max * 0.6))
        else:
            ctx_max = 128000

            # Reserve space for the model completion + headers.
            overhead = 4000
            completion_reserve = int(self.settings.models.tool_max_tokens or 24000)
            prompt_budget = max(4000, int(ctx_max * 0.9) - overhead - min(completion_reserve, int(ctx_max * 0.3)))
        domain_budget = int(prompt_budget * 0.6)
        deviation_budget = int(prompt_budget * 0.35)

        domain_data_toon = truncate_text_by_tokens(domain_data_toon, domain_budget, model_hint="gpt-5")
        deviation_toon = truncate_text_by_tokens(deviation_toon, deviation_budget, model_hint="gpt-5")

        prompt_parts = [
            f"## DOMAIN TO COMPRESS: {domain_label}",
            f"\n## TARGET CONDITION: {target}",
            #f"\n## COMPRESSION RATIO: {compression_ratio}x",
            
            f"\n## DOMAIN DATA (TOON Format)",
            f"Note: Values are GAMLSS-normalized Z-scores (Mean=0, SD=1).",
            f"Interpretation: |Z| > 0.5 is Deviant. |Z| > 2.0 is Abnormal.",
            f"```text\n{domain_data_toon}\n```",
            
            f"\n## HIERARCHICAL DEVIATION FOR THIS DOMAIN (TOON Format)",
            f"```text\n{deviation_toon}\n```",
            
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
        Uses fuzzy matching logic to handle LLM-generated path inaccuracies.
        """
        import difflib
        
        if not hierarchical_deviation:
            return {}

        # 1. Start with the domain root
        domain_root = None
        if "root" in hierarchical_deviation:
            root = hierarchical_deviation["root"]
            
            # Try exact match first
            if root.get("node_name", "").upper() == domain.upper():
                domain_root = root
            else:
                # Try children
                for child in root.get("children", []):
                    if child.get("node_name", "").upper() == domain.upper():
                        domain_root = child
                        break
                
                # If still not found, try fuzzy match on children node_names
                if not domain_root:
                    child_names = {c.get("node_name", "").upper(): c for c in root.get("children", [])}
                    matches = difflib.get_close_matches(domain.upper(), child_names.keys(), n=1, cutoff=0.7)
                    if matches:
                        domain_root = child_names[matches[0]]

        if not domain_root:
            return {}

        # If no specific paths requested, return whole domain
        if not node_paths:
            return domain_root

        # 2. Filter children based on paths
        synthetic_root = {
            "node_name": domain_root.get("node_name"),
            "z_score": domain_root.get("z_score"),
            "severity": domain_root.get("severity"),
            "children": []
        }

        def fuzzy_find_node(current_node, segments):
            """Recursive search with fuzzy matching failsafe."""
            if not segments:
                return current_node
            
            target = normalize_segment(segments[0])
            children = current_node.get("children", [])
            child_map = {normalize_segment(c.get("node_name", "")): c for c in children}
            
            # Try exact match
            if target in child_map:
                return fuzzy_find_node(child_map[target], segments[1:])
            
            # Try fuzzy match
            matches = difflib.get_close_matches(target, child_map.keys(), n=1, cutoff=0.6)
            if matches:
                 return fuzzy_find_node(child_map[matches[0]], segments[1:])
            
            # FALLBACK: Try search in ALL descendants if it's a leaf segment
            if len(segments) == 1:
                def deep_search(node, name):
                    if normalize_segment(node.get("node_name", "")) == name:
                        return node
                    for c in node.get("children", []):
                        found = deep_search(c, name)
                        if found: return found
                    return None
                
                # Try exact deep search
                found = deep_search(current_node, target)
                if found: return found
                
                # Try fuzzy deep search (flatten names)
                all_node_names = {}
                def collect_names(node):
                    name = normalize_segment(node.get("node_name", ""))
                    if name: all_node_names[name] = node
                    for c in node.get("children", []): collect_names(c)
                
                collect_names(current_node)
                deep_matches = difflib.get_close_matches(target, all_node_names.keys(), n=1, cutoff=0.7)
                if deep_matches:
                    return all_node_names[deep_matches[0]]

            return None

        # Process each requested path
        added_nodes = set()
        for path in node_paths:
            segments = path if isinstance(path, list) else split_node_path(path)
            
            # Remove domain name from start if present
            if segments and normalize_segment(segments[0]) == normalize_segment(domain):
                segments = segments[1:]
            
            match = fuzzy_find_node(domain_root, segments)
            if match:
                # Avoid duplicates
                node_id = id(match)
                if node_id not in added_nodes:
                    synthetic_root["children"].append(match)
                    added_nodes.add(node_id)

        # If no matches found fuzzy, fallback to providing the whole domain to avoid empty input
        if not synthetic_root["children"]:
            return domain_root

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
                segments = p if isinstance(p, list) else split_node_path(p)
                # If segment[0] is domain, skip it
                if segments and normalize_segment(segments[0]) == normalize_segment(domain):
                    segments = segments[1:]
                if segments:
                    labels.append(segments[-1]) # Use leaf name
            
            if labels:
                domain = f"{domain}:{'+'.join(labels)}"

        output_data["domain"] = domain
        return output_data
