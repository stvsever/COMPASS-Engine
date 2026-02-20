"""
COMPASS Anomaly Narrative Builder Tool

Builds narratives from hierarchical deviation maps.
"""

from typing import Dict, Any, Optional, List

from .base_tool import BaseTool
from ..utils.toon import json_to_toon
from ..utils.token_packer import truncate_text_by_tokens


class AnomalyNarrativeBuilder(BaseTool):
    """
    Builds token-efficient narratives from hierarchical deviation maps.
    
    Creates clear, interpretable summaries of abnormality patterns
    across the multimodal data hierarchy.
    """
    
    TOOL_NAME = "AnomalyNarrativeBuilder"
    PROMPT_FILE = "anomaly_narrative.txt"
    TOOL_EXPECTED_KEYS = [
        "overall_profile",
        "domain_narratives",
        "integrated_narrative",
        "clinical_highlights",
    ]
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate that required inputs are present."""
        if not input_data.get("hierarchical_deviation"):
            return "Missing hierarchical_deviation data"
        
        if "target_condition" not in input_data:
            return "Missing target_condition"
        
        return None
    
    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build the narrative building prompt."""
        target = input_data.get("target_condition", "target phenotype")
        control = input_data.get("control_condition", "")
        hierarchical_deviation = input_data.get("hierarchical_deviation", {})
        non_numerical_data = input_data.get("non_numerical_data", "")
        
        # Analyze the deviation structure
        analysis = self._analyze_deviation(hierarchical_deviation)
        analysis_toon = truncate_text_by_tokens(
            json_to_toon(analysis),
            1200,
            model_hint="gpt-5",
        )
        deviation_toon = truncate_text_by_tokens(
            json_to_toon(hierarchical_deviation),
            2800,
            model_hint="gpt-5",
        )
        notes_text = truncate_text_by_tokens(
            str(non_numerical_data or "Not provided"),
            1200,
            model_hint="gpt-5",
        )
        
        prompt_parts = [
            f"## TARGET CONDITION: {target}",
            f"## CONTROL CONDITION: {control}",
            
            f"\n## HIERARCHICAL DEVIATION MAP (high absolute value is not necessary pathological; just abnormal (e.g., high cognitive scores))",
            f"```text\n{analysis_toon}\n```",
            
            f"\n## FULL DEVIATION STRUCTURE (TOON, TRUNCATED)",
            f"```text\n{deviation_toon}\n```",

            f"\n## NON-NUMERICAL CLINICAL NOTES (TRUNCATED)",
            f"```text\n{notes_text}\n```",
            
            "\n## TASK",
            "Build a token-efficient narrative from this hierarchical deviation map.",
            "Identify affected domains, subsystems, and key abnormalities.",
            f"Focus on patterns relevant to {target}.",
            "Create a scannable, clinically useful narrative."
        ]
        
        return "\n".join(prompt_parts)
    
    def _analyze_deviation(self, deviation: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-analyze the deviation structure."""
        analysis = {
            "total_features": 0,
            "abnormal_features": 0,
            "domains_summary": {},
            "severity_distribution": {
                "SEVERE": 0,
                "MODERATE": 0,
                "MILD": 0,
                "NORMAL": 0
            },
            "most_extreme_features": []
        }
        
        all_features = []

        meta_keys = {
            "z_score",
            "score",
            "mean_abs",
            "node_name",
            "children",
            "severity",
            "n",
            "direction",
            "percentile",
            "std",
            "mean",
        }

        def _to_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except Exception:
                return None

        def _node_score(node: Dict[str, Any]) -> Optional[float]:
            for key in ("z_score", "score", "mean_abs"):
                if key in node and node.get(key) is not None:
                    return _to_float(node.get(key))
            return None

        def _ensure_domain(domain_name: str) -> None:
            if domain_name not in analysis["domains_summary"]:
                analysis["domains_summary"][domain_name] = {
                    "total": 0,
                    "abnormal": 0,
                    "direction": {"HIGH": 0, "LOW": 0},
                }

        def traverse(node: Any, domain: str = "ROOT", node_name: str = ""):
            if isinstance(node, list):
                for idx, child in enumerate(node):
                    traverse(child, domain=domain, node_name=f"{node_name or domain}_item_{idx}")
                return

            if not isinstance(node, dict):
                return

            current_domain = domain
            explicit_name = str(node.get("node_name", "")).strip()
            if domain == "ROOT" and explicit_name:
                current_domain = explicit_name

            z = _node_score(node)
            if z is not None:
                analysis["total_features"] += 1
                severity = self._classify_severity(z)
                analysis["severity_distribution"][severity] += 1

                _ensure_domain(current_domain)
                analysis["domains_summary"][current_domain]["total"] += 1

                if severity != "NORMAL":
                    analysis["abnormal_features"] += 1
                    analysis["domains_summary"][current_domain]["abnormal"] += 1
                    all_features.append({
                        "name": explicit_name or node_name or "",
                        "z_score": z,
                        "domain": current_domain,
                        "severity": severity
                    })

                if z > 0:
                    analysis["domains_summary"][current_domain]["direction"]["HIGH"] += 1
                elif z < 0:
                    analysis["domains_summary"][current_domain]["direction"]["LOW"] += 1

            children = node.get("children", [])
            if isinstance(children, list):
                for child in children:
                    child_name = str(child.get("node_name", "")).strip() if isinstance(child, dict) else ""
                    traverse(child, domain=current_domain, node_name=child_name or node_name)
            elif isinstance(children, dict):
                for child_key, child_val in children.items():
                    traverse(child_val, domain=current_domain, node_name=str(child_key))

            for key, value in node.items():
                if key in meta_keys:
                    continue
                if isinstance(value, (dict, list)):
                    traverse(value, domain=current_domain if domain != "ROOT" else str(key), node_name=str(key))

        if isinstance(deviation, dict) and isinstance(deviation.get("root"), dict):
            traverse(deviation["root"])
        elif isinstance(deviation, dict):
            for top_key, top_val in deviation.items():
                if isinstance(top_val, (dict, list)):
                    traverse(top_val, domain=str(top_key), node_name=str(top_key))
        
        # Get most extreme features
        all_features.sort(key=lambda f: abs(f.get("z_score", 0)), reverse=True)
        analysis["most_extreme_features"] = all_features[:10]
        
        return analysis

    def _process_output(
        self,
        output_data: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate/repair model output against deterministic pre-analysis."""
        analysis = self._analyze_deviation(input_data.get("hierarchical_deviation", {}) or {})
        total_features = int(analysis.get("total_features", 0) or 0)
        abnormal_features = int(analysis.get("abnormal_features", 0) or 0)

        domain_rows = sorted(
            (analysis.get("domains_summary") or {}).items(),
            key=lambda kv: int((kv[1] or {}).get("abnormal", 0)),
            reverse=True,
        )
        affected_domains = [name for name, row in domain_rows if int((row or {}).get("abnormal", 0)) > 0]
        unaffected_domains = [name for name, row in domain_rows if int((row or {}).get("total", 0)) > 0 and int((row or {}).get("abnormal", 0)) == 0]
        top_features = analysis.get("most_extreme_features", [])[:5]

        fallback_lines = []
        if total_features <= 0:
            fallback_lines.append("No scored features were detected in the hierarchical deviation map.")
        else:
            fallback_lines.append(
                f"Deviation map contains {total_features} scored nodes with {abnormal_features} abnormal nodes (|z| > 1.5)."
            )
            if affected_domains:
                fallback_lines.append(f"Most affected domains: {', '.join(affected_domains[:5])}.")
            if top_features:
                top_txt = ", ".join(
                    f"{f.get('name') or 'feature'} (z={float(f.get('z_score', 0.0)):.2f})"
                    for f in top_features
                )
                fallback_lines.append(f"Top extreme findings: {top_txt}.")

        fallback_narrative = " ".join(fallback_lines).strip() or "Narrative unavailable."

        integrated = str(output_data.get("integrated_narrative", "") or "").strip()
        lower = integrated.lower()
        no_data_claim = any(
            phrase in lower
            for phrase in (
                "no multimodal data available",
                "absence of data",
                "no abnormalities detected",
                "total features: 0",
            )
        )
        if not integrated or (total_features > 0 and no_data_claim):
            output_data["integrated_narrative"] = fallback_narrative
            output_data["integrated_narrative_source"] = "deterministic_fallback"

        overall = output_data.get("overall_profile")
        if not isinstance(overall, dict):
            overall = {}
        if not overall.get("affected_domains"):
            overall["affected_domains"] = affected_domains
        if not overall.get("unaffected_domains"):
            overall["unaffected_domains"] = unaffected_domains
        if not overall.get("severity"):
            if analysis["severity_distribution"].get("SEVERE", 0) > 0:
                overall["severity"] = "SEVERE"
            elif analysis["severity_distribution"].get("MODERATE", 0) > 0:
                overall["severity"] = "MODERATE"
            elif analysis["severity_distribution"].get("MILD", 0) > 0:
                overall["severity"] = "MILD"
            else:
                overall["severity"] = "NORMAL"
        if not overall.get("dominant_direction"):
            total_high = sum(int((row or {}).get("direction", {}).get("HIGH", 0)) for _, row in domain_rows)
            total_low = sum(int((row or {}).get("direction", {}).get("LOW", 0)) for _, row in domain_rows)
            if total_high > 0 and total_low > 0:
                overall["dominant_direction"] = "MIXED"
            elif total_high > 0:
                overall["dominant_direction"] = "ELEVATED"
            elif total_low > 0:
                overall["dominant_direction"] = "REDUCED"
            else:
                overall["dominant_direction"] = "MIXED"
        output_data["overall_profile"] = overall

        if not isinstance(output_data.get("clinical_highlights"), list):
            output_data["clinical_highlights"] = []
        if not output_data["clinical_highlights"] and top_features:
            output_data["clinical_highlights"] = [
                f"{f.get('name') or 'feature'}: z={float(f.get('z_score', 0.0)):.2f}"
                for f in top_features[:3]
            ]

        token_eff = output_data.get("token_efficiency")
        if not isinstance(token_eff, dict):
            token_eff = {}
        token_eff.setdefault("original_features", total_features)
        token_eff.setdefault("narrative_tokens", len(str(output_data.get("integrated_narrative", "")).split()))
        token_eff.setdefault("compression_ratio", "n/a")
        output_data["token_efficiency"] = token_eff

        return output_data
    
    def _classify_severity(self, z_score: Optional[float]) -> str:
        """Classify severity based on z-score."""
        if z_score is None:
            return "NORMAL"
        
        abs_z = abs(z_score)
        if abs_z > 3.0:
            return "SEVERE"
        elif abs_z > 2.0:
            return "MODERATE"
        elif abs_z > 1.5:
            return "MILD"
        return "NORMAL"
