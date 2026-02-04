"""
COMPASS Fusion Layer

Compresses and integrates outputs from multiple tools into unified representations.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
import hashlib
import pickle
from pathlib import Path

from ...config.settings import get_settings
from ..llm_client import get_llm_client
from ..json_parser import parse_json_response
import json
import tiktoken
import numpy as np
import hashlib
import os
import pickle
from pathlib import Path

# Define 128k context limit and 90% threshold



logger = logging.getLogger("compass.fusion")


@dataclass
class FusionResult:
    """Result of fusing multiple tool outputs."""
    fused_narrative: str
    domain_summaries: Dict[str, str]
    key_findings: List[Dict[str, Any]]
    cross_modal_patterns: List[Dict[str, Any]]
    evidence_summary: Dict[str, List[str]]
    tokens_used: int
    source_outputs: List[str]
    # New field to signal raw pass-through
    skipped_fusion: bool = False
    raw_multimodal_data: Optional[Dict[str, Any]] = None
    raw_step_outputs: Optional[Dict[int, Dict[str, Any]]] = None



# FUSION_PROMPT removed - now loaded by Integrator Agent from agents/prompts/integrator_prompt.txt


class FusionLayer:
    """
    Fuses multiple tool outputs into unified representations.
    
    This layer:
    1. Collects outputs from all executed tool steps
    2. Compresses redundant information
    3. Integrates narratives across modalities
    4. Prepares consolidated input for the Predictor
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.llm_client = get_llm_client()
        # Initialize encoder for token counting (approximate)
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-5")
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")
            
        # Determine threshold based on backend
        from ...config.settings import LLMBackend
        if self.settings.models.backend == LLMBackend.LOCAL:
            max_ctx = self.settings.models.local_max_tokens
            self.threshold = int(0.9 * max_ctx)
            logger.info(f"FusionLayer: Dynamic Threshold set to {self.threshold} (Local Backend: {max_ctx} max)")
        else:
            max_ctx = 128000 # GPT-4/5 standard
            self.threshold = int(0.9 * max_ctx)
            
        logger.info("FusionLayer initialized")
    
    def smart_fuse(
        self,
        step_outputs: Dict[int, Dict[str, Any]],
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        multimodal_data: Dict[str, Any],
        target_condition: str,
        system_prompt: str = ""
    ) -> FusionResult:
        """
        Intelligently decide whether to fuse via LLM or pass raw data based on token usage.
        
        Logic:
        1. Identify processed subtrees from step_outputs
        2. Filter multimodal_data to only include UNPROCESSED subtrees
        3. Estimate total tokens (deviation + notes + unprocessed_multimodal + step_outputs)
        4. If < 90% of context window, SKIP FUSION and pass raw.
        5. Else, perform standard fusion.
        """

        print(f"\n[FusionLayer] Smart Fusion initiated. Threshold: {self.threshold:,} tokens")
        
        # 1. Identify processed domains/subtrees
        processed_domains = set()
        for output in step_outputs.values():
            if not output: continue
            if "domain" in output:
                # Handle sub-domains (e.g., BRAIN_MRI:structural -> BRAIN_MRI)
                domain_name = output["domain"].split(':')[0]
                processed_domains.add(domain_name)
                
        print(f"[FusionLayer] Processed top-level domains: {processed_domains}")

        # 2. Separate multimodal data into PROCESSED (ignore) and UNPROCESSED (keep raw)
        unprocessed_multimodal = {}
        for domain, features in multimodal_data.items():
            # Check if this domain (or a version of it) was processed
            if domain not in processed_domains:
                unprocessed_multimodal[domain] = features
        
        # 3. Estimate tokens
        # Raw Data Components
        dev_str = json.dumps(hierarchical_deviation, default=str)
        notes_str = non_numerical_data
        multi_str = json.dumps(unprocessed_multimodal, default=str)
        
        # Step Outputs
        outputs_desc = self._build_outputs_description(step_outputs)
        
        # System instructions estimate (approx 2000 tokens)
        system_overhead = 2000
        
        # Total estimate
        total_tokens = (
            len(self.encoder.encode(dev_str)) +
            len(self.encoder.encode(notes_str)) +
            len(self.encoder.encode(multi_str)) +
            len(self.encoder.encode(outputs_desc)) +
            system_overhead
        )
        
        print(f"[FusionLayer] Total estimated tokens: {total_tokens:,} (Limit: {self.threshold:,})")
        print(f"  - Deviation Map: {len(self.encoder.encode(dev_str)):,} tokens")
        print(f"  - Clinical Notes: {len(self.encoder.encode(notes_str)):,} tokens")
        print(f"  - Unprocessed Multimodal: {len(self.encoder.encode(multi_str)):,} tokens")
        print(f"  - Step Outputs: {len(self.encoder.encode(outputs_desc)):,} tokens")
        
        # 4. Decision Logic
        if total_tokens < self.threshold:
            print(f"[FusionLayer] ✓ UNDER THRESHOLD - SKIPPING LLM FUSION. Passing raw data.")
            
            # Context Filling: Add back PROCESSED raw data using Semantic RAG (Lowest Priority)
            remaining_tokens = self.threshold - total_tokens
            filled_multimodal = self._fill_context_with_rag(
                remaining_tokens=remaining_tokens,
                multimodal_data=multimodal_data,
                current_unprocessed=unprocessed_multimodal,
                target_condition=target_condition,
                processed_domains=list(processed_domains)
            )

            # Construct a "Pass-Through" FusionResult
            # We still populate summaries from tool outputs but don't do new synthesis
            
            # Aggregate findings from tools
            findings = []
            domain_data = {}
            for step_id, output in step_outputs.items():
                if not output: continue
                if "key_abnormalities" in output:
                    findings.extend(output["key_abnormalities"])
                if "key_findings" in output:
                    findings.extend(output["key_findings"])
                if "domain" in output and "summary" in output:
                    domain_data[output["domain"]] = output["summary"]

            return FusionResult(
                fused_narrative="Raw pass-through mode: detailed synthesis skipped in favor of raw data integrity.",
                domain_summaries=domain_data,
                key_findings=findings,
                cross_modal_patterns=[], # No cross-modal analysis done without LLM
                evidence_summary={"for_case": [], "for_control": []},
                tokens_used=0,
                source_outputs=list(step_outputs.keys()),
                skipped_fusion=True,
                raw_multimodal_data=filled_multimodal,
                raw_step_outputs=step_outputs
            )
        else:
            print(f"[FusionLayer] ⚠ OVER THRESHOLD - PERFORMING LLM FUSION to compress.")
            return self.fuse(step_outputs, hierarchical_deviation, non_numerical_data, target_condition, system_prompt, multimodal_data)

    
    def fuse(
        self,
        step_outputs: Dict[int, Dict[str, Any]],
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        target_condition: str,
        system_prompt: str = "",
        multimodal_data: Optional[Dict[str, Any]] = None
    ) -> FusionResult:
        """
        Fuse all step outputs into unified representation.
        
        Args:
            step_outputs: Map of step_id to output from plan execution
            hierarchical_deviation: Deviation map (always passed through)
            non_numerical_data: Non-numerical data (always passed through)
            target_condition: Prediction target
        
        Returns:
            FusionResult with integrated outputs
        """
        print(f"\n[FusionLayer] Fusing {len(step_outputs)} tool outputs")
        
        # Organize outputs by type
        narratives = []
        findings = []
        domain_data = {}
        
        for step_id, output in step_outputs.items():
            if not output:
                continue
                
            # Extract narratives
            if "clinical_narrative" in output:
                narratives.append(output["clinical_narrative"])
            if "integrated_narrative" in output:
                narratives.append(output["integrated_narrative"])
            
            # Extract findings - support both old (key_abnormalities) and new (abnormality_patterns) format
            if "abnormality_patterns" in output:
                for pattern in output["abnormality_patterns"]:
                    findings.append({
                        "pattern": pattern.get("pattern_name", "Unknown"),
                        "type": pattern.get("pattern_type", "UNKNOWN"),
                        "severity": pattern.get("severity", "UNKNOWN"),
                        "interpretation": pattern.get("clinical_interpretation", ""),
                        "relevance": pattern.get("relevance_score", 0.5)
                    })
            elif "key_abnormalities" in output:
                findings.extend(output["key_abnormalities"])
            if "key_findings" in output:
                findings.extend(output["key_findings"])
            
            # Extract domain summaries - support new domain_synthesis format
            if "domain" in output:
                if "domain_synthesis" in output:
                    domain_data[output["domain"]] = output["domain_synthesis"]
                elif "summary" in output:
                    domain_data[output["domain"]] = output["summary"]
        
        print(f"[FusionLayer] Collected {len(narratives)} narratives, {len(findings)} findings")
        
        # Build description of tool outputs for LLM
        outputs_description = self._build_outputs_description(step_outputs)
        
        # Create fusion prompt
        if not system_prompt:
             # Fallback if no prompt provided (should be provided by Integrator)
             logger.warning("No system_prompt provided to FusionLayer.fuse - using minimal fallback.")
             system_prompt = "You are the Fusion Layer.Fuse these outputs."
             
        prompt = system_prompt.format(tool_outputs_description=outputs_description)
        
        user_prompt = f"""## TOOL OUTPUTS TO FUSE

{outputs_description}

## HIERARCHICAL DEVIATION MAP (MEAN ABSOLUTE HIERARCHICAL DEVIATIONS)
{self._format_deviation_raw(hierarchical_deviation)}

## NON-NUMERICAL DATA (NON-TABULAR DATA)
{non_numerical_data}

## TARGET CONDITION
{target_condition}

Please fuse these outputs into a unified representation. PRESERVE all clinical notes and deviation data."""
        
        # Call LLM for intelligent fusion with auto-repair retry
        max_retries = 2
        last_error = None
        current_user_prompt = user_prompt
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"[FusionLayer] ⚠ Auto-repair attempt {attempt}/{max_retries}...")
                error_feedback = f"\n\n### PREVIOUS ERROR\nYour previous response failed validation with error: {last_error}\nPlease fix the JSON format and ensure all required fields are present."
                current_user_prompt = user_prompt + error_feedback

            try:
                response = self.llm_client.call_tool(
                    system_prompt=prompt,
                    user_prompt=current_user_prompt
                )
                
                result_json = parse_json_response(response.content)
                break # Success!
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[FusionLayer] Attempt {attempt} failed: {last_error}")
                if attempt == max_retries:
                    print(f"[FusionLayer] ✗ Fusion failed after {max_retries} retries: {last_error}")
                    raise
        
        fusion_result = FusionResult(
            fused_narrative=result_json.get("fused_narrative", ""),
            domain_summaries=result_json.get("domain_summaries", domain_data),
            key_findings=result_json.get("key_findings", findings[:10]),
            cross_modal_patterns=result_json.get("cross_modal_patterns", []),
            evidence_summary=result_json.get("evidence_summary", {"for_case": [], "for_control": []}),
            tokens_used=response.total_tokens,
            source_outputs=list(step_outputs.keys())
        )

        
        # --- POST-FUSION BACKFILL LOGIC ---
        # Did we compress too much? If so, fill with RAG.
        # Estimate current size
        current_tokens = response.total_tokens + 2000 # Overhead buffer
        remaining_buffer = self.threshold - current_tokens
        
        if remaining_buffer > 2000 and multimodal_data:
            print(f"[FusionLayer] Post-Compression Space Available: {remaining_buffer} tokens. Initiating Smart Backfill.")
            
            # For backfill, we consider ALL domains as candidates since we compressed everything
            all_domains = list(multimodal_data.keys())
            
            rag_filled_data = self._fill_context_with_rag(
                remaining_tokens=remaining_buffer,
                multimodal_data=multimodal_data,
                current_unprocessed={}, # Start fresh
                target_condition=target_condition,
                processed_domains=all_domains
            )
            
            # Attach this backfilled data to the FusionResult so compress_for_predictor can include it
            if rag_filled_data:
                 print(f"[FusionLayer] ✓ Smart Backfill successful. Enhancing context for Predictor.")
                 fusion_result.raw_multimodal_data = rag_filled_data
        
        
        print(f"[FusionLayer] ✓ Fusion complete - {len(fusion_result.key_findings)} key findings")
        print(f"[FusionLayer] Evidence direction: {result_json.get('overall_direction', 'UNKNOWN')}")
        
        return fusion_result
    
    def _build_outputs_description(
        self,
        step_outputs: Dict[int, Dict[str, Any]]
    ) -> str:
        """Build text description of all tool outputs."""
        descriptions = []
        
        for step_id, output in step_outputs.items():
            if not output:
                continue
            
            tool_name = output.get("tool_name", f"Step {step_id}")
            
            # Extract key information
            parts = [f"### {tool_name} (Step {step_id})"]
            
            if "clinical_narrative" in output:
                parts.append(f"**Narrative**: {output['clinical_narrative'][:500]}")
            
            if "key_abnormalities" in output:
                abnormalities = output["key_abnormalities"][:5]
                parts.append(f"**Abnormalities**: {abnormalities}")
            
            if "domain" in output:
                parts.append(f"**Domain**: {output['domain']}")
            
            if "confidence" in output:
                parts.append(f"**Confidence**: {output['confidence']}")
            
            descriptions.append("\n".join(parts))
        
        return "\n\n".join(descriptions)
    
    def _summarize_deviation(self, deviation: Dict[str, Any]) -> str:
        """Create brief summary of hierarchical deviation."""
        if not deviation:
            return "No deviation data available"
        
        summary_parts = []
        
        if "domain_summaries" in deviation:
            for domain, summary in deviation["domain_summaries"].items():
                if isinstance(summary, dict):
                    severity = summary.get("severity", "UNKNOWN")
                    summary_parts.append(f"- {domain}: {severity}")
                else:
                    summary_parts.append(f"- {domain}: {str(summary)[:100]}")
        
        return "\n".join(summary_parts) if summary_parts else "Deviation map available but no summaries"
    
    def compress_for_predictor(
        self,
        fusion_result: FusionResult,
        hierarchical_deviation: Dict[str, Any],
        non_numerical_data: str,
        max_tokens: int = 100000
    ) -> Dict[str, Any]:
        """
        Create final representation for Predictor.
        Handles both fused summaries and raw pass-through.
        """
        print(f"[FusionLayer] Preparing for Predictor...")
        
        if fusion_result.skipped_fusion:
            # RAW PASS-THROUGH MODE
            print(f"[FusionLayer] using RAW PASS-THROUGH format")
            compressed = {
                # Signal context
                "mode": "RAW_PASS_THROUGH",
                
                # Raw Data
                "hierarchical_deviation_raw": hierarchical_deviation,
                "non_numerical_data_raw": non_numerical_data,
                "unprocessed_multimodal_data_raw": fusion_result.raw_multimodal_data,
                
                # Tool Outputs (preserved)
                "tool_findings": fusion_result.key_findings,
                "tool_summaries": fusion_result.domain_summaries,
                "tool_outputs_raw": fusion_result.raw_step_outputs,
                
                # Placeholder for schema compatibility
                "fused_narrative": "Raw data provided - see raw fields.",
                "evidence_summary": fusion_result.evidence_summary,
            }
        else:
            # COMPRESSED MODE (Legacy)
            print(f"[FusionLayer] using COMPRESSED FUSION format")
            compressed = {
                "mode": "COMPRESSED",
                
                # Tool-derived summaries
                "fused_narrative": fusion_result.fused_narrative[:5000],
                "domain_summaries": fusion_result.domain_summaries,
                "key_findings": fusion_result.key_findings[:15],
                "cross_modal_patterns": fusion_result.cross_modal_patterns[:8],
                "evidence_summary": fusion_result.evidence_summary,
                
                # Still pass critical raw data
                "hierarchical_deviation_raw": hierarchical_deviation,
                "non_numerical_data_raw": non_numerical_data,
                
                # Note: No multimodal raw in this mode as it was too big
                # UNLESS: We performed Post-Fusion Backfill
                "backfilled_multimodal_context": fusion_result.raw_multimodal_data
            }
        
        print(f"[FusionLayer] ✓ Final predictor input ready")
        return compressed
    
    def _format_deviation_raw(self, deviation: Dict[str, Any]) -> str:
        """Format raw deviation map for inclusion in prompts."""
        import json
        try:
            # User requested large token limits - increased to 25000 chars
            return json.dumps(deviation, indent=2, default=str)[:200000]
        except:
            return str(deviation)[:200000]

    def _get_cached_embedding(self, text: str, cache_dir: Path) -> List[float]:
        """Get embedding from cache or generate new one."""
        # Create stable hash of input text
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = cache_dir / f"{text_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass # Fallback to regenerate
        
        # Generate new
        embedding = self.llm_client.get_embedding(text, model="text-embedding-3-large")
        
        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to write embedding cache: {e}")
            
        return embedding

    def _fill_context_with_rag(
        self,
        remaining_tokens: int,
        multimodal_data: Dict[str, Any],
        current_unprocessed: Dict[str, Any],
        target_condition: str,
        processed_domains: List[str]
    ) -> Dict[str, Any]:
        """
        Fill remaining context window with highest-relevance raw features using Semantic RAG.
        """
        filled_multimodal = current_unprocessed.copy()
        added_domains = []
        
        if not processed_domains:
            return filled_multimodal
            
        try:
            # 1. Prepare candidate chunks (Granular Feature Level)
            candidates = []
            for domain in processed_domains:
                if domain in multimodal_data:
                    # Extract all leaf features with breadcrumbs
                    features_with_keys = self._flatten_multimodal_features(multimodal_data[domain], parents=[domain])
                    
                    for feat, cache_key in features_with_keys:
                        # Create candidate entry
                        feat_str = json.dumps(feat, default=str)
                        feat_tokens = len(self.encoder.encode(feat_str))
                        
                        # Only add if it fits remaining individually
                        if feat_tokens < remaining_tokens:
                            candidates.append({
                                "domain": domain,
                                "data": feat,
                                "feature_name": feat.get("feature", "unknown"),
                                "text": cache_key, # Use the breadcrumb string for embedding lookup
                                "tokens": feat_tokens
                            })
            
            if candidates:
                # 2. Semantic Ranking
                # Ensure cache dir exists
                cache_dir = self.settings.paths.base_dir / ".cache" / "embeddings"
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Embed Target
                target_emb = self._get_cached_embedding(target_condition, cache_dir)
                
                scored_candidates = []
                print(f"  > RAG: Scoring {len(candidates)} feature candidates against target: '{target_condition}'")
                
                for cand in candidates:
                    # Embed candidate text
                    # Feature embedding is cheap and precise
                    cand_emb = self._get_cached_embedding(cand["text"], cache_dir)
                    
                    # Cosine Similarity
                    score = np.dot(target_emb, cand_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(cand_emb))
                    scored_candidates.append((score, cand))
                
                # Sort by score desc
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                
                # 3. Greedy Fill based on Rank
                for score, cand in scored_candidates:
                    if cand["tokens"] < remaining_tokens:
                        filled_multimodal.setdefault(cand["domain"], []).append(cand["data"])
                        remaining_tokens -= cand["tokens"]
                        added_domains.append(f"{cand['domain']}:{cand['feature_name']} (sim={score:.3f})")
                        
                if added_domains:
                    print(f"[FusionLayer] RAG Context Fill: Added {len(added_domains)} high-value features to context.")
                    for d in added_domains[:3]: 
                        print(f"  + Added: {d}")
                    if len(added_domains) > 3: print(f"  + ...and {len(added_domains)-3} more.")
                    
        except Exception as e:
            logger.error(f"RAG Context Fill failed: {e}")
            print(f"[FusionLayer] ⚠ RAG Error: {e}")
            
        return filled_multimodal

    def _flatten_multimodal_features(self, subdomain_data: Any, parents: List[str] = None) -> List[Tuple[Dict, str]]:
        """
        Recursively extract all '_leaves' lists from nested multimodal dictionary structure.
        Returns a list of tuples: (feature_dict, breadcrumb_string_for_embedding).
        Breadcrumb format: feature <- parent_last <- parent_prev <- parent_prev_prev
        """
        if parents is None:
            parents = []
            
        features = []
        if isinstance(subdomain_data, dict):
            if "_leaves" in subdomain_data and isinstance(subdomain_data["_leaves"], list):
                for leaf in subdomain_data["_leaves"]:
                    if isinstance(leaf, dict) and "feature" in leaf:
                        # Construct breadcrumb
                        context_parents = parents[-3:][::-1]
                        parts = [leaf["feature"]]
                        parts.extend(context_parents)
                        cache_key = " <- ".join(parts)
                        features.append((leaf, cache_key))
            
            for key, value in subdomain_data.items():
                if key != "_leaves" and isinstance(value, (dict, list)):
                    new_parents = parents + [key]
                    features.extend(self._flatten_multimodal_features(value, new_parents))
        
        elif isinstance(subdomain_data, list):
             for item in subdomain_data:
                if isinstance(item, dict) and "feature" in item: 
                     context_parents = parents[-3:][::-1]
                     parts = [item["feature"]]
                     parts.extend(context_parents)
                     cache_key = " <- ".join(parts)
                     features.append((item, cache_key))
                else: 
                     # Recurse if list contains non-feature logic (though unusual for this schema)
                     features.extend(self._flatten_multimodal_features(item, parents))
                    
        return features
