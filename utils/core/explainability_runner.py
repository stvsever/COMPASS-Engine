"""Orchestrates explainability methods for a selected COMPASS attempt."""

from __future__ import annotations

import importlib.util
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..json_parser import parse_json_response
from ..llm_client import get_llm_client
from .explainability_feature_space import (
    aggregate_leaf_scores_to_parent,
    normalize_abs,
)
from .explainability_prompt_builder import build_prompt_and_spans, build_prompt_only


def _load_script_module(module_name: str, filename: str):
    base = Path(__file__).resolve().parents[1] / "validation" / "explainable_AI"
    module_path = base / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_case_probability(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            text = text[:-1].strip()
            try:
                return max(0.0, min(1.0, float(text) / 100.0))
            except Exception:
                return 0.5
        try:
            return max(0.0, min(1.0, float(text)))
        except Exception:
            return 0.5
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.5


def _logit(prob: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(prob)))
    return math.log(p) - math.log(1.0 - p)


def _strip_dom_prefix(parent_id: str) -> str:
    text = str(parent_id or "")
    return text.split("dom::", 1)[1] if text.startswith("dom::") else text


def _top_leaf_summary(
    normalized_scores: Dict[str, float],
    leaf_to_feature: Dict[str, Dict[str, Any]],
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for leaf_id, score in sorted(normalized_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]:
        feat = leaf_to_feature.get(leaf_id, {})
        rows.append(
            {
                "leaf_id": leaf_id,
                "score": float(score),
                "domain": feat.get("domain"),
                "feature_name": feat.get("feature_name"),
                "path_in_hierarchy": feat.get("path_in_hierarchy") or [],
            }
        )
    return rows


def _extract_predictor_context(selected_attempt: Dict[str, Any]) -> str:
    executor_output = selected_attempt.get("executor_output") or {}
    ctx = executor_output.get("predictor_call_context") or {}
    if not isinstance(ctx, dict):
        return ""
    chunks: List[str] = []
    high = str(ctx.get("high_priority_context") or "").strip()
    if high:
        chunks.append(high)
    low = str(ctx.get("non_core_context") or "").strip()
    if low:
        chunks.append(low[:8000])
    return "\n\n".join(chunks)


def _resolve_hybrid_client_kwargs(settings) -> Dict[str, Any]:
    if getattr(settings, "openrouter_api_key", ""):
        headers: Dict[str, str] = {}
        if getattr(settings, "openrouter_site_url", ""):
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if getattr(settings, "openrouter_app_name", ""):
            headers["X-Title"] = settings.openrouter_app_name
        kwargs: Dict[str, Any] = {
            "api_key": settings.openrouter_api_key,
            "base_url": settings.openrouter_base_url,
        }
        if headers:
            kwargs["default_headers"] = headers
        return kwargs
    if getattr(settings, "openai_api_key", ""):
        return {"api_key": settings.openai_api_key}
    return {}


def _resolve_openrouter_model(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value:
        return "openai/gpt-5-nano"
    if "/" in value:
        return value
    if value.startswith(("gpt-", "o1", "o3", "o4", "text-embedding-")):
        return f"openai/{value}"
    return value


def run_explainability_methods(
    *,
    settings,
    participant_id: str,
    target_condition: str,
    control_condition: str,
    selected_attempt: Dict[str, Any],
    feature_space: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    config = settings.explainability
    requested = [str(m).strip().lower() for m in (config.methods or []) if str(m).strip()]
    methods = [m for m in requested if m in {"external", "internal", "hybrid"}]
    enabled = bool(config.enabled and methods)

    result: Dict[str, Any] = {
        "enabled": enabled,
        "methods_requested": methods,
        "run_on_final_selected_attempt": bool(config.run_on_final_selected_attempt),
        "feature_space": {
            "leaf_count": len(feature_space.get("leaf_nodes") or []),
            "parent_count": len(feature_space.get("parent_nodes") or []),
        },
        "methods": {},
    }

    if not enabled:
        result["status"] = "skipped"
        result["reason"] = "Explainability disabled or no methods requested."
        return result

    leaf_nodes: List[str] = list(feature_space.get("leaf_nodes") or [])
    leaf_to_feature: Dict[str, Dict[str, Any]] = dict(feature_space.get("leaf_to_feature") or {})
    leaf_to_parent: Dict[str, str] = dict(feature_space.get("leaf_to_parent") or {})
    if not leaf_nodes or not leaf_to_feature:
        result["status"] = "failed"
        result["reason"] = "Feature space is empty; no explainability methods executed."
        return result

    predictor_context = _extract_predictor_context(selected_attempt)
    artifacts: Dict[str, Any] = {}

    if "internal" in methods:
        started = time.time()
        try:
            ig = _load_script_module("compass_xai_ig", "ig_attribution.py")
            model, tokenizer, device = ig.load_model(model_name=config.internal_model)
            labels = ig.prepare_label_tokens(tokenizer, case_str=" CASE", control_str=" CONTROL")
            prompt, spans = build_prompt_and_spans(
                target_condition=target_condition,
                control_condition=control_condition,
                leaf_features=leaf_to_feature,
                active_leaf_ids=set(leaf_nodes),
                predictor_context=predictor_context,
            )
            raw_scores, debug = ig.integrated_gradients_feature_importance(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                feature_spans=spans,
                labels=labels,
                device=device,
                steps=int(config.internal_steps),
                baseline_mode=str(config.internal_baseline_mode),
                span_mode=str(config.internal_span_mode),
                check_completeness=bool(config.run_full_validation),
                return_debug=True,
            )
            normalized = normalize_abs(raw_scores)
            parent_raw = aggregate_leaf_scores_to_parent(raw_scores, leaf_to_parent)
            parent_norm = normalize_abs(parent_raw)
            artifacts["internal"] = {
                "status": "success",
                "model": config.internal_model,
                "steps": int(config.internal_steps),
                "baseline_mode": str(config.internal_baseline_mode),
                "span_mode": str(config.internal_span_mode),
                "duration_seconds": round(time.time() - started, 3),
                "leaf_scores_raw": raw_scores,
                "leaf_scores_l1": normalized,
                "parent_scores_raw": parent_raw,
                "parent_scores_l1": { _strip_dom_prefix(k): v for k, v in parent_norm.items() },
                "top_leaf_features": _top_leaf_summary(normalized, leaf_to_feature, top_n=20),
                "debug": debug,
            }
        except Exception as exc:
            artifacts["internal"] = {
                "status": "failed",
                "error": str(exc),
                "duration_seconds": round(time.time() - started, 3),
            }

    if "external" in methods:
        started = time.time()
        call_errors = 0
        try:
            hierarchy = feature_space.get("hierarchy_children") or {}
            root_node = feature_space.get("root_node")
            if not isinstance(hierarchy, dict) or not hierarchy or not root_node:
                raise ValueError("Missing hierarchy for external aHFR-TokenSHAP.")
            hfr = _load_script_module("compass_xai_ahfr", "aHFR_TokenSHAP.py")
            llm_client = get_llm_client()

            def score_fn(active_leaves: Set[str]) -> float:
                nonlocal call_errors
                prompt = build_prompt_only(
                    target_condition=target_condition,
                    control_condition=control_condition,
                    leaf_features=leaf_to_feature,
                    active_leaf_ids=set(active_leaves),
                    predictor_context=predictor_context,
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Return JSON only. Decide binary phenotype label and CASE probability."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}\n\n"
                            "Return JSON with keys: "
                            '{"binary_classification":"CASE|CONTROL","probability_score":0.0}'
                        ),
                    },
                ]
                try:
                    response = llm_client.call(
                        messages=messages,
                        model=settings.models.predictor_model,
                        max_tokens=min(256, int(settings.models.predictor_max_tokens or 256)),
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    parsed = parse_json_response(response.content)
                    prob = _as_case_probability(parsed.get("probability_score"))
                    return _logit(prob)
                except Exception:
                    call_errors += 1
                    return 0.0

            if int(config.external_runs) > 1 and hasattr(hfr, "shapley_with_repeats"):
                leaf_mean, leaf_std = hfr.shapley_with_repeats(
                    score_fn=score_fn,
                    feature_ids=leaf_nodes,
                    hierarchy_children=hierarchy,
                    root=root_node,
                    leaf_ids=leaf_nodes,
                    K=int(config.external_k),
                    runs=int(config.external_runs),
                    adaptive_search=bool(config.external_adaptive),
                    mixed_players=True,
                    verbose=False,
                )
                raw_scores = leaf_mean
                std_scores = leaf_std
            else:
                raw_scores = hfr.monte_carlo_hfr_tokenshap(
                    score_fn=score_fn,
                    feature_ids=leaf_nodes,
                    hierarchy_children=hierarchy,
                    root=root_node,
                    leaf_ids=leaf_nodes,
                    K=int(config.external_k),
                    adaptive_search=bool(config.external_adaptive),
                    mixed_players=True,
                    verbose=False,
                )
                std_scores = {}

            normalized = normalize_abs(raw_scores)
            parent_raw = aggregate_leaf_scores_to_parent(raw_scores, leaf_to_parent)
            parent_norm = normalize_abs(parent_raw)
            artifacts["external"] = {
                "status": "success",
                "k": int(config.external_k),
                "runs": int(config.external_runs),
                "adaptive": bool(config.external_adaptive),
                "score_function": "logit(case_probability)",
                "model": settings.models.predictor_model,
                "duration_seconds": round(time.time() - started, 3),
                "scoring_call_errors": int(call_errors),
                "leaf_scores_raw": raw_scores,
                "leaf_scores_std": std_scores,
                "leaf_scores_l1": normalized,
                "parent_scores_raw": parent_raw,
                "parent_scores_l1": {_strip_dom_prefix(k): v for k, v in parent_norm.items()},
                "top_leaf_features": _top_leaf_summary(normalized, leaf_to_feature, top_n=20),
            }
        except Exception as exc:
            artifacts["external"] = {
                "status": "failed",
                "error": str(exc),
                "duration_seconds": round(time.time() - started, 3),
                "scoring_call_errors": int(call_errors),
            }

    if "hybrid" in methods:
        started = time.time()
        try:
            llm_select = _load_script_module("compass_xai_hybrid", "LLM_select.py")
            leaf_ids = list(leaf_nodes)
            parent_ids = list(feature_space.get("parent_nodes") or [])
            client_kwargs = _resolve_hybrid_client_kwargs(settings)
            hybrid_model = _resolve_openrouter_model(str(config.hybrid_model))
            leaf_runs, parent_runs = llm_select.get_llm_select_scores(
                phenotype=target_condition,
                leaf_ids=leaf_ids,
                parent_ids=parent_ids,
                repeats=int(config.hybrid_repeats),
                model=hybrid_model,
                temperature=float(config.hybrid_temperature),
                client_kwargs=client_kwargs,
            )

            mean_leaf: Dict[str, float] = {}
            for leaf_id in leaf_ids:
                vals = [float(run.get("norm", {}).get(leaf_id, 0.0)) for run in leaf_runs]
                mean_leaf[leaf_id] = (sum(vals) / len(vals)) if vals else 0.0
            mean_parent: Dict[str, float] = {}
            for parent_id in parent_ids:
                vals = [float(run.get("norm", {}).get(parent_id, 0.0)) for run in parent_runs]
                mean_parent[parent_id] = (sum(vals) / len(vals)) if vals else 0.0

            artifacts["hybrid"] = {
                "status": "success",
                "model": hybrid_model,
                "repeats": int(config.hybrid_repeats),
                "temperature": float(config.hybrid_temperature),
                "uses_participant_values": False,
                "duration_seconds": round(time.time() - started, 3),
                "leaf_scores_l1": mean_leaf,
                "parent_scores_l1": {_strip_dom_prefix(k): v for k, v in mean_parent.items()},
                "top_leaf_features": _top_leaf_summary(mean_leaf, leaf_to_feature, top_n=20),
            }
        except Exception as exc:
            artifacts["hybrid"] = {
                "status": "failed",
                "error": str(exc),
                "duration_seconds": round(time.time() - started, 3),
            }

    result["methods"] = artifacts
    result["status"] = "success" if any(v.get("status") == "success" for v in artifacts.values()) else "failed"
    result["generated_at"] = datetime.now().isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"xai_feature_importance_{participant_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    result["artifact_path"] = str(out_path)
    return result
