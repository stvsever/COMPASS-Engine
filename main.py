#!/usr/bin/env python3
"""
COMPASS Multi-Agent System

Clinical Ontology-driven Multi-modal Predictive Agentic Support System

Main entry point for running the COMPASS pipeline on participant data.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Default control condition (non-case comparator)
DEFAULT_CONTROL_CONDITION = "brain-implicated pathology, but NOT psychiatric"

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_system.config.settings import get_settings, LLMBackend
from multi_agent_system.utils.core.data_loader import DataLoader
from multi_agent_system.utils.core.token_manager import TokenManager
from multi_agent_system.utils.llm_client import get_llm_client, reset_llm_client
from multi_agent_system.utils.core.fusion_layer import FusionLayer, FusionResult
from multi_agent_system.utils.core.predictor_input_assembler import PredictorInputAssembler
from multi_agent_system.utils.core.explainability_feature_space import build_feature_space
from multi_agent_system.utils.core.explainability_runner import run_explainability_methods
from multi_agent_system.utils.token_packer import count_tokens
from multi_agent_system.agents.orchestrator import Orchestrator
from multi_agent_system.agents.executor import Executor
from multi_agent_system.agents.predictor import Predictor
from multi_agent_system.agents.critic import Critic
from multi_agent_system.agents.communicator import Communicator
from multi_agent_system.utils.compass_logging.execution_logger import ExecutionLogger
from multi_agent_system.utils.compass_logging.decision_trace import DecisionTrace
from multi_agent_system.utils.compass_logging.patient_report import PatientReportGenerator
from multi_agent_system.data.models.prediction_result import Verdict
from multi_agent_system.frontend.compass_ui import get_ui, reset_ui, start_ui_loop
from multi_agent_system.utils.participant_resolver import resolve_participant_dir


def _resolve_output_dir(participant_dir: Path, participant_id: str, settings) -> Path:
    pseudo_inputs = settings.paths.base_dir / "data" / "pseudo_data" / "inputs"
    pseudo_outputs = settings.paths.base_dir / "data" / "pseudo_data" / "outputs"
    try:
        if pseudo_inputs in participant_dir.resolve().parents:
            return pseudo_outputs / f"participant_{participant_id}"
    except Exception:
        pass
    return settings.paths.output_dir / f"participant_{participant_id}"




def _build_report_context_note(final_evaluation, selected_iteration: int, selection_reason: str) -> str:
    if not final_evaluation or final_evaluation.verdict == Verdict.SATISFACTORY:
        return ""
    return (
        "WARNING: Selected final verdict remains UNSATISFACTORY. "
        f"Selected iteration: {selected_iteration}. "
        f"Selection basis: {selection_reason}. "
        "Interpret deep phenotype content as exploratory and not production-final."
    )


def _append_execution_log_entry(output_dir: Path, participant_id: str, entry: Dict[str, Any]) -> None:
    log_path = output_dir / f"execution_log_{participant_id}.json"
    payload = []
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    payload = loaded
        except Exception:
            payload = []
    payload.append(entry)
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2)


def _clamp_role_token_limits(settings) -> None:
    role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
    for role in role_names:
        model_attr = f"{role}_model"
        max_attr = f"{role}_max_tokens"
        model_name = getattr(settings.models, model_attr, "")
        configured = int(getattr(settings.models, max_attr, 0) or 0)
        safe_cap = int(settings.auto_output_token_limit(model_name=model_name))
        if configured <= 0:
            setattr(settings.models, max_attr, safe_cap)
        else:
            setattr(settings.models, max_attr, min(configured, safe_cap))


def _strip_provider_prefix(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value:
        return ""
    if "/" in value:
        return value.split("/", 1)[1].strip()
    return value


def _activate_openai_fallback(settings, reason: str, ui=None) -> None:
    settings.models.backend = LLMBackend.OPENAI
    role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
    settings.models.public_model_name = _strip_provider_prefix(settings.models.public_model_name) or "gpt-5-nano"
    for role in role_names:
        role_attr = f"{role}_model"
        current = getattr(settings.models, role_attr, "")
        setattr(settings.models, role_attr, _strip_provider_prefix(current) or settings.models.public_model_name)
    reset_llm_client()
    msg = f"OpenRouter unavailable. Falling back to OpenAI backend ({reason})."
    logger.warning(msg)
    print(f"[Init] {msg}")
    if ui is not None:
        try:
            ui.set_status("OpenRouter unavailable. Fallback to OpenAI backend.", stage=0)
        except Exception:
            pass


def _apply_role_model_overrides(settings, role_models: Dict[str, Any]) -> None:
    if not isinstance(role_models, dict):
        return
    role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
    for role in role_names:
        value = role_models.get(role)
        if value:
            setattr(settings.models, f"{role}_model", str(value))


def _apply_role_max_token_overrides(settings, role_max_tokens: Dict[str, Any]) -> None:
    if not isinstance(role_max_tokens, dict):
        return
    role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
    for role in role_names:
        value = role_max_tokens.get(role)
        if value in (None, ""):
            continue
        setattr(settings.models, f"{role}_max_tokens", int(value))

def _compute_token_budget_defaults(context_window: int) -> Dict[str, int]:
    ctx = max(1, int(context_window or 0))
    return {
        "max_agent_input": int(ctx * 0.95),
        "max_agent_output": int(ctx * 0.25),
        "max_tool_input": int(ctx * 0.40),
        "max_tool_output": int(ctx * 0.25),
    }


def _apply_token_budget_defaults(settings, overrides: Dict[str, Any]) -> None:
    defaults = _compute_token_budget_defaults(settings.effective_context_window())
    if overrides.get("max_agent_input") in (None, "", 0):
        settings.token_budget.max_agent_input_tokens = defaults["max_agent_input"]
    if overrides.get("max_agent_output") in (None, "", 0):
        settings.token_budget.max_agent_output_tokens = defaults["max_agent_output"]
    if overrides.get("max_tool_input") in (None, "", 0):
        settings.token_budget.max_tool_input_tokens = defaults["max_tool_input"]
    if overrides.get("max_tool_output") in (None, "", 0):
        settings.token_budget.max_tool_output_tokens = defaults["max_tool_output"]


def _parse_xai_methods(raw_methods: Optional[str]) -> List[str]:
    """Parse comma-separated XAI methods with support for `all` alias."""
    if raw_methods is None:
        return []
    text = str(raw_methods).strip().lower()
    if not text:
        return []
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    if not parts:
        return []
    valid = {"external", "internal", "hybrid"}
    if "all" in parts:
        parts = ["external", "internal", "hybrid"]
    invalid = sorted([p for p in parts if p not in valid])
    if invalid:
        raise ValueError(f"Invalid --xai_methods entries: {', '.join(invalid)}")
    # preserve order, remove duplicates
    dedup: List[str] = []
    for p in parts:
        if p not in dedup:
            dedup.append(p)
    return dedup


def _apply_explainability_overrides(settings, args: argparse.Namespace) -> None:
    methods = _parse_xai_methods(getattr(args, "xai_methods", None))
    settings.explainability.methods = methods
    settings.explainability.enabled = bool(methods)
    settings.explainability.run_full_validation = bool(getattr(args, "xai_full_validation", False))

    if getattr(args, "xai_external_k", None) is not None:
        settings.explainability.external_k = max(1, int(args.xai_external_k))
    if getattr(args, "xai_external_runs", None) is not None:
        settings.explainability.external_runs = max(1, int(args.xai_external_runs))
    if getattr(args, "xai_external_adaptive", None) is not None:
        settings.explainability.external_adaptive = bool(args.xai_external_adaptive)

    if getattr(args, "xai_internal_model", None):
        settings.explainability.internal_model = str(args.xai_internal_model)
    if getattr(args, "xai_internal_steps", None) is not None:
        settings.explainability.internal_steps = max(1, int(args.xai_internal_steps))
    if getattr(args, "xai_internal_baseline", None):
        settings.explainability.internal_baseline_mode = str(args.xai_internal_baseline)
    if getattr(args, "xai_internal_span_mode", None):
        settings.explainability.internal_span_mode = str(args.xai_internal_span_mode)

    if getattr(args, "xai_hybrid_model", None):
        settings.explainability.hybrid_model = str(args.xai_hybrid_model)
    if getattr(args, "xai_hybrid_repeats", None) is not None:
        settings.explainability.hybrid_repeats = max(1, int(args.xai_hybrid_repeats))
    if getattr(args, "xai_hybrid_temperature", None) is not None:
        settings.explainability.hybrid_temperature = float(args.xai_hybrid_temperature)


def _generate_deep_phenotype_report(
    *,
    communicator: Communicator,
    prediction: Any,
    evaluation: Any,
    executor_output: Dict[str, Any],
    data_overview: Dict[str, Any],
    execution_summary: Dict[str, Any],
    control_condition: str,
    report_context_note: str,
    base_output_dir: Path,
    participant_id: str,
    user_focus_modalities: str = "",
    user_general_instruction: str = "",
    trigger_source: str = "manual",
    ui_step_id: Optional[int] = None,
    interactive_ui: bool = False,
) -> Dict[str, Any]:
    ui = get_ui()
    if interactive_ui:
        ui.set_status("Generating deep phenotype report...", stage=6)
        ui.on_step_start(
            step_id=ui_step_id or 930,
            tool_name="Communicator Agent",
            description="Generating deep phenotype report...",
            stage=6,
        )
    try:
        deep_report = communicator.execute(
            prediction=prediction,
            evaluation=evaluation,
            executor_output=executor_output,
            data_overview=data_overview,
            execution_summary=execution_summary,
            report_context_note=report_context_note,
            control_condition=control_condition,
            user_focus_modalities=user_focus_modalities,
            user_general_instruction=user_general_instruction,
            status_callback=(lambda msg: ui.set_status(msg, stage=6)) if interactive_ui else None,
        )
        deep_path = base_output_dir / "deep_phenotype.md"
        with open(deep_path, "w") as f:
            f.write(deep_report)

        metadata = dict(getattr(communicator, "last_run_metadata", {}) or {})
        metadata.update(
            {
                "trigger_source": trigger_source,
                "user_focus_modalities_present": bool(str(user_focus_modalities or "").strip()),
                "user_general_instruction_present": bool(str(user_general_instruction or "").strip()),
                "output_path": str(deep_path),
            }
        )
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "COMMUNICATOR",
                "timestamp": datetime.now().isoformat(),
                "data": metadata,
            },
        )
        perf_path = base_output_dir / f"performance_report_{participant_id}.json"
        if perf_path.exists():
            try:
                with open(perf_path, "r") as f:
                    perf = json.load(f)
                perf["deep_phenotype"] = {
                    "generated": True,
                    "path": str(deep_path),
                    "trigger_source": trigger_source,
                    "metadata": metadata,
                }
                with open(perf_path, "w") as f:
                    json.dump(perf, f, indent=2)
            except Exception:
                pass

        if interactive_ui:
            ui.on_step_complete(
                step_id=ui_step_id or 930,
                tokens=0,
                duration_ms=0,
                preview="Deep phenotype report generated.",
            )
        print(f"[Communicator] Saved to: {deep_path}")
        return {"success": True, "path": str(deep_path), "metadata": metadata}
    except Exception as e:
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "COMMUNICATOR",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "trigger_source": trigger_source,
                    "success": False,
                    "error": str(e),
                    "user_focus_modalities_present": bool(str(user_focus_modalities or "").strip()),
                    "user_general_instruction_present": bool(str(user_general_instruction or "").strip()),
                },
            },
        )
        if interactive_ui:
            ui.on_step_failed(step_id=ui_step_id or 930, error=str(e))
        return {"success": False, "error": str(e), "metadata": {}}


def _generate_xai_explainability_report(
    *,
    communicator: Communicator,
    xai_result: Dict[str, Any],
    prediction: Any,
    evaluation: Any,
    execution_summary: Dict[str, Any],
    target_condition: str,
    control_condition: str,
    base_output_dir: Path,
    participant_id: str,
    trigger_source: str = "cli",
    ui_step_id: Optional[int] = None,
    interactive_ui: bool = False,
) -> Dict[str, Any]:
    ui = get_ui()
    methods = dict((xai_result or {}).get("methods") or {})
    successful_methods = sorted(
        [name for name, payload in methods.items() if (payload or {}).get("status") == "success"]
    )

    if not successful_methods:
        reason = "No successful explainability method outputs available; skipping XAI report."
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "XAI_REPORT",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "trigger_source": trigger_source,
                    "success": False,
                    "skipped": True,
                    "reason": reason,
                    "methods_requested": list((xai_result or {}).get("methods_requested") or []),
                },
            },
        )
        return {"success": False, "skipped": True, "reason": reason, "metadata": {}}

    if interactive_ui:
        ui.set_status("Generating explainability report...", stage=6)
        ui.on_step_start(
            step_id=ui_step_id or 995,
            tool_name="Communicator Agent",
            description="Generating explainability report...",
            stage=6,
        )

    try:
        xai_report = communicator.execute_xai_report(
            xai_result=xai_result,
            prediction=prediction,
            evaluation=evaluation,
            execution_summary=execution_summary,
            target_condition=target_condition,
            control_condition=control_condition,
            status_callback=(lambda msg: ui.set_status(msg, stage=6)) if interactive_ui else None,
        )
        xai_path = base_output_dir / "xai_explainability_report.md"
        with open(xai_path, "w") as f:
            f.write(xai_report)

        metadata = dict(getattr(communicator, "last_run_metadata", {}) or {})
        metadata.update(
            {
                "trigger_source": trigger_source,
                "output_path": str(xai_path),
                "methods_requested": list((xai_result or {}).get("methods_requested") or []),
                "methods_successful": successful_methods,
            }
        )
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "XAI_REPORT",
                "timestamp": datetime.now().isoformat(),
                "data": metadata,
            },
        )

        if interactive_ui:
            ui.on_step_complete(
                step_id=ui_step_id or 995,
                tokens=0,
                duration_ms=0,
                preview="XAI explainability report generated.",
            )
        print(f"[Communicator] XAI report saved to: {xai_path}")
        return {"success": True, "path": str(xai_path), "metadata": metadata}
    except Exception as e:
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "XAI_REPORT",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "trigger_source": trigger_source,
                    "success": False,
                    "error": str(e),
                    "methods_requested": list((xai_result or {}).get("methods_requested") or []),
                    "methods_successful": successful_methods,
                },
            },
        )
        if interactive_ui:
            ui.on_step_failed(step_id=ui_step_id or 995, error=str(e))
        return {"success": False, "error": str(e), "metadata": {}}


def run_compass_pipeline(
    participant_dir: Path,
    target_condition: str,
    control_condition: str = DEFAULT_CONTROL_CONDITION,
    max_iterations: int = 3,
    verbose: bool = True,
    interactive_ui: bool = False,
    generate_deep_phenotype: bool = False,
    generate_xai_report: bool = False,
    deep_report_focus_modalities: str = "",
    deep_report_general_instruction: str = "",
) -> dict:
    """
    Run the complete COMPASS pipeline for a participant.
    
    Args:
        participant_dir: Path to participant data directory
        target_condition: Target phenotype string
        control_condition: Control comparator string
        max_iterations: Maximum orchestration iterations
        verbose: Enable verbose output
    
    Returns:
        Dictionary with prediction result and metadata
    """
    settings = get_settings()
    start_time = datetime.now()
    
    # Initialize UI
    ui = get_ui(enabled=interactive_ui)
    
    # Initialize components
    if interactive_ui:
        ui.on_pipeline_start(
            participant_id=participant_dir.name,
            target=target_condition,
            control=control_condition,
            participant_dir=str(participant_dir),
            max_iterations=max_iterations
        )
    else:
        print("\n" + "=" * 70)
        print("  COMPASS - Clinical Ontology-driven Multi-modal Predictive Agentic Support System")
        print("=" * 70)
    
    # Load participant data
    if interactive_ui: ui.set_status("Loading Participant Data...", stage=0)
    print(f"\n[1/5] Loading participant data from: {participant_dir}")
    data_loader = DataLoader()
    participant_data = data_loader.load(participant_dir)
    participant_id_raw = str(getattr(participant_data, "participant_id", "") or "").strip()
    if not participant_id_raw or participant_id_raw.lower() == "unknown":
        participant_id = participant_dir.name
        try:
            participant_data.participant_id = participant_id
        except Exception:
            pass
    else:
        participant_id = participant_id_raw

    # Build explainability feature space once from loaded artifacts.
    multimodal_for_xai = getattr(participant_data.multimodal_data, "features", {}) or {}
    try:
        raw_multimodal_path = (participant_data.raw_files or {}).get("multimodal_data")
        if raw_multimodal_path and Path(raw_multimodal_path).exists():
            with open(raw_multimodal_path, "r") as f:
                multimodal_for_xai = json.load(f)
    except Exception:
        multimodal_for_xai = getattr(participant_data.multimodal_data, "features", {}) or {}

    non_numerical_for_xai = str(
        getattr(participant_data.non_numerical_data, "raw_text", "") or ""
    )
    feature_space = build_feature_space(
        multimodal_for_xai,
        non_numerical_text=non_numerical_for_xai,
    )
    
    # Initialize logging
    exec_logger = ExecutionLogger(participant_id, verbose=verbose)
    decision_trace = DecisionTrace(participant_id)
    exec_logger.log_pipeline_start(target_condition, control_condition)
    
    # Initialize token manager
    token_manager = TokenManager()
    
    # Preflight connectivity check (public API backend)
    if settings.models.backend in (LLMBackend.OPENAI, LLMBackend.OPENROUTER):
        provider_label = "OpenRouter" if settings.models.backend == LLMBackend.OPENROUTER else "OpenAI"
        if interactive_ui:
            ui.set_status(f"Checking {provider_label} connectivity...", stage=0)
        try:
            get_llm_client().ping()
        except Exception as e:
            if settings.models.backend == LLMBackend.OPENROUTER and settings.openai_api_key:
                _activate_openai_fallback(settings, str(e), ui=ui if interactive_ui else None)
                if interactive_ui:
                    ui.set_status("Checking OpenAI fallback connectivity...", stage=0)
                try:
                    get_llm_client().ping()
                except Exception as fallback_error:
                    raise RuntimeError(
                        "OpenRouter connectivity failed and OpenAI fallback also failed. "
                        "Verify network access and API keys."
                    ) from fallback_error
            else:
                key_name = "OPENROUTER_API_KEY" if settings.models.backend == LLMBackend.OPENROUTER else "OPENAI_API_KEY"
                raise RuntimeError(
                    f"{provider_label} connectivity check failed. Verify network access and {key_name}."
                ) from e
    elif settings.models.backend == LLMBackend.LOCAL:
        if interactive_ui:
            ui.set_status("Initializing local model...", stage=0)
        try:
            get_llm_client()
        except Exception as e:
            raise RuntimeError(
                f"Local backend initialization failed: {e}"
            ) from e

    # Initialize agents
    if interactive_ui: ui.set_status("Initializing Agents...", stage=0)
    print(f"\n[2/5] Initializing COMPASS agents...")
    orchestrator = Orchestrator(token_manager=token_manager)
    executor = Executor(token_manager=token_manager)
    predictor = Predictor(token_manager=token_manager)
    critic = Critic(token_manager=token_manager)
    communicator = Communicator(token_manager=token_manager)
    
    # Main loop: Orchestrator -> Executor -> Predictor -> Critic
    iteration = 1
    previous_feedback = None
    final_prediction = None
    final_evaluation = None
    final_executor_output = None
    final_plan = None
    attempts: List[Dict[str, Any]] = []
    
    while iteration <= max_iterations:
        print(f"\n{'='*70}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'='*70}")
        
        # Step 3: Orchestrator creates plan
        if interactive_ui: 
            ui.set_status("Orchestrator creating execution plan...", stage=1, iteration=iteration)
        print(f"\n[3/5] Orchestrator creating execution plan...")
        plan = orchestrator.execute(
            participant_data=participant_data,
            target_condition=target_condition,
            control_condition=control_condition,
            previous_feedback=previous_feedback,
            iteration=iteration
        )
        
        exec_logger.log_orchestrator({
            "plan_id": plan.plan_id,
            "total_steps": plan.total_steps,
            "priority_domains": plan.priority_domains
        })
        
        decision_trace.record_orchestrator_plan(
            domains=plan.priority_domains,
            num_steps=plan.total_steps,
            reasoning=plan.reasoning[:500]
        )
        
        # Step 4: Executor runs plan
        print(f"\n[4/5] Executor processing plan...")
        executor_output = executor.execute(
            plan=plan,
            participant_data=participant_data,
            target_condition=target_condition,
            control_condition=control_condition,
        )
        final_executor_output = executor_output
        final_plan = plan
        
        # Log each step
        exec_result = executor_output.get("execution_result")
        if exec_result and hasattr(exec_result, 'step_statuses'):
            for step_id, status in exec_result.step_statuses.items():
                exec_logger.log_executor_step(
                    step_id=step_id,
                    tool_name=status.get("tool_name", "unknown"),
                    status=status.get("status", "UNKNOWN"),
                    tokens=status.get("tokens", 0)
                )
        
        # Send fused input to UI for inspection
        # This event sets status to "Fusion Complete", so we must set prediction status AFTER it
        if interactive_ui and "predictor_input" in executor_output:
            ui.on_fusion_complete(executor_output["predictor_input"])

        # Step 5: Predictor makes prediction
        if interactive_ui: ui.set_status("Predictor evaluating CASE vs CONTROL...", stage=4)
        if interactive_ui:
            ui.on_step_start(
                step_id=910 + iteration,
                tool_name="Predictor Agent",
                description="Evaluating integrated evidence for final CASE/CONTROL prediction...",
                stage=4,
            )

        prediction = predictor.execute(
            executor_output=executor_output,
            target_condition=target_condition,
            control_condition=control_condition,
            iteration=iteration
        )

        dataflow_summary = _build_dataflow_summary(
            executor_output=executor_output,
            target_condition=target_condition,
            control_condition=control_condition,
            iteration=iteration,
        )
        executor_output["dataflow_summary"] = dataflow_summary
        exec_logger.log_dataflow_summary(dataflow_summary, iteration=iteration)
        
        exec_logger.log_predictor({
            "classification": prediction.binary_classification.value,
            "probability": prediction.probability_score
        })
        
        if interactive_ui:
            ui.on_prediction(
                classification=prediction.binary_classification.value,
                probability=prediction.probability_score,
                confidence=prediction.confidence_level.value
            )
        
        decision_trace.record_prediction(
            classification=prediction.binary_classification.value,
            probability=prediction.probability_score,
            key_findings=[f.finding for f in prediction.key_findings[:3]],
            reasoning=prediction.clinical_summary[:500]
        )
        
        # Step 6: Critic evaluates
        if interactive_ui: ui.set_status("Critic Evaluating...", stage=5)
        print(f"\n[6/6] Critic evaluating prediction...")
        # Pass FULL data overview as dictionary (User Requirement)
        data_overview_dict = participant_data.data_overview.model_dump()
        
        evaluation = critic.execute(
            prediction=prediction,
            executor_output=executor_output,
            data_overview=data_overview_dict,
            hierarchical_deviation=participant_data.hierarchical_deviation.model_dump(),
            non_numerical_data=participant_data.non_numerical_data.raw_text,
            control_condition=control_condition,
        )
        
        exec_logger.log_critic({
            "verdict": evaluation.verdict.value,
            "confidence": evaluation.confidence_in_verdict
        })
        
        decision_trace.record_critic_verdict(
            verdict=evaluation.verdict.value,
            checklist_passed=evaluation.checklist.pass_count,
            checklist_total=7,
            reasoning=evaluation.reasoning[:500]
        )

        if interactive_ui:
            if hasattr(evaluation.checklist, "model_dump"):
                checklist_data = evaluation.checklist.model_dump()
            elif hasattr(evaluation.checklist, "dict"):
                checklist_data = evaluation.checklist.dict()
            else:
                checklist_data = {}
            
            improvements = []
            for s in evaluation.improvement_suggestions[:5]:
                if hasattr(s, "model_dump"):
                    improvements.append(s.model_dump())
                elif hasattr(s, "dict"):
                    improvements.append(s.dict())
                else:
                    improvements.append({
                        "issue": getattr(s, "issue", ""),
                        "suggestion": getattr(s, "suggestion", ""),
                        "priority": getattr(s, "priority", "")
                    })
            
            ui.on_critic_verdict(
                verdict=evaluation.verdict.value,
                confidence=evaluation.confidence_in_verdict,
                checklist_passed=evaluation.checklist.pass_count,
                checklist_total=7,
                summary=evaluation.concise_summary or evaluation.reasoning[:240],
                checklist=checklist_data,
                weaknesses=evaluation.weaknesses[:5],
                improvement_suggestions=improvements,
                domains_missed=evaluation.domains_missed[:5],
                composite_score=evaluation.composite_score,
                score_breakdown=evaluation.score_breakdown,
                iteration=iteration
            )
        
        final_prediction = prediction
        final_evaluation = evaluation
        attempts.append(
            {
                "iteration": iteration,
                "prediction": prediction,
                "evaluation": evaluation,
                "executor_output": executor_output,
                "plan": plan,
            }
        )
        
        # Check if satisfactory / decide on re-orchestration
        if evaluation.verdict == Verdict.SATISFACTORY:
            print(f"\n✓ Prediction deemed SATISFACTORY by Critic")
            break

        print(f"\n✗ Prediction deemed UNSATISFACTORY by Critic")
        if iteration >= max_iterations:
            # Final attempt reached; do not increment `iteration` (keeps accurate count for reports/UI).
            break

        print(f"  Re-orchestrating with critic feedback...")
        previous_feedback = _format_feedback(evaluation)
        iteration += 1

    selected_attempt, selection_reason = _select_best_attempt(attempts)
    if not selected_attempt:
        raise RuntimeError("No prediction attempts were generated by the pipeline.")

    selected_iteration = int(selected_attempt["iteration"])
    final_prediction = selected_attempt["prediction"]
    final_evaluation = selected_attempt["evaluation"]
    final_executor_output = selected_attempt["executor_output"]
    final_plan = selected_attempt["plan"]
    coverage_summary = (
        final_executor_output.get("coverage_summary")
        or (final_executor_output.get("coverage_ledger") or {}).get("summary")
        or {}
    )
    
    # Prepare output directory and token usage
    token_usage = token_manager.get_detailed_usage()
    base_output_dir = _resolve_output_dir(participant_dir, participant_id, settings)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    explainability_result = _run_explainability_for_selected_attempt(
        settings=settings,
        participant_id=participant_id,
        target_condition=target_condition,
        control_condition=control_condition,
        selected_attempt=selected_attempt,
        feature_space=feature_space,
        output_dir=base_output_dir,
        exec_logger=exec_logger,
    )

    # Generate final report (standard outputs first)
    print(f"\n{'='*70}")
    print(f"  GENERATING FINAL REPORT")
    print(f"{'='*70}")
    
    report_generator = PatientReportGenerator()
    
    # Collect detailed logs from Logger or Trace? 
    # Actually, we need to collect them from the executor results if they are stored there.
    # But executor returns the result of the LAST iteration.
    # We should rely on `exec_logger` to track them across all steps?
    # For now, let's grab them from the final executor_output if available, or just empty list.
    detailed_logs_collection = []
    # (Implementation Note: Ideally we'd aggregate them properly. Basic placeholder for now.)
    
    execution_summary = {
        "iterations": len(attempts),
        "selected_iteration": selected_iteration,
        "selection_reason": selection_reason,
        "coverage_summary": coverage_summary,
        "dataflow_summary": (final_executor_output or {}).get("dataflow_summary", {}),
        "target_condition": target_condition,
        "control_condition": control_condition,
        "tokens_used": token_usage.get("total_tokens", 0),
        "domains_processed": (final_plan.priority_domains if final_plan else plan.priority_domains),
        "detailed_logs": detailed_logs_collection, # We need to populate this
        "explainability": {
            "enabled": bool(explainability_result.get("enabled")),
            "status": explainability_result.get("status"),
            "methods_requested": explainability_result.get("methods_requested") or [],
            "artifact_path": explainability_result.get("artifact_path"),
        },
    }
    
    report = report_generator.generate(
        participant_id=participant_id,
        prediction=final_prediction,
        evaluation=final_evaluation,
        execution_summary=execution_summary,
        decision_trace=decision_trace.get_trace()
    )
    
    # Save outputs to configured output directory
    
    report_generator.save(report, base_output_dir)
    report_generator.save_markdown(report, base_output_dir)
    exec_logger.save_structured_log(base_output_dir / f"execution_log_{participant_id}.json")

    # Log completion duration early so standard reports can include it
    duration_so_far = (datetime.now() - start_time).total_seconds()

    # Generate Performance Report
    performance_report = {
        "participant_id": participant_id,
        "target_condition": target_condition,
        "control_condition": control_condition,
        "execution_timestamp": start_time.isoformat(),
        "total_duration_seconds": round(duration_so_far, 2),
        "iterations": len(attempts),
        "selected_iteration": selected_iteration,
        "selection_reason": selection_reason,
        "coverage_summary": coverage_summary,
        "dataflow_summary": (final_executor_output or {}).get("dataflow_summary", {}),
        "prediction_result": {
            "classification": final_prediction.binary_classification.value,
            "probability": round(final_prediction.probability_score, 4),
            "confidence": final_prediction.confidence_level.value
        },
        "control_condition": control_condition,
        "critic_verdict": final_evaluation.verdict.value,
        "token_usage": {
            "total_tokens": token_usage.get("total_tokens", 0),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "calls": token_usage.get("calls", [])
        },
        "plan_summary": {
            "plan_id": final_plan.plan_id if final_plan else plan.plan_id,
            "total_steps": final_plan.total_steps if final_plan else plan.total_steps,
            "priority_domains": final_plan.priority_domains if final_plan else plan.priority_domains,
        },
        "explainability": explainability_result,
    }
    
    # Save performance report as JSON
    import json
    performance_report_path = base_output_dir / f"performance_report_{participant_id}.json"
    with open(performance_report_path, 'w') as f:
        json.dump(performance_report, f, indent=2)

    data_overview_dict = participant_data.data_overview.model_dump()
    report_context_note = _build_report_context_note(
        final_evaluation=final_evaluation,
        selected_iteration=selected_iteration,
        selection_reason=selection_reason,
    )
    xai_report_result = {"success": False, "metadata": {}, "path": None, "skipped": False}
    if generate_xai_report:
        xai_report_result = _generate_xai_explainability_report(
            communicator=communicator,
            xai_result=explainability_result,
            prediction=final_prediction,
            evaluation=final_evaluation,
            execution_summary=execution_summary,
            target_condition=target_condition,
            control_condition=control_condition,
            base_output_dir=base_output_dir,
            participant_id=participant_id,
            trigger_source="cli",
            ui_step_id=980 + iteration,
            interactive_ui=interactive_ui,
        )

    deep_report_result = {"success": False, "metadata": {}, "path": None}
    if generate_deep_phenotype and final_prediction and final_evaluation and final_executor_output:
        deep_report_result = _generate_deep_phenotype_report(
            communicator=communicator,
            prediction=final_prediction,
            evaluation=final_evaluation,
            executor_output=final_executor_output,
            data_overview=data_overview_dict,
            execution_summary=execution_summary,
            control_condition=control_condition,
            report_context_note=report_context_note,
            base_output_dir=base_output_dir,
            participant_id=participant_id,
            user_focus_modalities=deep_report_focus_modalities,
            user_general_instruction=deep_report_general_instruction,
            trigger_source="cli",
            ui_step_id=930 + iteration,
            interactive_ui=interactive_ui,
        )
    performance_report["xai_report"] = {
        "generated": bool(xai_report_result.get("success")),
        "path": xai_report_result.get("path"),
        "trigger_source": "cli" if generate_xai_report else None,
        "skipped": bool(xai_report_result.get("skipped", False)),
        "reason": xai_report_result.get("reason"),
        "metadata": xai_report_result.get("metadata") or {},
    }
    performance_report["deep_phenotype"] = {
        "generated": bool(deep_report_result.get("success")),
        "path": deep_report_result.get("path"),
        "trigger_source": "cli" if generate_deep_phenotype else None,
        "metadata": deep_report_result.get("metadata") or {},
    }
    with open(performance_report_path, 'w') as f:
        json.dump(performance_report, f, indent=2)

    # Log completion
    duration = (datetime.now() - start_time).total_seconds()
    if abs(duration - duration_so_far) > 0.05:
        performance_report["total_duration_seconds"] = round(duration, 2)
        with open(performance_report_path, 'w') as f:
            json.dump(performance_report, f, indent=2)

    exec_logger.log_pipeline_end(
        success=True,
        summary={
            "prediction": final_prediction.binary_classification.value,
            "probability": final_prediction.probability_score,
            "iterations": len(attempts),
            "duration_seconds": duration
        }
    )

    if interactive_ui:
        ui.on_pipeline_complete(
            result=final_prediction.binary_classification.value,
            probability=final_prediction.probability_score,
            iterations=len(attempts),
            total_duration_secs=duration,
            total_tokens=token_usage.get("total_tokens", 0)
        )
    
    print(f"\n{'='*70}")
    print(f"  COMPASS PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Participant: {participant_id}")
    print(f"  Prediction: {final_prediction.binary_classification.value}")
    print(f"  Probability: {final_prediction.probability_score:.1%}")
    print(f"  Iterations: {len(attempts)} (selected iteration {selected_iteration})")
    print(f"  Selection reason: {selection_reason}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Total Tokens: {token_usage.get('total_tokens', 0)}")
    print(f"  Output: {base_output_dir}")
    print(f"{'='*70}\n")
    
    return {
        "participant_id": participant_id,
        "prediction": final_prediction.binary_classification.value,
        "probability": final_prediction.probability_score,
        "confidence": final_prediction.confidence_level.value,
        "verdict": final_evaluation.verdict.value,
        "iterations": len(attempts),
        "selected_iteration": selected_iteration,
        "selection_reason": selection_reason,
        "control_condition": control_condition,
        "coverage_summary": coverage_summary,
        "duration_seconds": duration,
        "output_dir": str(base_output_dir),
        "report": report,
        "deep_phenotype_generated": bool(deep_report_result.get("success")),
        "deep_phenotype_path": deep_report_result.get("path"),
        "xai_report_generated": bool(xai_report_result.get("success")),
        "xai_report_path": xai_report_result.get("path"),
        "explainability": explainability_result,
        "internal_context": {
            "participant_id": participant_id,
            "prediction": final_prediction,
            "evaluation": final_evaluation,
            "executor_output": final_executor_output,
            "data_overview": data_overview_dict,
            "execution_summary": execution_summary,
            "control_condition": control_condition,
            "report_context_note": report_context_note,
            "base_output_dir": str(base_output_dir),
            "explainability": explainability_result,
            "xai_report": xai_report_result,
        },
    }


def run_dataflow_audit(
    participant_dir: Path,
    target_condition: str,
    control_condition: str = DEFAULT_CONTROL_CONDITION,
    verbose: bool = True,
) -> dict:
    """
    Offline dataflow audit: build predictor payload and chunking without LLM calls.
    """
    settings = get_settings()
    data_loader = DataLoader()
    participant_data = data_loader.load(participant_dir)

    token_manager = TokenManager()
    executor = Executor(token_manager=token_manager)
    context = executor._build_context(participant_data, target_condition, control_condition)

    fusion_layer = FusionLayer()
    pass_through = FusionResult(
        fused_narrative="Audit pass-through",
        domain_summaries={},
        key_findings=[],
        cross_modal_patterns=[],
        evidence_summary={"for_case": [], "for_control": []},
        tokens_used=0,
        source_outputs=[],
        skipped_fusion=True,
        raw_multimodal_data=context.get("multimodal_data") or {},
        raw_processed_multimodal_data=None,
        raw_step_outputs={},
        context_fill_report={"audit": True},
    )
    predictor_input = fusion_layer.compress_for_predictor(
        fusion_result=pass_through,
        hierarchical_deviation=context.get("hierarchical_deviation") or {},
        non_numerical_data=context.get("non_numerical_data") or "",
    )

    coverage_ledger = executor._build_coverage_ledger(
        multimodal_data=context.get("multimodal_data") or {},
        step_outputs={},
        predictor_input=predictor_input,
    )
    predictor_input["coverage_ledger"] = coverage_ledger

    max_tool_input = int(getattr(settings.token_budget, "max_tool_input_tokens", 20000) or 20000)
    chunk_budget = max(30000, min(60000, int(max_tool_input * 2.0)))
    assembler = PredictorInputAssembler(max_chunk_tokens=chunk_budget, model_hint=settings.models.tool_model)
    executor_stub = {
        "step_outputs": {},
        "data_overview": context.get("data_overview") or {},
        "hierarchical_deviation": context.get("hierarchical_deviation") or {},
        "non_numerical_data": context.get("non_numerical_data") or "",
    }
    sections = assembler.build_sections(
        executor_output=executor_stub,
        predictor_input=predictor_input,
        coverage_ledger=coverage_ledger,
    )
    core_names = {
        "non_numerical_data_raw",
        "hierarchical_deviation_raw",
        "data_overview",
        "phenotype_representation",
        "feature_synthesizer",
        "differential_diagnosis",
    }
    def _is_core(name: str) -> bool:
        base = name.split("#", 1)[0]
        return base in core_names
    chunk_sections = [s for s in sections if not _is_core(s.name)]
    chunks = assembler.build_chunks(chunk_sections)

    section_stats = []
    for sec in sections:
        section_stats.append(
            {
                "name": sec.name,
                "tokens": count_tokens(sec.text, model_hint=settings.models.tool_model),
                "feature_key_count": len(sec.feature_keys),
            }
        )

    chunk_stats = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_text = assembler.chunk_to_text(chunk, idx, len(chunks))
        chunk_stats.append(
            {
                "chunk_index": idx,
                "sections": [s.name for s in chunk],
                "tokens": count_tokens(chunk_text, model_hint=settings.models.tool_model),
            }
        )

    try:
        payload_tokens = len(
            fusion_layer.encoder.encode(json.dumps(predictor_input, default=str))
        )
    except Exception:
        payload_tokens = count_tokens(str(predictor_input), model_hint=settings.models.tool_model)

    report = {
        "participant_id": participant_data.participant_id,
        "target_condition": target_condition,
        "control_condition": control_condition,
        "coverage_summary": coverage_ledger.get("summary", {}),
        "predictor_payload_tokens": payload_tokens,
        "chunk_budget_tokens": chunk_budget,
        "chunk_count": len(chunks),
        "section_stats": section_stats,
        "chunk_stats": chunk_stats,
        "predictor_input_mode": predictor_input.get("mode"),
    }

    output_dir = _resolve_output_dir(participant_dir, participant_data.participant_id, settings)
    output_path = output_dir / f"dataflow_audit_{participant_data.participant_id}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        fallback_dir = settings.paths.logs_dir
        fallback_dir.mkdir(parents=True, exist_ok=True)
        output_path = fallback_dir / f"dataflow_audit_{participant_data.participant_id}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    if verbose:
        print(f"[Audit] Dataflow audit saved: {output_path}")
        print(f"[Audit] Payload tokens: {payload_tokens}")
        print(f"[Audit] Chunk count: {len(chunks)} (budget {chunk_budget})")

    return report


def _format_feedback(evaluation) -> str:
    """Format critic evaluation as feedback for re-orchestration."""
    lines = [
        f"Previous prediction was deemed {evaluation.verdict.value}.",
        "",
        "Weaknesses identified:",
    ]
    
    for weakness in evaluation.weaknesses[:3]:
        lines.append(f"- {weakness}")
    
    lines.append("")
    lines.append("Suggested improvements:")
    
    for sugg in evaluation.high_priority_issues[:3]:
        lines.append(f"- [{sugg.priority.value}] {sugg.issue}: {sugg.suggestion}")
    
    if evaluation.domains_missed:
        lines.append("")
        lines.append(f"Domains missed: {', '.join(evaluation.domains_missed)}")
    
    return "\n".join(lines)


def _build_dataflow_summary(
    *,
    executor_output: Dict[str, Any],
    target_condition: str,
    control_condition: str,
    iteration: int,
) -> Dict[str, Any]:
    predictor_input = executor_output.get("predictor_input") or {}
    coverage_ledger = executor_output.get("coverage_ledger") or {}
    coverage_summary = (
        executor_output.get("coverage_summary")
        or coverage_ledger.get("summary")
        or {}
    )
    context_fill_report = predictor_input.get("context_fill_report") or {}
    chunk_evidence = executor_output.get("chunk_evidence") or []
    predictor_chunk_count = int(
        executor_output.get("predictor_chunk_count") or len(chunk_evidence) or 0
    )
    chunk_evidence_count = len(chunk_evidence)

    processed_raw_included = context_fill_report.get("processed_raw_full_included")
    processed_raw_present = bool(predictor_input.get("multimodal_processed_raw_low_priority"))

    missing_count = (
        coverage_summary.get("missing_feature_count")
        or coverage_summary.get("missing_count")
        or 0
    )
    invariant_ok = coverage_summary.get("invariant_ok")
    if invariant_ok is None:
        invariant_ok = int(missing_count) == 0

    assertions = {
        "invariant_ok": bool(invariant_ok),
        "missing_feature_count_zero": int(missing_count) == 0,
        "chunk_evidence_matches_count": (
            predictor_chunk_count == 0 or chunk_evidence_count == predictor_chunk_count
        ),
        "processed_raw_flag_consistent": (
            processed_raw_included is None or processed_raw_included == processed_raw_present
        ),
    }

    payload_estimate = context_fill_report.get("predictor_payload_estimate") or {}
    coverage_block = {
        "summary": coverage_summary,
        "forced_raw_count": len(coverage_ledger.get("forced_raw_features") or []),
    }
    chunking_block = {
        "predictor_chunk_count": predictor_chunk_count,
        "chunk_evidence_count": chunk_evidence_count,
        "chunked_two_pass_required": payload_estimate.get("chunked_two_pass_required"),
        "single_chunk_limit": payload_estimate.get("single_chunk_limit"),
        "chunking_skipped": bool(executor_output.get("chunking_skipped")),
        "chunking_reason": executor_output.get("chunking_reason"),
    }
    context_block = {
        "processed_raw_full_included": processed_raw_included,
        "rag_added_count": context_fill_report.get("added_count")
            or context_fill_report.get("top_added_count"),
        "predictor_payload_estimate": payload_estimate,
        "coverage_snapshot": context_fill_report.get("coverage"),
        "embedding_store": context_fill_report.get("embedding_store"),
    }

    return {
        "iteration": iteration,
        "target_condition": target_condition,
        "control_condition": control_condition,
        "predictor_input_mode": predictor_input.get("mode"),
        "coverage": coverage_block,
        "chunking": chunking_block,
        "context_fill": context_block,
        "assertions": assertions,
    }


def _run_explainability_for_selected_attempt(
    *,
    settings,
    participant_id: str,
    target_condition: str,
    control_condition: str,
    selected_attempt: Dict[str, Any],
    feature_space: Dict[str, Any],
    output_dir: Path,
    exec_logger: ExecutionLogger,
) -> Dict[str, Any]:
    if not bool(getattr(settings.explainability, "enabled", False)):
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "Explainability disabled.",
            "methods_requested": [],
            "methods": {},
        }
    try:
        result = run_explainability_methods(
            settings=settings,
            participant_id=participant_id,
            target_condition=target_condition,
            control_condition=control_condition,
            selected_attempt=selected_attempt,
            feature_space=feature_space,
            output_dir=output_dir,
        )
    except Exception as exc:
        result = {
            "enabled": True,
            "status": "failed",
            "reason": f"Runner failure: {exc}",
            "methods_requested": list(getattr(settings.explainability, "methods", []) or []),
            "methods": {},
        }

    try:
        exec_logger.log_explainability(result)
    except Exception:
        pass
    return result


def _select_best_attempt(attempts: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Select final attempt:
    1) Any SATISFACTORY verdict wins (highest composite/checklist among them).
    2) Otherwise choose highest critic composite/checklist score.
    """
    if not attempts:
        return None, "No attempts available"

    def _quality_tuple(item: Dict[str, Any]) -> Tuple[float, float, float]:
        evaluation = item.get("evaluation")
        if evaluation is None:
            return (0.0, 0.0, 0.0)
        composite = float(getattr(evaluation, "composite_score", 0.0) or 0.0)
        checklist = float(getattr(getattr(evaluation, "checklist", None), "pass_count", 0) or 0.0)
        confidence = float(getattr(evaluation, "confidence_in_verdict", 0.0) or 0.0)
        return (composite, checklist, confidence)

    satisfactory = [a for a in attempts if getattr(a.get("evaluation"), "verdict", None) == Verdict.SATISFACTORY]
    if satisfactory:
        selected = max(satisfactory, key=_quality_tuple)
        iteration = selected.get("iteration", "?")
        return selected, f"Satisfactory verdict available; chose strongest satisfactory attempt (iteration {iteration})."

    selected = max(attempts, key=_quality_tuple)
    score = _quality_tuple(selected)
    iteration = selected.get("iteration", "?")
    return (
        selected,
        "No satisfactory verdict; selected best unsatisfactory attempt by critic composite/checklist "
        f"(iteration {iteration}, composite={score[0]:.3f}, checklist={score[1]:.0f}).",
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="COMPASS Multi-Agent System for Disorder Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py /path/to/participant_001 --target neuropsychiatric
  python main.py /path/to/participant_001 --target neurologic --iterations 5
        """
    )
    
    parser.add_argument(
        "participant_dir",
        type=Path,
        nargs='?',
        help="Path to participant data directory (Optional if using --ui)"
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        # choices=["neuropsychiatric", "neurologic"], # Removed to allow dynamic targets
        default="neuropsychiatric",
        help="Target condition to predict (e.g. 'neuropsychiatric', or specific phenotype string)"
    )
    parser.add_argument(
        "--control", "-c",
        type=str,
        default=DEFAULT_CONTROL_CONDITION,
        help="Control condition comparator (default: brain-implicated pathology, but NOT psychiatric)"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=3,
        help="Maximum orchestration iterations (default: 3)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Enable interactive UI mode with detailed step-by-step monitoring"
    )

    parser.add_argument(
        "--detailed_log", "-d",
        action="store_true",
        help="Enable full raw I/O logging for all tool calls"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run offline dataflow audit without LLM calls"
    )
    parser.add_argument(
        "--generate_deep_phenotype",
        action="store_true",
        help="Generate deep phenotype report at pipeline completion (manual opt-in)"
    )
    parser.add_argument(
        "--generate_xai_report",
        action="store_true",
        help="Generate communicator explainability report from XAI outputs at pipeline completion"
    )

    # --- LOCAL LLM ARGUMENTS ---
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=["openrouter", "openai", "local"],
        default="openrouter",
        help="Choose LLM backend: 'openrouter' (default), 'openai', or 'local'"
    )
    parser.add_argument(
        "--public_model",
        type=str,
        default="gpt-5-nano",
        help="Model name for public API backend (default: gpt-5-nano)"
    )
    parser.add_argument(
        "--public_max_context_tokens",
        type=int,
        default=128000,
        help="Public API context window override used for thresholding (default: 128000)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model for retrieval/rag (default: text-embedding-3-large)"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Name/Path of local model (default: Qwen/Qwen2.5-0.5B-Instruct). Only used if --backend local"
    )
    
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=2048,
        help="Max context tokens for local model (default: 2048). Only used if --backend local"
    )

    parser.add_argument(
        "--local_engine",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="Local backend engine preference (auto|vllm|transformers)"
    )
    parser.add_argument(
        "--local_dtype",
        type=str,
        default="auto",
        help="Local dtype (auto|float16|bfloat16|float32|fp8)"
    )
    parser.add_argument(
        "--local_quant",
        type=str,
        default=None,
        help="Local quantization (e.g., awq|gptq|4bit|8bit|fp8)"
    )
    parser.add_argument(
        "--local_kv_cache_dtype",
        type=str,
        default=None,
        help="vLLM KV cache dtype (e.g., fp8_e4m3|fp8_e5m2)"
    )
    parser.add_argument(
        "--local_tensor_parallel",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default 1)"
    )
    parser.add_argument(
        "--local_pipeline_parallel",
        type=int,
        default=1,
        help="Pipeline parallel size for vLLM (default 1)"
    )
    parser.add_argument(
        "--local_gpu_mem_util",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default 0.9)"
    )
    parser.add_argument(
        "--local_max_model_len",
        type=int,
        default=0,
        help="Max model length override (0 = auto)"
    )
    parser.add_argument(
        "--local_enforce_eager",
        action="store_true",
        help="Force vLLM eager execution"
    )
    parser.add_argument(
        "--local_trust_remote_code",
        action="store_true",
        help="Trust remote code for local model"
    )
    parser.add_argument(
        "--local_attn",
        type=str,
        default="auto",
        help="Transformers attention implementation (auto|flash_attention_2|sdpa|eager)"
    )
    
    # --- TOKEN CONTROLS ---
    parser.add_argument(
        "--total_budget",
        type=int,
        help="Override total token budget"
    )
    parser.add_argument(
        "--max_agent_input",
        type=int,
        help="Max limit for agent input context (Prompt)"
    )
    parser.add_argument(
        "--max_agent_output",
        type=int,
        help="Max tokens for agent generation"
    )
    parser.add_argument(
        "--max_tool_input",
        type=int,
        help="Max limit for tool input size"
    )
    parser.add_argument(
        "--max_tool_output",
        type=int,
        help="Max limit for tool output size"
    )
    # --- XAI CONTROLS ---
    parser.add_argument(
        "--xai_methods",
        type=str,
        default="",
        help="Comma-separated methods: external,internal,hybrid (or 'all')"
    )
    parser.add_argument(
        "--xai_external_k",
        type=int,
        default=None,
        help="aHFR-TokenSHAP permutations (default from settings)"
    )
    parser.add_argument(
        "--xai_external_runs",
        type=int,
        default=None,
        help="aHFR-TokenSHAP repeat runs (default from settings)"
    )
    parser.add_argument(
        "--xai_external_adaptive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable adaptive sampling for external aHFR-TokenSHAP"
    )
    parser.add_argument(
        "--xai_internal_model",
        type=str,
        default=None,
        help="Internal IGA model (local HF model id/path)"
    )
    parser.add_argument(
        "--xai_internal_steps",
        type=int,
        default=None,
        help="Internal IGA integration steps"
    )
    parser.add_argument(
        "--xai_internal_baseline",
        type=str,
        default=None,
        help="Internal IGA baseline mode (mask|prompt|eos|zero)"
    )
    parser.add_argument(
        "--xai_internal_span_mode",
        type=str,
        default=None,
        help="Internal IGA span mode (value|line)"
    )
    parser.add_argument(
        "--xai_hybrid_model",
        type=str,
        default=None,
        help="Hybrid LLM-select model (default from settings)"
    )
    parser.add_argument(
        "--xai_hybrid_repeats",
        type=int,
        default=None,
        help="Hybrid LLM-select repeats"
    )
    parser.add_argument(
        "--xai_hybrid_temperature",
        type=float,
        default=None,
        help="Hybrid LLM-select temperature"
    )
    parser.add_argument(
        "--xai_full_validation",
        action="store_true",
        help="Enable stricter explainability validation checks"
    )
    # ---------------------------
    
    args = parser.parse_args()
    
    # Validate participant directory
    if not args.ui and not args.participant_dir:
        print("Error: participant_dir is required when not using --ui mode.")
        parser.print_help()
        sys.exit(1)

    if args.participant_dir and not args.participant_dir.exists():
        print(f"Error: Participant directory not found: {args.participant_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        if args.audit:
            run_dataflow_audit(
                participant_dir=args.participant_dir,
                target_condition=args.target,
                control_condition=args.control,
                verbose=not args.quiet,
            )
            sys.exit(0)
        if args.ui:
            # Run with GUI: Main thread -> UI, Background logic via callback
            
            # --- SMART DATA DISCOVERY ---
            def find_compass_data(start_path: Path) -> Path:
                """
                Locate COMPASS_data directory using heuristic scan.
                1. Check specific relative paths (fastest)
                2. Check specific common data folders in parents
                3. Shallow BFS scan of project tree (fallback)
                """
                search_target = "COMPASS_data"
                
                # S1: Check standard legacy paths
                candidates = [
                    start_path / "data" / "__FEATURES__" / search_target,  # Original
                    start_path.parent / "data" / "__FEATURES__" / search_target,
                    start_path / "data" / search_target,
                    start_path.parent / "data" / search_target,
                    start_path.parent.parent / "data" / search_target
                ]
                
                for cand in candidates:
                    if cand.exists() and cand.is_dir():
                        return cand

                # S2: Upward Search + Shallow Downward Scan
                # We go up to 3 levels to find a likely project root
                curr = start_path
                project_root = start_path
                for _ in range(3):
                    if (curr / search_target).exists(): return curr / search_target
                    if (curr / ".git").exists(): # Stop at git root
                        project_root = curr
                        break
                    if curr.parent == curr: break
                    curr = curr.parent
                    project_root = curr # Assume highest reachable is root if no .git
                
                # S3: Limited Scan from Project Root (Max Depth 3)
                print(f"[*] Scanning for '{search_target}' in {project_root}...")
                for path in project_root.rglob(search_target):
                    if path.is_dir():
                        if "node_modules" in str(path) or ".git" in str(path): continue
                        return path

                # Fallback: Return standard path even if missing (will be created or error later)
                return start_path.parent / "data" / "__FEATURES__" / "COMPASS_data"

            # Execute Discovery
            script_path = Path(__file__).parent
            compass_data_root = find_compass_data(script_path)

            if not compass_data_root.exists():
                # One last try check user argument
                if args.participant_dir and args.participant_dir.exists():
                     compass_data_root = args.participant_dir.parent
                else:    
                     logger.warning(f"[!] Could not auto-locate 'COMPASS_data'. Assumed default: {compass_data_root}")

            print(f"[*] Data Root: {compass_data_root}")
            latest_run_context: Dict[str, Any] = {"internal": None}
            
            def launch_wrapper(config: dict):
                """Callback triggered by UI Launch button"""
                participant_id = config.get("id")
                target_condition = config.get("target")
                control_condition = config.get("control") or DEFAULT_CONTROL_CONDITION
                
                # Apply Dynamic Settings
                from multi_agent_system.config.settings import get_settings, LLMBackend
                settings = get_settings()
                
                backend = (config.get("backend") or "openrouter").lower()
                if backend == "local":
                    settings.models.backend = LLMBackend.LOCAL
                elif backend == "openai":
                    settings.models.backend = LLMBackend.OPENAI
                else:
                    settings.models.backend = LLMBackend.OPENROUTER

                if config.get("public_model"):
                    public_model = str(config.get("public_model"))
                    settings.models.public_model_name = public_model
                    if settings.models.backend != LLMBackend.LOCAL:
                        settings.models.orchestrator_model = public_model
                        settings.models.critic_model = public_model
                        settings.models.predictor_model = public_model
                        settings.models.integrator_model = public_model
                        settings.models.communicator_model = public_model
                        settings.models.tool_model = public_model
                if config.get("public_max_context_tokens"):
                    settings.models.public_max_context_tokens = int(config.get("public_max_context_tokens"))
                if config.get("embedding_model"):
                    settings.models.embedding_model = str(config.get("embedding_model"))
                if config.get("local_embedding_model"):
                    settings.models.embedding_model = str(config.get("local_embedding_model"))

                if config.get("model"):
                    settings.models.local_model_name = str(config.get("model"))
                if config.get("max_tokens"):
                    settings.models.local_max_tokens = int(config.get("max_tokens"))
                if config.get("local_engine"):
                    settings.models.local_backend_type = str(config.get("local_engine"))
                if config.get("local_dtype"):
                    settings.models.local_dtype = str(config.get("local_dtype"))
                if config.get("local_quant") is not None:
                    settings.models.local_quantization = config.get("local_quant")
                if config.get("local_kv_cache_dtype"):
                    settings.models.local_kv_cache_dtype = str(config.get("local_kv_cache_dtype"))
                if config.get("local_attn"):
                    settings.models.local_attn_implementation = str(config.get("local_attn"))
                if config.get("local_tensor_parallel"):
                    settings.models.local_tensor_parallel_size = int(config.get("local_tensor_parallel"))
                if config.get("local_pipeline_parallel"):
                    settings.models.local_pipeline_parallel_size = int(config.get("local_pipeline_parallel"))
                if config.get("local_gpu_mem_util"):
                    settings.models.local_gpu_memory_utilization = float(config.get("local_gpu_mem_util"))
                if config.get("local_max_model_len"):
                    settings.models.local_max_model_len = int(config.get("local_max_model_len"))
                if config.get("local_enforce_eager") is not None:
                    settings.models.local_enforce_eager = bool(config.get("local_enforce_eager"))
                if config.get("local_trust_remote_code") is not None:
                    settings.models.local_trust_remote_code = bool(config.get("local_trust_remote_code"))

                role_models = config.get("role_models") or {}
                _apply_role_model_overrides(settings, role_models)

                role_token_limits = config.get("role_max_tokens") or {}
                _apply_role_max_token_overrides(settings, role_token_limits)

                if settings.models.backend == LLMBackend.LOCAL:
                    local_model = settings.models.local_model_name
                    settings.models.orchestrator_model = local_model
                    settings.models.critic_model = local_model
                    settings.models.predictor_model = local_model
                    settings.models.integrator_model = local_model
                    settings.models.communicator_model = local_model
                    settings.models.tool_model = local_model

                _clamp_role_token_limits(settings)

                # Apply Token Limits from UI (defaults derive from context window)
                _apply_token_budget_defaults(settings, config)
                if config.get("total_budget"):
                    settings.token_budget.total_budget = int(config.get("total_budget"))
                if config.get("max_agent_input") not in (None, "", 0):
                    settings.token_budget.max_agent_input_tokens = int(config.get("max_agent_input"))
                if config.get("max_agent_output") not in (None, "", 0):
                    settings.token_budget.max_agent_output_tokens = int(config.get("max_agent_output"))
                if config.get("max_tool_input") not in (None, "", 0):
                    settings.token_budget.max_tool_input_tokens = int(config.get("max_tool_input"))
                if config.get("max_tool_output") not in (None, "", 0):
                    settings.token_budget.max_tool_output_tokens = int(config.get("max_tool_output"))

                _apply_explainability_overrides(settings, args)

                reset_llm_client()
                print(f"[*] UI Triggered Launch: {participant_id} -> {target_condition} (control: {control_condition})")
                
                p_dir = resolve_participant_dir(participant_id, compass_data_root, settings)
                if not p_dir or not p_dir.exists():
                    print(f"[!] Error: Participant folder not found for ID: {participant_id}")
                    return
                print(f"[*] Fuzzy matched folder: {p_dir.name}")
                
                result = run_compass_pipeline(
                    participant_dir=p_dir,
                    target_condition=target_condition,
                    control_condition=control_condition,
                    max_iterations=args.iterations,
                    verbose=not args.quiet,
                    interactive_ui=args.ui,
                    generate_deep_phenotype=False,
                    generate_xai_report=False,
                )
                latest_run_context["internal"] = result.get("internal_context")

            def deep_report_wrapper(payload: Dict[str, Any]) -> Dict[str, Any]:
                internal = latest_run_context.get("internal")
                if not internal:
                    raise RuntimeError("No completed pipeline run found. Run a participant first.")

                communicator = Communicator()
                result = _generate_deep_phenotype_report(
                    communicator=communicator,
                    prediction=internal.get("prediction"),
                    evaluation=internal.get("evaluation"),
                    executor_output=internal.get("executor_output") or {},
                    data_overview=internal.get("data_overview") or {},
                    execution_summary=internal.get("execution_summary") or {},
                    control_condition=str(internal.get("control_condition") or DEFAULT_CONTROL_CONDITION),
                    report_context_note=str(internal.get("report_context_note") or ""),
                    base_output_dir=Path(str(internal.get("base_output_dir"))),
                    participant_id=str(internal.get("participant_id") or "unknown"),
                    user_focus_modalities=str(payload.get("focus_modalities") or ""),
                    user_general_instruction=str(payload.get("general_instruction") or ""),
                    trigger_source="ui",
                    ui_step_id=990,
                    interactive_ui=True,
                )
                if not result.get("success"):
                    raise RuntimeError(result.get("error") or "Deep phenotype generation failed.")
                return result

            print("Launching COMPASS Dashboard...")
            
            # Auto-trigger if path provided via CLI
            if args.participant_dir and args.participant_dir.exists():
                participant_id = args.participant_dir.name
                target_condition = args.target or "neuropsychiatric"
                control_condition = args.control or DEFAULT_CONTROL_CONDITION
                # Small delay to ensure server is up before first event
                def auto_launch():
                    time.sleep(2)
                    launch_wrapper({"id": participant_id, "target": target_condition, "control": control_condition})
                threading.Thread(target=auto_launch, daemon=True).start()

            start_ui_loop(launch_wrapper, deep_report_wrapper)
            print("\nPipeline complete. Closing dashboard server...")
        else:
            # Update settings with detailed log flag
            settings = get_settings()
            settings.detailed_tool_logging = args.detailed_log

            # Apply Backend Settings
            from multi_agent_system.config.settings import LLMBackend
            if args.backend == "local":
                settings.models.backend = LLMBackend.LOCAL
                settings.models.local_model_name = args.model
                settings.models.local_max_tokens = args.max_tokens
                settings.models.local_backend_type = args.local_engine
                settings.models.local_dtype = args.local_dtype
                settings.models.local_quantization = args.local_quant
                settings.models.local_kv_cache_dtype = args.local_kv_cache_dtype
                settings.models.local_tensor_parallel_size = args.local_tensor_parallel
                settings.models.local_pipeline_parallel_size = args.local_pipeline_parallel
                settings.models.local_gpu_memory_utilization = args.local_gpu_mem_util
                settings.models.local_max_model_len = args.local_max_model_len
                if args.local_enforce_eager:
                    settings.models.local_enforce_eager = True
                if args.local_trust_remote_code:
                    settings.models.local_trust_remote_code = True
                settings.models.local_attn_implementation = args.local_attn
                settings.models.orchestrator_model = args.model
                settings.models.critic_model = args.model
                settings.models.predictor_model = args.model
                settings.models.integrator_model = args.model
                settings.models.communicator_model = args.model
                settings.models.tool_model = args.model
                print(f"[Init] Switching to LOCAL Backend with model: {args.model}")
            elif args.backend == "openai":
                settings.models.backend = LLMBackend.OPENAI
                settings.models.public_model_name = args.public_model
                settings.models.public_max_context_tokens = args.public_max_context_tokens
                settings.models.embedding_model = args.embedding_model
                settings.models.orchestrator_model = args.public_model
                settings.models.critic_model = args.public_model
                settings.models.predictor_model = args.public_model
                settings.models.integrator_model = args.public_model
                settings.models.communicator_model = args.public_model
                settings.models.tool_model = args.public_model
            else:
                settings.models.backend = LLMBackend.OPENROUTER
                settings.models.public_model_name = args.public_model
                settings.models.public_max_context_tokens = args.public_max_context_tokens
                settings.models.embedding_model = args.embedding_model
                settings.models.orchestrator_model = args.public_model
                settings.models.critic_model = args.public_model
                settings.models.predictor_model = args.public_model
                settings.models.integrator_model = args.public_model
                settings.models.communicator_model = args.public_model
                settings.models.tool_model = args.public_model

            _clamp_role_token_limits(settings)

            # Apply Token Limits (CLI)
            _apply_token_budget_defaults(
                settings,
                {
                    "max_agent_input": args.max_agent_input,
                    "max_agent_output": args.max_agent_output,
                    "max_tool_input": args.max_tool_input,
                    "max_tool_output": args.max_tool_output,
                },
            )
            if args.total_budget:
                settings.token_budget.total_budget = args.total_budget
            if args.max_agent_input:
                settings.token_budget.max_agent_input_tokens = args.max_agent_input
            if args.max_agent_output:
                settings.token_budget.max_agent_output_tokens = args.max_agent_output
            if args.max_tool_input:
                settings.token_budget.max_tool_input_tokens = args.max_tool_input
            if args.max_tool_output:
                settings.token_budget.max_tool_output_tokens = args.max_tool_output

            _apply_explainability_overrides(settings, args)

            reset_llm_client()

            # Run standard CLI
            result = run_compass_pipeline(
                participant_dir=args.participant_dir,
                target_condition=args.target,
                control_condition=args.control or DEFAULT_CONTROL_CONDITION,
                max_iterations=args.iterations,
                verbose=not args.quiet,
                interactive_ui=False,
                generate_deep_phenotype=bool(args.generate_deep_phenotype),
                generate_xai_report=bool(args.generate_xai_report),
            )
        
        # Exit with appropriate code
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError running COMPASS pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
