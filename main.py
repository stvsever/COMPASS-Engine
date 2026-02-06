#!/usr/bin/env python3
"""
COMPASS Multi-Agent System

Clinical Orchestrated Multi-modal Predictive Agent Support System

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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_system.config.settings import get_settings, LLMBackend
from multi_agent_system.utils.core.data_loader import DataLoader
from multi_agent_system.utils.core.token_manager import TokenManager
from multi_agent_system.utils.llm_client import get_llm_client
from multi_agent_system.utils.core.fusion_layer import FusionLayer, FusionResult
from multi_agent_system.utils.core.predictor_input_assembler import PredictorInputAssembler
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


def run_compass_pipeline(
    participant_dir: Path,
    target_condition: str,
    max_iterations: int = 3,
    verbose: bool = True,
    interactive_ui: bool = False
) -> dict:
    """
    Run the complete COMPASS pipeline for a participant.
    
    Args:
        participant_dir: Path to participant data directory
        target_condition: "neuropsychiatric" or "neurologic"
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
            participant_dir=str(participant_dir),
            max_iterations=max_iterations
        )
    else:
        print("\n" + "=" * 70)
        print("  COMPASS - Clinical Orchestrated Multi-modal Predictive Agent System")
        print("=" * 70)
    
    # Load participant data
    if interactive_ui: ui.set_status("Loading Participant Data...", stage=0)
    print(f"\n[1/5] Loading participant data from: {participant_dir}")
    data_loader = DataLoader()
    participant_data = data_loader.load(participant_dir)
    participant_id = participant_data.participant_id
    
    # Initialize logging
    exec_logger = ExecutionLogger(participant_id, verbose=verbose)
    decision_trace = DecisionTrace(participant_id)
    exec_logger.log_pipeline_start(target_condition)
    
    # Initialize token manager
    token_manager = TokenManager()
    
    # Preflight connectivity check (OpenAI backend only)
    if settings.models.backend == LLMBackend.OPENAI:
        if interactive_ui:
            ui.set_status("Checking OpenAI connectivity...", stage=0)
        try:
            get_llm_client().ping()
        except Exception as e:
            raise RuntimeError(
                "OpenAI connectivity check failed. Verify network access and OPENAI_API_KEY."
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
            target_condition=target_condition
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
        if interactive_ui: ui.set_status("Generating Prediction...", stage=4)
        if interactive_ui:
            ui.on_step_start(
                step_id=910 + iteration,
                tool_name="Predictor Agent",
                description="Prediction synthesis (chunked no-loss evidence flow)...",
                stage=4,
            )

        prediction = predictor.execute(
            executor_output=executor_output,
            target_condition=target_condition,
            iteration=iteration
        )
        
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
            non_numerical_data=participant_data.non_numerical_data.raw_text
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
    base_output_dir = settings.paths.output_dir / f"participant_{participant_id}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

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
        "tokens_used": token_usage.get("total_tokens", 0),
        "domains_processed": (final_plan.priority_domains if final_plan else plan.priority_domains),
        "detailed_logs": detailed_logs_collection # We need to populate this
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
        "execution_timestamp": start_time.isoformat(),
        "total_duration_seconds": round(duration_so_far, 2),
        "iterations": len(attempts),
        "selected_iteration": selected_iteration,
        "selection_reason": selection_reason,
        "coverage_summary": coverage_summary,
        "prediction_result": {
            "classification": final_prediction.binary_classification.value,
            "probability": round(final_prediction.probability_score, 4),
            "confidence": final_prediction.confidence_level.value
        },
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
        }
    }
    
    # Save performance report as JSON
    import json
    performance_report_path = base_output_dir / f"performance_report_{participant_id}.json"
    with open(performance_report_path, 'w') as f:
        json.dump(performance_report, f, indent=2)

    # Communicator: deep phenotyping report (final verdict only)
    if final_prediction and final_evaluation and final_executor_output:
        if interactive_ui:
            ui.set_status("Generating deep phenotype report...", stage=6)
            ui.on_step_start(
                step_id=930 + iteration,
                tool_name="Communicator Agent",
                description="Generating deep phenotype report...",
                stage=6,
            )

        try:
            data_overview_dict = participant_data.data_overview.model_dump()
            report_context_note = ""
            if final_evaluation.verdict != Verdict.SATISFACTORY:
                report_context_note = (
                    "WARNING: Selected final verdict remains UNSATISFACTORY. "
                    f"Selected iteration: {selected_iteration}. "
                    f"Selection basis: {selection_reason}. "
                    "Interpret deep phenotype content as exploratory and not production-final."
                )

            deep_report = communicator.execute(
                prediction=final_prediction,
                evaluation=final_evaluation,
                executor_output=final_executor_output,
                data_overview=data_overview_dict,
                execution_summary=execution_summary,
                report_context_note=report_context_note,
            )

            if deep_report:
                deep_path = base_output_dir / "deep_phenotype.md"
                with open(deep_path, "w") as f:
                    f.write(deep_report)
                print(f"[Communicator] Saved to: {deep_path}")

            if interactive_ui:
                ui.on_step_complete(
                    step_id=930 + iteration,
                    tokens=0,
                    duration_ms=0,
                    preview="Deep phenotype report generated."
                )
        except Exception as e:
            if interactive_ui:
                ui.on_step_failed(step_id=930 + iteration, error=str(e))
            print(f"[Communicator] Error: {e}")

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
        "coverage_summary": coverage_summary,
        "duration_seconds": duration,
        "output_dir": str(base_output_dir),
        "report": report
    }


def run_dataflow_audit(
    participant_dir: Path,
    target_condition: str,
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
    context = executor._build_context(participant_data, target_condition)

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
    chunk_budget = max(1200, min(30000, int(max_tool_input * 0.80)))
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
        "coverage_summary": coverage_ledger.get("summary", {}),
        "predictor_payload_tokens": payload_tokens,
        "chunk_budget_tokens": chunk_budget,
        "chunk_count": len(chunks),
        "section_stats": section_stats,
        "chunk_stats": chunk_stats,
        "predictor_input_mode": predictor_input.get("mode"),
    }

    output_dir = settings.paths.output_dir
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

    # --- LOCAL LLM ARGUMENTS ---
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=["openai", "local"],
        default="openai",
        help="Choose LLM backend: 'openai' (default) or 'local' (vLLM/Transformers)"
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
            
            def launch_wrapper(config: dict):
                """Callback triggered by UI Launch button"""
                participant_id = config.get("id")
                target_condition = config.get("target")
                
                # Apply Dynamic Settings
                from multi_agent_system.config.settings import get_settings, LLMBackend
                settings = get_settings()
                
                # Apply Token Limits from UI
                if config.get("total_budget"):
                    settings.token_budget.total_budget = int(config.get("total_budget"))
                if config.get("max_agent_input"):
                    settings.token_budget.max_agent_input_tokens = int(config.get("max_agent_input"))
                if config.get("max_agent_output"):
                    settings.token_budget.max_agent_output_tokens = int(config.get("max_agent_output"))
                if config.get("max_tool_input"):
                    settings.token_budget.max_tool_input_tokens = int(config.get("max_tool_input"))
                if config.get("max_tool_output"):
                    settings.token_budget.max_tool_output_tokens = int(config.get("max_tool_output"))

                print(f"[*] UI Triggered Launch: {participant_id} -> {target_condition}")
                
                # Construct full path
                p_dir = compass_data_root / participant_id
                
                if not p_dir.exists():
                    # Try finding it if user just typed ID number
                    potential = list(compass_data_root.glob(f"*{participant_id}*"))
                    if potential:
                        p_dir = potential[0]
                        print(f"[*] Fuzzy matched folder: {p_dir.name}")
                    else:
                        print(f"[!] Error: Participant folder not found for ID: {participant_id}")
                        return
                
                run_compass_pipeline(
                    participant_dir=p_dir,
                    target_condition=target_condition,
                    max_iterations=args.iterations,
                    verbose=not args.quiet,
                    interactive_ui=args.ui
                )

            print("Launching COMPASS Dashboard...")
            
            # Auto-trigger if path provided via CLI
            if args.participant_dir and args.participant_dir.exists():
                participant_id = args.participant_dir.name
                target_condition = args.target or "neuropsychiatric"
                # Small delay to ensure server is up before first event
                def auto_launch():
                    time.sleep(2)
                    launch_wrapper({"id": participant_id, "target": target_condition})
                threading.Thread(target=auto_launch, daemon=True).start()

            start_ui_loop(launch_wrapper)
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
                print(f"[Init] Switching to LOCAL Backend with model: {args.model}")
            else:
                settings.models.backend = LLMBackend.OPENAI

            # Apply Token Limits (CLI)
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

            # Run standard CLI
            result = run_compass_pipeline(
                participant_dir=args.participant_dir,
                target_condition=args.target,
                max_iterations=args.iterations,
                verbose=not args.quiet,
                interactive_ui=False
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
