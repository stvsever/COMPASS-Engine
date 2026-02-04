#!/usr/bin/env python3
"""
COMPASS Multi-Agent System

Clinical Orchestrated Multi-modal Predictive Agent Support System

Main entry point for running the COMPASS pipeline on participant data.
"""

import argparse
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_system.config.settings import get_settings
from multi_agent_system.utils.core.data_loader import DataLoader
from multi_agent_system.utils.core.token_manager import TokenManager
from multi_agent_system.agents.orchestrator import Orchestrator
from multi_agent_system.agents.executor import Executor
from multi_agent_system.agents.predictor import Predictor
from multi_agent_system.agents.critic import Critic
from multi_agent_system.utils.compass_logging.execution_logger import ExecutionLogger
from multi_agent_system.utils.compass_logging.decision_trace import DecisionTrace
from multi_agent_system.utils.compass_logging.patient_report import PatientReportGenerator
from multi_agent_system.models.prediction_result import Verdict
from multi_agent_system.models.prediction_result import Verdict
from multi_agent_system.utils.compass_ui import get_ui, reset_ui, start_ui_loop


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
    
    # Initialize agents
    if interactive_ui: ui.set_status("Initializing Agents...", stage=0)
    print(f"\n[2/5] Initializing COMPASS agents...")
    orchestrator = Orchestrator(token_manager=token_manager)
    executor = Executor(token_manager=token_manager)
    predictor = Predictor(token_manager=token_manager)
    critic = Critic(token_manager=token_manager)
    
    # Main loop: Orchestrator -> Executor -> Predictor -> Critic
    iteration = 1
    previous_feedback = None
    final_prediction = None
    final_evaluation = None
    
    while iteration <= max_iterations:
        print(f"\n{'='*70}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'='*70}")
        
        # Step 3: Orchestrator creates plan
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
        
        # Step 5: Predictor makes prediction
        if interactive_ui: ui.set_status("Generating Prediction...", stage=3)
        print(f"\n[5/5] Predictor generating prediction...")
        
        # Send fused input to UI for inspection
        if interactive_ui and "predictor_input" in executor_output:
            ui.on_fusion_complete(executor_output["predictor_input"])

        prediction = predictor.execute(
            executor_output=executor_output,
            target_condition=target_condition,
            iteration=iteration
        )
        
        exec_logger.log_predictor({
            "classification": prediction.binary_classification.value,
            "probability": prediction.probability_score
        })
        
        decision_trace.record_prediction(
            classification=prediction.binary_classification.value,
            probability=prediction.probability_score,
            key_findings=[f.finding for f in prediction.key_findings[:3]],
            reasoning=prediction.clinical_summary[:500]
        )
        
        # Step 6: Critic evaluates
        if interactive_ui: ui.set_status("Critic Evaluating...", stage=4)
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
        
        final_prediction = prediction
        final_evaluation = evaluation
        
        # Check if satisfactory
        if evaluation.verdict == Verdict.SATISFACTORY:
            print(f"\n✓ Prediction deemed SATISFACTORY by Critic")
            break
        else:
            print(f"\n✗ Prediction deemed UNSATISFACTORY by Critic")
            if iteration < max_iterations:
                print(f"  Re-orchestrating with critic feedback...")
                previous_feedback = _format_feedback(evaluation)
            iteration += 1
    
    # Generate final report
    print(f"\n{'='*70}")
    print(f"  GENERATING FINAL REPORT")
    print(f"{'='*70}")
    
    report_generator = PatientReportGenerator()
    
    # Get token usage details
    token_usage = token_manager.get_detailed_usage()
    
    # Collect detailed logs from Logger or Trace? 
    # Actually, we need to collect them from the executor results if they are stored there.
    # But executor returns the result of the LAST iteration.
    # We should rely on `exec_logger` to track them across all steps?
    # For now, let's grab them from the final executor_output if available, or just empty list.
    detailed_logs_collection = []
    # (Implementation Note: Ideally we'd aggregate them properly. Basic placeholder for now.)
    
    execution_summary = {
        "iterations": iteration,
        "tokens_used": token_usage.get("total_tokens", 0),
        "domains_processed": plan.priority_domains,
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
    base_output_dir = settings.paths.output_dir / f"participant_{participant_id}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    report_generator.save(report, base_output_dir)
    report_generator.save_markdown(report, base_output_dir)
    exec_logger.save_structured_log(base_output_dir / f"execution_log_{participant_id}.json")
    
    # Log completion
    duration = (datetime.now() - start_time).total_seconds()
    
    # Generate Performance Report
    performance_report = {
        "participant_id": participant_id,
        "target_condition": target_condition,
        "execution_timestamp": start_time.isoformat(),
        "total_duration_seconds": round(duration, 2),
        "iterations": iteration,
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
            "plan_id": plan.plan_id,
            "total_steps": plan.total_steps,
            "priority_domains": plan.priority_domains
        }
    }
    
    # Save performance report as JSON
    import json
    with open(base_output_dir / f"performance_report_{participant_id}.json", 'w') as f:
        json.dump(performance_report, f, indent=2)
    
    exec_logger.log_pipeline_end(
        success=True,
        summary={
            "prediction": final_prediction.binary_classification.value,
            "probability": final_prediction.probability_score,
            "iterations": iteration,
            "duration_seconds": duration
        }
    )
    
    print(f"\n{'='*70}")
    print(f"  COMPASS PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Participant: {participant_id}")
    print(f"  Prediction: {final_prediction.binary_classification.value}")
    print(f"  Probability: {final_prediction.probability_score:.1%}")
    print(f"  Iterations: {iteration}")
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
        "iterations": iteration,
        "duration_seconds": duration,
        "output_dir": str(base_output_dir),
        "report": report
    }


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
            
            def launch_wrapper(participant_id: str, target_condition: str):
                """Callback triggered by UI Launch button"""
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
                    interactive_ui=True
                )

            print("Launching COMPASS Dashboard...")
            
            # Auto-trigger if path provided via CLI
            if args.participant_dir and args.participant_dir.exists():
                participant_id = args.participant_dir.name
                target_condition = args.target or "neuropsychiatric"
                # Small delay to ensure server is up before first event
                def auto_launch():
                    time.sleep(2)
                    launch_wrapper(participant_id, target_condition)
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
