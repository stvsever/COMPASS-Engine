import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import timedelta
from typing import List, Optional

# Config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# DATA_ROOT: configurable via environment variable for HPC vs local
# On HPC: set DATA_ROOT env var in the Slurm script (points to HPC_data)
# Locally: defaults to COMPASS_data
DATA_ROOT = Path(os.getenv(
    "DATA_ROOT",
    str(PROJECT_ROOT.parent / "data" / "__FEATURES__" / "HPC_data")
))

MAIN_SCRIPT = PROJECT_ROOT / "main.py"
VALIDATION_DIR = PROJECT_ROOT / "utils" / "validation" / "with_annotated_dataset"
VALIDATION_METRICS_SCRIPT = VALIDATION_DIR / "run_validation_metrics.py"
VALIDATION_DETAILED_SCRIPT = VALIDATION_DIR / "detailed_analysis.py"
VALIDATION_TEMPLATE_EXAMPLES_DIR = VALIDATION_DIR / "annotation_templates" / "examples"
RESULTS_DIR = PROJECT_ROOT.parent / "results"

VALIDATION_TEMPLATE_BY_MODE = {
    "binary": "binary_targets_example.json",
    "multiclass": "multiclass_annotations_example.json",
    "regression_univariate": "regression_univariate_annotations_example.json",
    "regression_multivariate": "regression_multivariate_annotations_example.json",
    "hierarchical": "hierarchical_annotations_example.json",
}

# ─── Participant Cohort ───────────────────────────────────────────────────
# Example participant cohort (anonymized placeholders).
# Replace these IDs with your real participant IDs / EIDs for your dataset.
#
# Format:
#   id         — participant identifier (folder: participant_ID{id})
#   expected   — ground-truth label (CASE or CONTROL)
#   target_str — phenotype label passed to main.py --target_label
PARTICIPANTS = [
    {"id": "01", "expected": "CASE",    "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "02", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "03", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "04", "expected": "CASE",    "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "05", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
]
def run_participant(pid_info):
    pid = pid_info["id"]
    target_str = pid_info["target_str"]
    
    # Build participant data path (format: participant_ID{eid})
    folder_name = f"participant_ID{pid}"
    path = os.path.join(DATA_ROOT, folder_name)
    
    # Fallback path resolution
    if not os.path.exists(path):
        # Try with ID prefix
        alt = os.path.join(DATA_ROOT, f"ID{pid}")
        if os.path.exists(alt):
            path = alt
        elif os.path.exists(os.path.join(DATA_ROOT, pid)):
            path = os.path.join(DATA_ROOT, pid)
    
    # Output file for this process (to avoid PIPE deadlock)
    results_dir = Path(RESULTS_DIR) / f"participant_{pid}"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file_path = results_dir / f"batch_out_{pid}.txt"
    out_file = open(out_file_path, "w")
    
    # Environment variables for the subprocess
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    
    prediction_type = str(BATCH_ARGS.get("prediction_type") or "binary")
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT), 
        str(path), 
        "--prediction_type", prediction_type,
        "--target_label", target_str,
        "--detailed_log",
        "--quiet"
    ]
    if BATCH_ARGS.get("control_label"):
        cmd.extend(["--control_label", str(BATCH_ARGS["control_label"])])
    if BATCH_ARGS.get("class_labels"):
        cmd.extend(["--class_labels", str(BATCH_ARGS["class_labels"])])
    if BATCH_ARGS.get("regression_output"):
        cmd.extend(["--regression_output", str(BATCH_ARGS["regression_output"])])
    elif BATCH_ARGS.get("regression_outputs"):
        cmd.extend(["--regression_outputs", str(BATCH_ARGS["regression_outputs"])])
    if BATCH_ARGS.get("task_spec_file"):
        cmd.extend(["--task_spec_file", str(BATCH_ARGS["task_spec_file"])])
    if BATCH_ARGS.get("task_spec_json"):
        cmd.extend(["--task_spec_json", str(BATCH_ARGS["task_spec_json"])])
    
    # Pass backend args if present in global config
    if BATCH_ARGS.get("backend"):
        cmd.extend(["--backend", BATCH_ARGS["backend"]])
    if BATCH_ARGS.get("model"):
        cmd.extend(["--model", BATCH_ARGS["model"]])
    if BATCH_ARGS.get("max_tokens") is not None:
        cmd.extend(["--max_tokens", str(BATCH_ARGS["max_tokens"])])
    if BATCH_ARGS.get("max_agent_input") is not None:
        cmd.extend(["--max_agent_input", str(BATCH_ARGS["max_agent_input"])])
    if BATCH_ARGS.get("max_agent_output") is not None:
        cmd.extend(["--max_agent_output", str(BATCH_ARGS["max_agent_output"])])
    if BATCH_ARGS.get("max_tool_input") is not None:
        cmd.extend(["--max_tool_input", str(BATCH_ARGS["max_tool_input"])])
    if BATCH_ARGS.get("max_tool_output") is not None:
        cmd.extend(["--max_tool_output", str(BATCH_ARGS["max_tool_output"])])
    if BATCH_ARGS.get("local_engine"):
        cmd.extend(["--local_engine", BATCH_ARGS["local_engine"]])
    if BATCH_ARGS.get("local_dtype"):
        cmd.extend(["--local_dtype", BATCH_ARGS["local_dtype"]])
    if BATCH_ARGS.get("local_quant"):
        cmd.extend(["--local_quant", BATCH_ARGS["local_quant"]])
    if BATCH_ARGS.get("local_kv_cache_dtype"):
        cmd.extend(["--local_kv_cache_dtype", BATCH_ARGS["local_kv_cache_dtype"]])
    if BATCH_ARGS.get("local_tensor_parallel") is not None:
        cmd.extend(["--local_tensor_parallel", str(BATCH_ARGS["local_tensor_parallel"])])
    if BATCH_ARGS.get("local_pipeline_parallel") is not None:
        cmd.extend(["--local_pipeline_parallel", str(BATCH_ARGS["local_pipeline_parallel"])])
    if BATCH_ARGS.get("local_gpu_mem_util") is not None:
        cmd.extend(["--local_gpu_mem_util", str(BATCH_ARGS["local_gpu_mem_util"])])
    if BATCH_ARGS.get("local_max_model_len") is not None:
        cmd.extend(["--local_max_model_len", str(BATCH_ARGS["local_max_model_len"])])
    if BATCH_ARGS.get("local_enforce_eager"):
        cmd.extend(["--local_enforce_eager"])
    if BATCH_ARGS.get("local_trust_remote_code"):
        cmd.extend(["--local_trust_remote_code"])
    if BATCH_ARGS.get("local_attn"):
        cmd.extend(["--local_attn", BATCH_ARGS["local_attn"]])
    
    print(f"Launching {pid} ({pid_info['expected']})...")
    print(f"  > Path:   {path}")
    print(f"  > Target: {target_str[:80]}...")
    print(f"  > Cmd:    {' '.join(cmd[:6])}...")
    # Use Popen with file stdout AND stderr merged, and custom env
    proc = subprocess.Popen(cmd, stdout=out_file, stderr=subprocess.STDOUT, text=True, env=env)
    return proc, pid, out_file, out_file_path

BATCH_ARGS = {}


def _resolve_validation_results_dir(explicit_path: str = "") -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        return None

    candidates = [
        RESULTS_DIR / "participant_runs",
        RESULTS_DIR,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _resolve_metrics_script() -> Optional[Path]:
    if VALIDATION_METRICS_SCRIPT.exists():
        return VALIDATION_METRICS_SCRIPT
    return None


def _validation_template_hint(prediction_type: str) -> Path:
    name = VALIDATION_TEMPLATE_BY_MODE.get(str(prediction_type or "binary").strip().lower(), "binary_targets_example.json")
    return VALIDATION_TEMPLATE_EXAMPLES_DIR / name


def _run_validation_command(cmd: List[str], label: str) -> int:
    print(f"\n  [{label}] {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"  ✗ {label} failed (exit={proc.returncode})")
    else:
        print(f"  ✓ {label} complete")
    return int(proc.returncode)


def run_posthoc_validation(args) -> None:
    if not args.run_validation:
        return

    print()
    print("=" * 60)
    print(" POST-HOC ANNOTATED VALIDATION")
    print("=" * 60)

    metrics_script = _resolve_metrics_script()
    if metrics_script is None:
        print("  ⚠ Skipped: no validation metrics script found.")
        return
    if not VALIDATION_DETAILED_SCRIPT.exists():
        print("  ⚠ Skipped: detailed validation script not found.")
        return

    results_dir = _resolve_validation_results_dir(args.validation_results_dir)
    if results_dir is None or not results_dir.exists():
        print("  ⚠ Skipped: validation results directory not found.")
        return

    prediction_type = str(args.prediction_type or "binary").strip().lower()
    needs_binary_targets = prediction_type == "binary"
    targets_file = str(args.validation_targets_file or "").strip()
    annotations_json = str(args.validation_annotations_json or "").strip()

    if needs_binary_targets and not targets_file:
        print("  ⚠ Skipped: binary validation requires --validation_targets_file.")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return
    if (not needs_binary_targets) and not annotations_json:
        print("  ⚠ Skipped: non-binary validation requires --validation_annotations_json.")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return
    if needs_binary_targets and not Path(targets_file).exists():
        print(f"  ⚠ Skipped: validation targets file not found: {targets_file}")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return
    if needs_binary_targets and Path(targets_file).suffix.lower() != ".json":
        print(f"  ⚠ Skipped: binary validation requires JSON --validation_targets_file, got: {targets_file}")
        print("  ↳ Legacy txt targets are no longer supported.")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return
    if (not needs_binary_targets) and not Path(annotations_json).exists():
        print(f"  ⚠ Skipped: validation annotations file not found: {annotations_json}")
        print(f"  ↳ Template example: {_validation_template_hint(prediction_type)}")
        return

    output_root = Path(args.validation_output_dir) if str(args.validation_output_dir or "").strip() else (RESULTS_DIR / "analysis")
    if prediction_type == "binary":
        metrics_out = output_root / "binary_confusion_matrix"
        detailed_out = output_root / "details"
    else:
        metrics_out = output_root / f"{prediction_type}_metrics"
        detailed_out = output_root / f"{prediction_type}_details"
    metrics_out.mkdir(parents=True, exist_ok=True)
    detailed_out.mkdir(parents=True, exist_ok=True)

    common_args = [
        "--results_dir", str(results_dir),
        "--prediction_type", prediction_type,
    ]
    if needs_binary_targets:
        common_args.extend(["--targets_file", targets_file])
    else:
        common_args.extend(["--annotations_json", annotations_json])
    if str(args.validation_disorder_groups or "").strip():
        common_args.extend(["--disorder_groups", str(args.validation_disorder_groups).strip()])

    metrics_cmd = [
        sys.executable,
        str(metrics_script),
        *common_args,
        "--output_dir", str(metrics_out),
    ]
    detailed_cmd = [
        sys.executable,
        str(VALIDATION_DETAILED_SCRIPT),
        *common_args,
        "--output_dir", str(detailed_out),
    ]

    print(f"  Results dir:      {results_dir}")
    print(f"  Metrics output:   {metrics_out}")
    print(f"  Detailed output:  {detailed_out}")
    print(f"  Template example: {_validation_template_hint(prediction_type)}")

    rc_metrics = _run_validation_command(metrics_cmd, "validation-metrics")
    rc_detailed = _run_validation_command(detailed_cmd, "validation-detailed")

    if rc_metrics == 0 and rc_detailed == 0:
        print("  ✓ Post-hoc annotated validation complete.")
    else:
        print("  ⚠ Post-hoc annotated validation completed with errors.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="COMPASS Batch Runner — Sequential participant processing")
    parser.add_argument("--backend", choices=["openrouter", "openai", "local"], default="local")
    parser.add_argument(
        "--prediction_type",
        choices=["binary", "multiclass", "regression_univariate", "regression_multivariate", "hierarchical"],
        default="binary",
        help="Prediction task type passed to main.py",
    )
    parser.add_argument("--control_label", type=str, default=None)
    parser.add_argument("--class_labels", type=str, default="")
    parser.add_argument("--regression_outputs", type=str, default="")
    parser.add_argument("--regression_output", type=str, default="")
    parser.add_argument("--task_spec_file", type=str, default="")
    parser.add_argument("--task_spec_json", type=str, default="")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--max_agent_input", type=int, default=None)
    parser.add_argument("--max_agent_output", type=int, default=None)
    parser.add_argument("--max_tool_input", type=int, default=None)
    parser.add_argument("--max_tool_output", type=int, default=None)
    parser.add_argument("--local_engine", type=str, default="auto")
    parser.add_argument("--local_dtype", type=str, default="auto")
    parser.add_argument("--local_quant", type=str, default=None)
    parser.add_argument("--local_kv_cache_dtype", type=str, default=None)
    parser.add_argument("--local_tensor_parallel", type=int, default=1)
    parser.add_argument("--local_pipeline_parallel", type=int, default=1)
    parser.add_argument("--local_gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--local_max_model_len", type=int, default=0)
    parser.add_argument("--local_enforce_eager", action="store_true")
    parser.add_argument("--local_trust_remote_code", action="store_true")
    parser.add_argument("--local_attn", type=str, default="auto")
    parser.add_argument(
        "--run_validation",
        action="store_true",
        help="Run post-hoc annotated validation after batch completion.",
    )
    parser.add_argument(
        "--validation_results_dir",
        type=str,
        default="",
        help="Directory containing participant output folders for validation (default: auto-detect).",
    )
    parser.add_argument(
        "--validation_output_dir",
        type=str,
        default="",
        help="Output root for validation artifacts (default: ../results/analysis).",
    )
    parser.add_argument(
        "--validation_targets_file",
        type=str,
        default="",
        help="Binary ground-truth targets JSON file (required when --prediction_type=binary and --run_validation).",
    )
    parser.add_argument(
        "--validation_annotations_json",
        type=str,
        default="",
        help="Generalized annotations JSON (required for non-binary validation).",
    )
    parser.add_argument(
        "--validation_disorder_groups",
        type=str,
        default="",
        help="Optional comma-separated groups/disorders for subgroup validation artifacts.",
    )
    args = parser.parse_args()
    regression_output = str(args.regression_output or "").strip()
    regression_outputs = str(args.regression_outputs or "").strip()
    if regression_output and regression_outputs:
        parsed_multi = [x.strip() for x in regression_outputs.split(",") if x.strip()]
        if parsed_multi != [regression_output]:
            raise ValueError("--regression_output conflicts with --regression_outputs. Use one form or provide matching values.")
    
    BATCH_ARGS["backend"] = args.backend
    BATCH_ARGS["prediction_type"] = args.prediction_type
    BATCH_ARGS["control_label"] = args.control_label
    BATCH_ARGS["class_labels"] = args.class_labels
    BATCH_ARGS["regression_outputs"] = regression_outputs
    BATCH_ARGS["regression_output"] = regression_output
    BATCH_ARGS["task_spec_file"] = args.task_spec_file
    BATCH_ARGS["task_spec_json"] = args.task_spec_json
    BATCH_ARGS["model"] = args.model
    BATCH_ARGS["max_tokens"] = args.max_tokens
    BATCH_ARGS["max_agent_input"] = args.max_agent_input
    BATCH_ARGS["max_agent_output"] = args.max_agent_output
    BATCH_ARGS["max_tool_input"] = args.max_tool_input
    BATCH_ARGS["max_tool_output"] = args.max_tool_output
    BATCH_ARGS["local_engine"] = args.local_engine
    BATCH_ARGS["local_dtype"] = args.local_dtype
    BATCH_ARGS["local_quant"] = args.local_quant
    BATCH_ARGS["local_kv_cache_dtype"] = args.local_kv_cache_dtype
    BATCH_ARGS["local_tensor_parallel"] = args.local_tensor_parallel
    BATCH_ARGS["local_pipeline_parallel"] = args.local_pipeline_parallel
    BATCH_ARGS["local_gpu_mem_util"] = args.local_gpu_mem_util
    BATCH_ARGS["local_max_model_len"] = args.local_max_model_len
    BATCH_ARGS["local_enforce_eager"] = args.local_enforce_eager
    BATCH_ARGS["local_trust_remote_code"] = args.local_trust_remote_code
    BATCH_ARGS["local_attn"] = args.local_attn
    
    n = len(PARTICIPANTS)
    n_cases = sum(1 for p in PARTICIPANTS if p["expected"] == "CASE")
    n_controls = n - n_cases
    
    print("=" * 60)
    print(" COMPASS Batch Runner")
    print("=" * 60)
    print(f"  Participants: {n} ({n_cases} CASE, {n_controls} CONTROL)")
    print(f"  Data root:    {DATA_ROOT}")
    print(f"  Backend:      {args.backend}")
    print(f"  Prediction:   {args.prediction_type}")
    print(f"  Model:        {args.model}")
    print(f"  Context:      {args.max_tokens}")
    print(
        f"  Budgets:      agent(in={args.max_agent_input}, out={args.max_agent_output}) | "
        f"tool(in={args.max_tool_input}, out={args.max_tool_output})"
    )
    print(
        f"  Local cfg:    engine={args.local_engine}, dtype={args.local_dtype}, "
        f"quant={args.local_quant}, gpu_mem={args.local_gpu_mem_util}, "
        f"max_model_len={args.local_max_model_len}"
    )
    print(f"  Validation:   {'ON' if args.run_validation else 'OFF'}")
    if args.run_validation:
        print(f"  Val results:  {args.validation_results_dir or '<auto>'}")
        print(f"  Val output:   {args.validation_output_dir or str(RESULTS_DIR / 'analysis')}")
        if args.prediction_type == "binary":
            print(f"  Val targets:  {args.validation_targets_file or '<required>'}")
        else:
            print(f"  Val annjson:  {args.validation_annotations_json or '<required>'}")
    print(f"  Processing:   SEQUENTIAL (1 GPU)")
    print("=" * 60)
    print()
    
    results = {}
    timings = {}
    batch_start = time.time()

    # Launch sequentially
    for i, p in enumerate(PARTICIPANTS, 1):
        pid = p["id"]
        print(f"\n{'─' * 60}")
        print(f" [{i}/{n}] Participant {pid} ({p['expected']})")
        print(f"{'─' * 60}")
        
        t0 = time.time()
        proc, pid, out_file, out_path = run_participant(p)
        
        # Wait for this one to finish immediately
        proc.wait()
        out_file.close()
        
        elapsed = time.time() - t0
        timings[pid] = elapsed
        td = timedelta(seconds=int(elapsed))

        if proc.returncode != 0:
            print(f"  ✗ ERROR for {pid} (exit code {proc.returncode}) — {td}")
            results[pid] = "ERROR"
        else:
            print(f"  ✓ Finished {pid} — {td}")
            results[pid] = "DONE"

    batch_elapsed = time.time() - batch_start
    batch_td = timedelta(seconds=int(batch_elapsed))
    
    # ─── Timing Summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(" TIMING SUMMARY")
    print("=" * 60)
    for p in PARTICIPANTS:
        pid = p["id"]
        t = timings.get(pid, 0)
        td = timedelta(seconds=int(t))
        status = results.get(pid, "UNKNOWN")
        print(f"  {pid} ({p['expected']:>7}) — {td}  [{status}]")
    print(f"\n  Total batch wall time: {batch_td}")
    if timings:
        avg = sum(timings.values()) / len(timings)
        print(f"  Avg per participant:   {timedelta(seconds=int(avg))}")
    
    # ─── Classification Summary (binary mode only) ──────────────────────
    if args.prediction_type != "binary":
        print()
        print("=" * 60)
        print(" SUMMARY")
        print("=" * 60)
        print("  Binary confusion summary is skipped for non-binary prediction modes.")
        print("  Use annotated-dataset validation scripts for multiclass/regression/hierarchical metrics.")
        print(f"  Total wall time: {batch_td}")
        run_posthoc_validation(args)
        return

    print()
    print("=" * 60)
    print(" CLASSIFICATION SUMMARY")
    print("=" * 60)
    
    correct = 0
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    for p in PARTICIPANTS:
        pid = p["id"]
        expected = p["expected"]
        
        # Parse actual result from report file
        report_path = Path(RESULTS_DIR) / f"participant_{pid}" / f"report_{pid}.md"
        actual = "UNKNOWN"
        
        if report_path.exists():
            with open(report_path, 'r') as f:
                content = f.read()
                if "**Classification**: CASE" in content:
                    actual = "CASE"
                elif "**Classification**: CONTROL" in content:
                    actual = "CONTROL"
        
        # Score
        is_correct = (actual == expected)
        if is_correct: correct += 1
        
        # Confusion Matrix
        if expected == "CASE" and actual == "CASE": confusion["TP"] += 1
        elif expected == "CONTROL" and actual == "CONTROL": confusion["TN"] += 1
        elif expected == "CONTROL" and actual == "CASE": confusion["FP"] += 1
        elif expected == "CASE" and actual == "CONTROL": confusion["FN"] += 1
        
        t = timings.get(pid, 0)
        td = timedelta(seconds=int(t))
        marker = "✓" if is_correct else "✗"
        print(f"  {marker} {pid}: Expected {expected:>7} → Actual {actual:>7}  ({td})")

    print(f"\n  CONFUSION MATRIX:")
    print(f"    TP: {confusion['TP']}  FN: {confusion['FN']}")
    print(f"    FP: {confusion['FP']}  TN: {confusion['TN']}")
    print(f"\n  Accuracy: {correct}/{len(PARTICIPANTS)}")
    print(f"  Total wall time: {batch_td}")
    run_posthoc_validation(args)

if __name__ == "__main__":
    main()
