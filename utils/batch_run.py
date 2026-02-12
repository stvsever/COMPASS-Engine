import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import timedelta

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
RESULTS_DIR = PROJECT_ROOT.parent / "results"

# ─── Participant Cohort ───────────────────────────────────────────────────
# UK Biobank participants for clinical validation (Major Depressive Disorder)
# 5 CASE + 5 CONTROL, balanced cohort for binary classification evaluation
#
# Format:
#   id         — UK Biobank EID (folder: participant_ID{eid})
#   expected   — ground-truth label (CASE or CONTROL)
#   target_str — phenotype string passed to main.py --target
PARTICIPANTS = [
    {"id": "1416463", "expected": "CASE",    "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "3950738", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "3674748", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "3530988", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "4437905", "expected": "CASE",    "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "4819798", "expected": "CASE",    "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "2519442", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "3895610", "expected": "CASE",    "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "4931262", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "4612723", "expected": "CASE",    "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
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
    
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT), 
        str(path), 
        "--target", target_str,  # Pass the dynamic target string
        "--detailed_log",
        "--quiet"
    ]
    
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="COMPASS Batch Runner — Sequential participant processing")
    parser.add_argument("--backend", choices=["openrouter", "openai", "local"], default="local")
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
    args = parser.parse_args()
    
    BATCH_ARGS["backend"] = args.backend
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
    
    # ─── Classification Summary ──────────────────────────────────────────
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

if __name__ == "__main__":
    main()
