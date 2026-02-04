
import sys
import os
import subprocess
import time
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / "data" / "__FEATURES__" / "COMPASS_data"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
RESULTS_DIR = PROJECT_ROOT.parent / "results"

# To be processed participants (both fake and real targets)
PARTICIPANTS = [
    {"id": "5755396", "expected": "CASE", "target_str": "ANXIETY_DISORDERS | F419:Anxiety disorder, unspecified"},
    {"id": "1452610", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "4072571", "expected": "CONTROL", "target_str": "ANXIETY_DISORDERS"},
    {"id": "5180280", "expected": "CASE", "target_str": "SUBSTANCE_USE_DISORDERS | F171:F17.1 Harmful use"},
    {"id": "1491991", "expected": "CASE", "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "3135697", "expected": "CASE", "target_str": "ANXIETY_DISORDERS | F411:Generalized anxiety disorder"},
    {"id": "5165165", "expected": "CONTROL", "target_str": "ANXIETY_DISORDERS"},
    {"id": "4528012", "expected": "CONTROL", "target_str": "ANXIETY_DISORDERS"},
    {"id": "5545315", "expected": "CASE", "target_str": "ANXIETY_DISORDERS | F410:Panic disorder [episodic paroxysmal anxiety]"},
    {"id": "1719479", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"}
]

def run_participant(pid_info):
    pid = pid_info["id"]
    target_str = pid_info["target_str"]
    
    # Ensure ID prefix is present
    folder_id = pid if pid.startswith("ID") else f"participant_ID{pid}"
    path = os.path.join(DATA_ROOT, folder_id)
    
    # Check if path exists logic (optional but good)
    if not os.path.exists(path):
         # Try without participant_ID prefix if user data differs
         if os.path.exists(os.path.join(DATA_ROOT, f"ID{pid}")):
             path = os.path.join(DATA_ROOT, f"ID{pid}")
         elif os.path.exists(os.path.join(DATA_ROOT, pid)):
             path = os.path.join(DATA_ROOT, pid)
    
    # Output file for this process (to avoid PIPE deadlock)
    results_dir = Path(RESULTS_DIR) / f"participant_{pid}"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file_path = results_dir / f"batch_out_{pid}.txt"
    out_file = open(out_file_path, "w")
    print(f"DEBUG: Created log file at {out_file_path}")
    
    cmd = [
        "python3.11",  # FORCE Python 3.11 for Local LLM compatibility
        MAIN_SCRIPT, 
        path, 
        "--target", target_str,  # Pass the dynamic target string
        "--detailed_log",
        "--quiet"
    ]
    
    # Pass backend args if present in global config (simplification for batch script)
    # We will accept these as args to batch_run.py
    if BATCH_ARGS.get("backend"):
        cmd.extend(["--backend", BATCH_ARGS["backend"]])
    if BATCH_ARGS.get("model"):
        cmd.extend(["--model", BATCH_ARGS["model"]])
    
    print(f"Launching {pid}...")
    print(f"  > Target: {target_str[:50]}...")
    # Clean up previous log if exists
    # Use Popen with file stdout AND stderr merged
    proc = subprocess.Popen(cmd, stdout=out_file, stderr=subprocess.STDOUT, text=True)
    return proc, pid, out_file, out_file_path

BATCH_ARGS = {}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "local"], default="openai")
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    args = parser.parse_args()
    
    BATCH_ARGS["backend"] = args.backend
    BATCH_ARGS["model"] = args.model
    
    procs = []
    
    print(f"Found {len(PARTICIPANTS)} participants to run.")
    results = {}

    # Launch sequentially
    for p in PARTICIPANTS:
        proc, pid, out_file, out_path = run_participant(p)
        
        # Wait for this one to finish immediately
        _, stderr = proc.communicate()
        out_file.close()

        if proc.returncode != 0:
            print(f"ERROR for {pid}")
            results[pid] = "ERROR"
        else:
            print(f"Finished {pid}")
            results[pid] = "DONE" # Default

    print("\n" + "="*50)


    print("\n" + "="*50)
    print("BATCH SUMMARY")
    print("="*50)
    
    correct = 0
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    for p in PARTICIPANTS:
        pid = p["id"]
        expected = p["expected"]
        
        # Parse actual result from file to be sure
        report_path = Path(RESULTS_DIR) / f"participant_{pid}" / f"report_{pid}.md"
        actual = "UNKNOWN"
        
        if report_path.exists():
            with open(report_path, 'r') as f:
                content = f.read()
                if "**Classification**: CASE" in content:
                    actual = "CASE"
                elif "**Classification**: CONTROL" in content:
                    actual = "CONTROL"
        else:
             print(f"Report not found for {pid} at {report_path}")
        
        # Score
        is_correct = (actual == expected)
        if is_correct: correct += 1
        
        # Confusion Matrix
        if expected == "CASE" and actual == "CASE": confusion["TP"] += 1
        elif expected == "CONTROL" and actual == "CONTROL": confusion["TN"] += 1
        elif expected == "CONTROL" and actual == "CASE": confusion["FP"] += 1
        elif expected == "CASE" and actual == "CONTROL": confusion["FN"] += 1
        
        print(f"{pid}: Expected {expected} -> Actual {actual} [{'checkmark' if is_correct else 'X'}]")

    print("\nCONFUSION MATRIX:")
    print(f"TP: {confusion['TP']}  FN: {confusion['FN']}")
    print(f"FP: {confusion['FP']}  TN: {confusion['TN']}")
    print(f"\nAccuracy: {correct}/{len(PARTICIPANTS)}")

if __name__ == "__main__":
    main()
