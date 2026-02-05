import sys
import os
import subprocess
import time
from pathlib import Path

# Config
# Config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT.parent / "data" / "__FEATURES__" / "COMPASS_data"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
RESULTS_DIR = PROJECT_ROOT.parent / "results"

# To be processed participants (Requested by user)
PARTICIPANTS = [
    {"id": "1386427", "expected": "CASE", "target_str": "ANXIETY_DISORDERS | F419:Anxiety disorder, unspecified"},
    {"id": "4414177", "expected": "CASE", "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "4073873", "expected": "CASE", "target_str": "MAJOR_DEPRESSIVE_DISORDER | F329:Major depressive disorder, single episode, unspecified"},
    {"id": "2636640", "expected": "CONTROL", "target_str": "ANXIETY_DISORDERS"},
    {"id": "5022191", "expected": "CONTROL", "target_str": "ANXIETY_DISORDERS"},
    {"id": "3759408", "expected": "CONTROL", "target_str": "BIPOLAR_AND_MANIC_DISORDERS"},
    {"id": "1385600", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "1364077", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "3026819", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
    {"id": "2546474", "expected": "CONTROL", "target_str": "SLEEP_WAKE_DISORDERS"}
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
    
    cmd = [
        "python",
        str(MAIN_SCRIPT), 
        str(path), 
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
    # Use Popen with file stdout AND stderr merged
    proc = subprocess.Popen(cmd, stdout=out_file, stderr=subprocess.STDOUT, text=True)
    return proc, pid, out_file, out_file_path

BATCH_ARGS = {}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "local"], default="openai")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    args = parser.parse_args()
    
    BATCH_ARGS["backend"] = args.backend
    BATCH_ARGS["model"] = args.model
    
    print(f"Found {len(PARTICIPANTS)} participants to run.")
    results = {}

    # Launch sequentially
    for p in PARTICIPANTS:
        proc, pid, out_file, out_path = run_participant(p)
        
        # Wait for this one to finish immediately
        proc.wait()
        out_file.close()

        if proc.returncode != 0:
            print(f"ERROR for {pid}")
            results[pid] = "ERROR"
        else:
            print(f"Finished {pid}")
            results[pid] = "DONE" # Default

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
        
        # Score
        is_correct = (actual == expected)
        if is_correct: correct += 1
        
        # Confusion Matrix
        if expected == "CASE" and actual == "CASE": confusion["TP"] += 1
        elif expected == "CONTROL" and actual == "CONTROL": confusion["TN"] += 1
        elif expected == "CONTROL" and actual == "CASE": confusion["FP"] += 1
        elif expected == "CASE" and actual == "CONTROL": confusion["FN"] += 1
        
        print(f"{pid}: Expected {expected} -> Actual {actual} [{'DONE' if is_correct else 'FAIL'}]")

    print("\nCONFUSION MATRIX:")
    print(f"TP: {confusion['TP']}  FN: {confusion['FN']}")
    print(f"FP: {confusion['FP']}  TN: {confusion['TN']}")
    print(f"\nAccuracy: {correct}/{len(PARTICIPANTS)}")

if __name__ == "__main__":
    main()
