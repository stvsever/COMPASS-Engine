"""
COMPASS Modern Web Dashboard
============================

A professional, high-aesthetic web interface for monitoring the COMPASS pipeline.
Uses Flask to serve a local dashboard with real-time updates.
Includes optimized logic for robustness and thread safety.
"""

import threading
import time
import json
import logging
import webbrowser
import queue
from flask import Flask, render_template, jsonify, request, send_file, make_response
from flask.json.provider import DefaultJSONProvider
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

# Disable flask logging for a cleaner terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class EventStore:
    """Thread-safe storage for pipeline events with optimized state tracking."""
    def __init__(self):
        self._lock = threading.Lock()
        self.reset() # init msg

    def reset(self):
        self.events = []
        self.state = {
            "participant_id": "Unknown",
            "participant_dir": None,
            "target": "None",
            "control": None,
            "status": "Ready to Launch",
            "start_time": None,
            "total_tokens": 0,
            "progress": 0,
            "max_steps": 1, 
            "steps": [],
            "history": [], # Archive of steps from previous iterations
            "prediction": None,
            "critic": None,
            "critic_summary": None,
            "completed": False,
            "completion": None,
            "latest_update_id": 0,
            "current_stage": -1, # -1: Setup, 0:Init, 1:Plan, 2:Exec, 3:Predict, 4:Evaluate
            "stages": ["Initialization", "Orchestration", "Execution", "Integration", "Prediction", "Evaluation", "Communication"],
            "iteration": 1
        }

    def add_event(self, event_type, data):
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            # Enriched event data
            event = {
                "id": self.state["latest_update_id"] + 1,
                "time": timestamp, 
                "type": event_type, 
                "data": data
            }
            self.events.append(event)
            self.state["latest_update_id"] = event["id"]
            
            # --- State Reducer Logic ---
            if event_type == "STATUS":
                self.state["status"] = data["message"]
                if "stage" in data:
                    self.state["current_stage"] = data["stage"]
                if "iteration" in data:
                     self.state["iteration"] = data["iteration"]

            elif event_type == "INIT":
                self.state["participant_id"] = data["participant_id"]
                self.state["target"] = data["target"]
                self.state["control"] = data.get("control")
                self.state["max_iterations"] = data.get("max_iterations", 1)
                self.state["config"] = data.get("config", {}) # Store token config
                self.state["start_time"] = timestamp
                self.state["status"] = "Initializing Engine..."
                self.state["current_stage"] = 0
                self.state["steps"] = [] 
                self.state["history"] = []
                self.state["plans"] = {} # Store plans by iteration
                self.state["prediction"] = None
                self.state["critic"] = None
                self.state["critic_summary"] = None
                self.state["total_tokens"] = 0
                self.state["progress"] = 0
                self.state["completed"] = False
                self.state["completion"] = None
                
            elif event_type == "PLAN":
                current_iter = self.state.get("iteration", 1)
                
                # Store full plan
                if "plans" not in self.state: self.state["plans"] = {}
                self.state["plans"][str(current_iter)] = data.get("plan", {})
                
                # Legacy support for progress bar
                self.state["max_steps"] = data.get("steps", 10)
                self.state["status"] = f"Orchestrating Plan (Iteration {current_iter})..."
                self.state["current_stage"] = 1
                
                # Archive previous steps steps to history
                if self.state["steps"]:
                    # Mark them as historical if needed, or distinct
                    self.state.setdefault("history", []).extend(self.state["steps"])
                
                self.state["steps"] = [] # Clear for new plan steps
                # Clear previous results to prevent stale modal
                self.state["prediction"] = None
                self.state["critic_summary"] = None
                self.state["critic"] = None
                self.state["progress"] = 0
                
            elif event_type == "STEP_START":
                existing = next((s for s in self.state["steps"] if s["id"] == data["id"]), None)
                if not existing:
                    self.state["steps"].append({
                        "id": data["id"],
                        "tool": data["tool"],
                        "desc": data["desc"],
                        "status": "running",
                        "tokens": 0,
                        "startTime": time.time(),
                        "duration": 0,
                        "iteration": self.state.get("iteration", 1)
                    })
                if "stage" in data and data["stage"] is not None:
                    self.state["current_stage"] = data["stage"]
                else:
                    # Fallback inference for backward compatibility.
                    step_id = int(data.get("id") or 0)
                    if step_id >= 930:
                        self.state["current_stage"] = 6
                    elif step_id >= 920:
                        self.state["current_stage"] = 5
                    elif step_id >= 910:
                        self.state["current_stage"] = 4
                    elif step_id >= 900:
                        self.state["current_stage"] = 3
                    else:
                        self.state["current_stage"] = 2
                self.state["status"] = f"Running Step {data['id']}: {data['tool']}"
                
            elif event_type == "STEP_COMPLETE":
                for s in self.state["steps"]:
                    if s["id"] == data["id"]:
                        s["status"] = "complete"
                        s["tokens"] = data["tokens"]
                        s["preview"] = data.get("preview", "")
                        if "startTime" in s:
                            s["duration"] = round(time.time() - s["startTime"], 2)
                self.state["total_tokens"] += data["tokens"]
                self.state["progress"] += 1
                self.state["status"] = "Step Complete"
                
            elif event_type == "STEP_FAIL":
                for s in self.state["steps"]:
                    if s["id"] == data["id"]:
                        s["status"] = "failed"
                        s["error"] = data["error"]
                self.state["status"] = "Step Failed"
            
            elif event_type == "REPAIR":
                 for s in self.state["steps"]:
                    if s["id"] == data["id"]:
                        s["status"] = "repairing"
                        s["msg"] = data["strategy"]
                 self.state["status"] = "Attempting Auto-Repair"

            elif event_type == "FUSION":
                self.state["fusion_data"] = data
                self.state["status"] = "Fusion Complete"
                self.state["current_stage"] = 3
                # Step is now handled via explicit STEP_START/COMPLETE in executor

            elif event_type == "PREDICTION":
                self.state["prediction"] = data
                self.state["status"] = "Prediction Generated"
                self.state["current_stage"] = 4 
                # Add or update virtual step for Predictor
                pred_id = 910 + self.state.get("iteration", 1)
                existing = next((s for s in self.state["steps"] if s["id"] == pred_id), None)
                if existing:
                    existing["status"] = "complete"
                    existing["tokens"] = 0
                    existing["desc"] = f"Generated prediction: {data.get('result', 'Unknown')} ({data.get('prob', 0):.1%})"
                    if "startTime" in existing:
                        existing["duration"] = round(time.time() - existing["startTime"], 2)
                else:
                    self.state["steps"].append({
                        "id": pred_id,
                        "tool": "Predictor Agent",
                        "desc": f"Generated prediction: {data.get('result', 'Unknown')} ({data.get('prob', 0):.1%})",
                        "status": "complete",
                        "tokens": 0,
                        "startTime": time.time(),
                        "duration": 0.5,
                        "iteration": self.state.get("iteration", 1)
                    })

            elif event_type == "CRITIC":
                self.state["status"] = f"Critic Verdict: {data['verdict']}"
                self.state["critic_summary"] = data.get("summary", "") 
                self.state["critic"] = data
                self.state["current_stage"] = 5
                
                # Determine status based on verdict
                verdict = data.get('verdict', 'UNKNOWN')
                is_pass = verdict == 'SATISFACTORY'
                step_status = "complete" if is_pass else "failed"
                
                # Add virtual step for Critic
                self.state["steps"].append({
                    "id": 920 + self.state["iteration"],
                    "tool": "Critic Agent",
                    "desc": f"Verdict: {verdict}",
                    "preview": data.get("summary", "No details provided."),
                    "status": step_status, 
                    "tokens": 0,
                    "startTime": time.time(),
                    "duration": 0.5,
                    "iteration": self.state.get("iteration", 1)
                })

            elif event_type == "COMPLETE":
                # Ensure we don't duplicate logic, just set status
                self.state["status"] = "Pipeline Completed"
                self.state["progress"] = self.state["max_steps"] 
                # Always snap to final stage in case new stages are added (e.g., Communication)
                self.state["current_stage"] = max(0, len(self.state.get("stages", [])) - 1)
                self.state["completed"] = True
                self.state["completion"] = data

    def get_snapshot(self, since_id=0):
        with self._lock:
            # Capture System Metrics (Optional)
            try:
                import psutil
                process = psutil.Process(os.getpid())
                cpu_pct = psutil.cpu_percent(interval=None) 
            except (ImportError, Exception):
                pass
            
            new_events = [e for e in self.events if e["id"] > int(since_id)]
            return {
                "state": self.state,
                "events": new_events
            }

# --- GLOBAL SINGLETONS ---
_event_store = EventStore()
_ui_instance = None
_launcher_callback: Optional[Callable[[Dict], None]] = None

# --- CUSTOM JSON ENCODER ---
from enum import Enum
import uuid

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, 'dict'): # Pydantic v1
            return obj.dict()
        if hasattr(obj, 'model_dump'): # Pydantic v2
            return obj.model_dump()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# --- FLASK APP ---
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.json_encoder = CustomJSONEncoder

# For newer Flask versions, we might need to override json.provider
class CustomProvider(DefaultJSONProvider):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, 'dict'):
            return obj.dict()
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, '__dict__'):
             return obj.__dict__
        return super().default(obj)

app.json = CustomProvider(app)

@app.route('/api/inputs')
def list_inputs():
    p_dir = _event_store.state.get("participant_dir")
    if not p_dir or not os.path.exists(p_dir):
        return jsonify([])
    
    try:
        files = [f for f in os.listdir(p_dir) if os.path.isfile(os.path.join(p_dir, f))]
        return jsonify(sorted(files))
    except Exception:
        return jsonify([])

@app.route('/api/inputs/content')
def get_input_content():
    filename = request.args.get('file')
    p_dir = _event_store.state.get("participant_dir")
    
    if not p_dir or not os.path.exists(p_dir):
        return "No participant data found", 404
        
    path = os.path.join(p_dir, filename)
    if os.path.exists(path) and os.path.isfile(path):
        return send_file(path)
    return "File not found", 404

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/snapshot')
def snapshot():
    since_id = request.args.get('since_id', 0)
    return jsonify(_event_store.get_snapshot(since_id))

@app.route('/api/launch', methods=['POST'])
def launch():
    data = request.json
    
    # Extract new config args
    config = {
        "id": data.get('id'),
        "target": data.get('target', 'neuropsychiatric'),
        "control": data.get('control'),
        "backend": data.get('backend'),
        "model": data.get('model'),
        "max_tokens": data.get('max_tokens'),
        "local_engine": data.get('local_engine'),
        "local_dtype": data.get('local_dtype'),
        "local_quant": data.get('local_quant'),
        "local_kv_cache_dtype": data.get('local_kv_cache_dtype'),
        "local_attn": data.get('local_attn'),
        "local_tensor_parallel": data.get('local_tensor_parallel'),
        "local_pipeline_parallel": data.get('local_pipeline_parallel'),
        "local_gpu_mem_util": data.get('local_gpu_mem_util'),
        "local_max_model_len": data.get('local_max_model_len'),
        "local_enforce_eager": data.get('local_enforce_eager'),
        "local_trust_remote_code": data.get('local_trust_remote_code'),
        "total_budget": data.get('total_budget'),
        "max_agent_input": data.get('max_agent_input'),
        "max_tool_output": data.get('max_tool_output'),
        "max_agent_output": data.get('max_agent_output'),
        "max_tool_input": data.get('max_tool_input')
    }
    
    if _launcher_callback:
        # Pass the full config dict to the callback
        threading.Thread(target=_launcher_callback, args=(config,), daemon=True).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "no_callback"}), 500

@app.route('/api/outputs')
def list_outputs():
    # List all files in results/participant_{id} if we can know participant ID
    pid = _event_store.state.get("participant_id")
    if not pid or pid == "Unknown":
        return jsonify([])
        
    # Locate results dir
    # From settings: ../../results (based on current location in frontend/ or utils/)
    # Current file is in frontend/, so root is ../
    # results is ../results
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # results is sibling of multi_agent_system, so project_root (parent of base) joined with results
    results_dir = os.path.join(os.path.dirname(base), "results")
    
    # Try finding folder
    target_dir = None
    if not os.path.exists(results_dir):
         # Try creating it or looking at old path relative to project
         pass

    # Safe fallback if results_dir check fails
    possible_names = [f"participant_{pid}", f"participant_ID{pid}", pid, f"ID{pid}"]
    for name in possible_names:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            target_dir = path
            break
            
    if not target_dir:
        return jsonify([])
        
    files = [f for f in os.listdir(target_dir) if f.endswith('.md') or f.endswith('.json') or f.endswith('.txt')]
    return jsonify(sorted(files))

@app.route('/api/outputs/content')
def get_output_content():
    filename = request.args.get('file')
    pid = _event_store.state.get("participant_id")
    if not pid : return "No participant active", 400
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # results is sibling of multi_agent_system, so project_root (parent of base) joined with results
    results_dir = os.path.join(os.path.dirname(base), "results")
    
    # Same logic to find dir
    target_dir = None
    possible_names = [f"participant_{pid}", f"participant_ID{pid}", pid, f"ID{pid}"]
    for name in possible_names:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            target_dir = path
            break
            
    if target_dir:
        return send_file(os.path.join(target_dir, filename))
    return "File not found", 404

class FlaskUI:
    """Interface exposed to the pipeline."""
    def __init__(self):
        self.enabled = False
        self.server_thread = None

    def start_server(self, port=5005):
        def run():
            app.run(port=port, debug=False, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run, daemon=True)
        self.server_thread.start()
        self.enabled = True
        print(f"[*] Dashboard live at http://127.0.0.1:{port}")
        
        def open_browser():
            time.sleep(1.5)
            # webbrowser.open(f"http://127.0.0.1:{port}")
            pass
        
        threading.Thread(target=open_browser, daemon=True).start()

    def set_status(self, message, stage=None, iteration=None):
        data = {"message": message}
        if stage is not None: data["stage"] = stage
        if iteration is not None: data["iteration"] = iteration
        _event_store.add_event("STATUS", data)

    def on_pipeline_start(self, participant_id, target, control=None, participant_dir=None, max_iterations=3, token_config=None):
        _event_store.add_event("INIT", {
            "participant_id": participant_id, 
            "target": target,
            "control": control,
            "max_iterations": max_iterations,
            "config": token_config or {}
        })
        if participant_dir:
            _event_store.state["participant_dir"] = participant_dir
        
    def on_plan_created(self, plan):
        # Support both object and legacy (though legacy should be gone)
        if hasattr(plan, 'total_steps'):
             steps = plan.total_steps
             domains = plan.priority_domains
             plan_dict = plan.dict() if hasattr(plan, 'dict') else plan.__dict__
        else:
             steps = plan.get('total_steps', 0)
             domains = plan.get('priority_domains', [])
             plan_dict = plan

        _event_store.add_event("PLAN", {
            "steps": steps, 
            "domains": domains,
            "plan": plan_dict
        })
        
    def on_step_start(self, step_id, tool_name, description, parallel_with=None, stage=None):
        payload = {"id": step_id, "tool": tool_name, "desc": description}
        if stage is not None:
            payload["stage"] = stage
        _event_store.add_event("STEP_START", payload)
        
    def on_step_complete(self, step_id, tokens, duration_ms, preview=""):
        _event_store.add_event("STEP_COMPLETE", {"id": step_id, "tokens": tokens, "preview": preview})
        
    def on_step_failed(self, step_id, error):
        _event_store.add_event("STEP_FAIL", {"id": step_id, "error": error})
        
    def on_auto_repair(self, step_id, strategy):
        _event_store.add_event("REPAIR", {"id": step_id, "strategy": strategy})
        
    def on_fusion_complete(self, fusion_data):
        _event_store.add_event("FUSION", fusion_data)

    def on_prediction(self, classification, probability, confidence):
        # Normalize label for UI (CASE/CONTROL)
        label = "UNKNOWN"
        if isinstance(classification, str):
            upper = classification.upper()
            if "CONTROL" in upper:
                label = "CONTROL"
            elif "CASE" in upper:
                label = "CASE"
        _event_store.add_event("PREDICTION", {
            "result": classification,
            "label": label,
            "prob": probability,
            "confidence": confidence
        })
        self.set_status(f"Predictor Assessment: {label} ({probability:.1%})", stage=4)
        
    def on_critic_verdict(
        self,
        verdict,
        confidence,
        checklist_passed,
        checklist_total,
        summary="",
        checklist=None,
        weaknesses=None,
        improvement_suggestions=None,
        domains_missed=None,
        composite_score=None,
        score_breakdown=None,
        iteration=None
    ):
        _event_store.add_event("CRITIC", {
            "verdict": verdict,
            "confidence": confidence,
            "passed": checklist_passed,
            "total": checklist_total,
            "summary": summary,
            "checklist": checklist or {},
            "weaknesses": weaknesses or [],
            "improvement_suggestions": improvement_suggestions or [],
            "domains_missed": domains_missed or [],
            "composite_score": composite_score,
            "score_breakdown": score_breakdown or {},
            "iteration": iteration
        })
        self.set_status(f"Critic Evaluation: {verdict}", stage=5)
        
    def on_pipeline_complete(self, result, probability, iterations, total_duration_secs, total_tokens):
        _event_store.add_event("COMPLETE", {
            "result": result,
            "probability": probability,
            "iterations": iterations,
            "duration": total_duration_secs,
            "tokens": total_tokens
        })

def get_ui(enabled=True):
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = FlaskUI()
    return _ui_instance

def reset_ui():
    global _ui_instance
    _ui_instance = None

def start_ui_loop(launcher_callback: Callable[[Dict], None]):
    """Start server and wait for user to launch via UI."""
    global _launcher_callback
    _launcher_callback = launcher_callback
    
    ui = get_ui()
    ui.start_server()
    
    print("[*] Dashboard ready. Waiting for user input via Web UI...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
