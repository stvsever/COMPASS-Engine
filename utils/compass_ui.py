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
from flask import Flask, render_template_string, jsonify, request, send_file, make_response
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
            "status": "Ready to Launch",
            "start_time": None,
            "total_tokens": 0,
            "progress": 0,
            "max_steps": 1, 
            "steps": [],
            "prediction": None,
            "latest_update_id": 0,
            "current_stage": -1, # -1: Setup, 0:Init, 1:Plan, 2:Exec, 3:Predict, 4:Evaluate
            "stages": ["Initialization", "Orchestration", "Execution", "Prediction", "Evaluation"],
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
                self.state["max_iterations"] = data.get("max_iterations", 1)
                self.state["start_time"] = timestamp
                self.state["status"] = "Initializing Engine..."
                self.state["current_stage"] = 0
                self.state["steps"] = [] # Clear steps on new run
                
            elif event_type == "PLAN":
                current_iter = self.state.get("iteration", 1)
                if current_iter > 1 and self.state["steps"]:
                    if "history" not in self.state: self.state["history"] = []
                    self.state["history"].append({
                        "iteration": current_iter - 1,
                        "steps": list(self.state["steps"])
                    })
                
                self.state["max_steps"] = data["steps"]
                self.state["status"] = f"Orchestrating Plan (Iteration {self.state.get('iteration', 1)})..."
                self.state["current_stage"] = 1
                self.state["steps"] = [] # Clear for new plan steps
                
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
            
            elif event_type == "PREDICTION":
                self.state["prediction"] = data
                self.state["status"] = "Prediction Generated"
                self.state["current_stage"] = 4 
                
            elif event_type == "CRITIC":
                self.state["status"] = f"Critic Verdict: {data['verdict']}"
                self.state["current_stage"] = 4
                
            elif event_type == "COMPLETE":
                self.state["status"] = "Pipeline Completed"
                self.state["progress"] = self.state["max_steps"] 

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
app = Flask(__name__)
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
    
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COMPASS | Intelligent Inference Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-body: #050505;
            --bg-sidebar: #0a0a0a;
            --bg-card: #121212;
            --bg-header: rgba(10, 10, 10, 0.8);
            --border: #27272a;
            --primary: #6366f1; /* Indigo-500 */
            --primary-glow: rgba(99, 102, 241, 0.3);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.2);
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #fafafa;
            --text-muted: #a1a1aa;
            --font-main: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }

        body { 
            background-color: var(--bg-body); 
            color: var(--text-main); 
            font-family: var(--font-main); 
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            font-size: 14px;
        }

        /* HEADER */
        header {
            background: var(--bg-header);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 0 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 60px;
            position: relative;
            z-index: 50;
        }
        
        .brand { font-size: 1.1rem; font-weight: 700; letter-spacing: -0.02em; display: flex; align-items: center; gap: 0.5rem; }
        .brand i { color: var(--primary); font-size: 1.2rem; }
        .brand span { opacity: 0.6; font-weight: 400; }
        
        /* RADAR SPINNER */
        .radar-spinner {
            width: 60px; height: 60px; position: relative; margin: 2rem auto;
            display: flex; justify-content: center; align-items: center;
        }
        .radar-ripple {
            position: absolute; border: 2px solid var(--primary); opacity: 0; border-radius: 50%;
            animation: ripple 2s cubic-bezier(0, 0.2, 0.8, 1) infinite;
        }
        .radar-ripple:nth-child(2) { animation-delay: -0.5s; }
        .radar-core { width: 12px; height: 12px; background: var(--primary); border-radius: 50%; box-shadow: 0 0 15px var(--primary); }
        
        @keyframes ripple {
            0% { width: 0; height: 0; opacity: 1; }
            100% { width: 100%; height: 100%; opacity: 0; }
        }
        
        .waiting-status { 
            text-align: center; color: var(--text-muted); font-size: 0.9rem; margin-top: 1rem; letter-spacing: 0.05em; 
            text-transform: uppercase; font-weight: 600;
        }

        /* PIPELINE STEPPER */
        .stepper { display: flex; align-items: center; gap: 0.5rem; background: rgba(255,255,255,0.03); padding: 6px 12px; border-radius: 20px; border: 1px solid var(--border); }
        .stepper-item { 
            display: flex; align-items: center; gap: 0.4rem; 
            font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: #52525b;
            transition: all 0.3s ease;
        }
        .stepper-item.active { color: var(--primary); text-shadow: 0 0 10px var(--primary-glow); }
        .stepper-item.completed { color: var(--success); }
        .stepper-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
        .stepper-line { width: 14px; height: 1px; background: #27272a; }
        
        /* LAYOUT */
        .container {
            display: flex;
            flex: 1;
            overflow: hidden; 
        }

        .main-panel {
            flex: 1;
            padding: 1.5rem;
            overflow-y: auto;
            position: relative;
            background: radial-gradient(circle at top right, #1a1a2e 0%, transparent 40%);
        }

        .log-panel {
            width: 380px;
            background: var(--bg-sidebar);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: width 0.3s;
        }

        .log-header {
            padding: 0.8rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0,0,0,0.2);
            text-transform: uppercase;
        }

        .log-content {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            font-family: var(--font-mono);
            font-size: 0.75rem;
            line-height: 1.5;
            background: #080808;
        }

        /* STATS */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr); /* Removed Memory MB */
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.2rem;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s, border-color 0.2s;
        }
        .stat-card:hover { border-color: #404040; transform: translateY(-2px); }
        
        .stat-icon { position: absolute; top: 1rem; right: 1rem; opacity: 0.1; font-size: 2rem; }
        .stat-title { font-size: 0.7rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.3rem; letter-spacing: 0.05em; }
        .stat-value { font-size: 1.5rem; font-weight: 600; font-family: var(--font-mono); display: flex; align-items: center; gap: 0.5rem; }
        
        /* TIMELINE & TABS */
        .timeline-wrapper {
            background: #09090b; /* Zinc-950 */
            border: 1px solid #27272a;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            min-height: 500px; /* Taller default */
        }
        
        .timeline-tabs {
            display: flex;
            background: #121212;
            border-bottom: 1px solid #27272a;
            padding: 4px;
            gap: 4px;
        }
        
        .tab-btn {
            flex: 1;
            background: transparent;
            border: none;
            color: #71717a; /* Zinc-500 */
            padding: 10px 16px;
            font-size: 0.85rem;
            font-weight: 600;
            font-family: var(--font-main);
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        
        .tab-btn:hover {
            color: #e4e4e7;
            background: rgba(255,255,255,0.03);
        }
        
        .tab-btn.active {
            color: #fff;
            background: #27272a;
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        
        .tab-btn i { font-size: 0.9rem; opacity: 0.8; }
        .tab-btn.active i { color: var(--primary); opacity: 1; }

        /* TAB CONTENT */
        .tab-content { 
            display: none; 
            flex: 1;
            opacity: 0; 
            animation: fadeIn 0.3s forwards; 
            position: relative;
            background: #09090b;
        }
        .tab-content.active { display: block; }

        /* RESULTS VIEW STYLING */
        .results-container {
            display: flex;
            height: 600px; /* Fixed height for scrolling */
            border-top: 1px solid #27272a;
        }
        
        .file-list-pane {
            width: 280px;
            background: #0c0c0e;
            border-right: 1px solid #27272a;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        
        .file-list-header {
            padding: 12px 16px;
            font-size: 0.75rem;
            font-weight: 700;
            color: #52525b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid #27272a;
            background: #121212;
            position: sticky; top: 0;
        }

        .file-item { 
            padding: 12px 16px; 
            font-size: 0.85rem; 
            color: #a1a1aa; 
            border-bottom: 1px solid #18181b; 
            cursor: pointer; 
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            transition: all 0.2s; 
        }
        
        .file-item:hover { 
            background: #18181b; 
            color: #fff; 
            padding-left: 20px; 
        }
        
        .file-item.active { 
            background: #1d1d20; 
            color: var(--primary); 
            border-right: 3px solid var(--primary); 
        }

        .file-preview-pane {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #050505;
            overflow: hidden;
        }
        
        .preview-header {
            padding: 10px 20px;
            border-bottom: 1px solid #27272a;
            background: #09090b;
            color: #d4d4d8;
            font-size: 0.85rem;
            font-family: var(--font-mono);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .file-content-view { 
            flex: 1;
            padding: 1.5rem; 
            
            font-family: 'JetBrains Mono', monospace; 
            font-size: 0.85rem; 
            line-height: 1.6;
            color: #d4d4d8;
            
            overflow-y: auto;
            white-space: pre-wrap;
        }

        /* INSPECTOR */
        #inspector-content {
            padding: 2rem;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            color: #52525b;
        }
        
        /* STEPS */
        .step-list { 
            padding: 1.5rem;
            display: flex; 
            flex-direction: column; 
            gap: 1rem; 
        }
        
        .step {
            display: flex; gap: 1rem; 
            opacity: 0.6; transition: all 0.4s ease;
        }
        .step.active { opacity: 1; transform: scale(1.01); }
        .step.active .step-card { border-color: var(--primary); box-shadow: 0 0 15px rgba(99, 102, 241, 0.1); }
        .step.failed .step-card { border-color: var(--danger); }
        
        .step-marker { display: flex; flex-direction: column; align-items: center; width: 24px; }
        .step-circle { 
            width: 10px; height: 10px; border-radius: 50%; 
            background: var(--bg-card); border: 2px solid var(--border);
            transition: all 0.3s; margin-top: 1.2rem;
        }
        .step.active .step-circle { border-color: var(--primary); background: var(--primary); box-shadow: 0 0 10px var(--primary); }
        .step.complete .step-circle { border-color: var(--success); background: var(--success); }
        .step-rail { flex: 1; width: 2px; background: var(--border); margin: 0.5rem 0; opactiy: 0.5; }

        .step-card {
            flex: 1; background: #18181b; border: 1px solid var(--border); border-radius: 8px; padding: 1rem; 
        }
        .step-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; align-items: center; }
        .step-tool { font-weight: 600; font-size: 0.95rem; color: #fff; display: flex; align-items: center; gap: 0.5rem; }
        .step-tool i { color: var(--primary); font-size: 0.8rem; }
        .step-desc { font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; }
        .step-meta { font-size: 0.75rem; color: #52525b; font-family: var(--font-mono); background: #000; padding: 2px 6px; border-radius: 4px; border: 1px solid #27272a; }
        
        .step-preview { 
            margin-top: 0.8rem; padding: 0.8rem; background: #0a0a0a; 
            border-radius: 6px; font-size: 0.75rem; font-family: var(--font-mono);
            border-left: 2px solid var(--primary); color: #d4d4d8;
        }

        /* OVERLAYS & MODALS */
        .modal-overlay {
            position: fixed; inset: 0;
            background: rgba(0,0,0,0.85); backdrop-filter: blur(8px);
            z-index: 1000; display: flex; align-items: center; justify-content: center;
            opacity: 0; pointer-events: none; transition: opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .modal-overlay.active { opacity: 1; pointer-events: all; }
        
        .glass-panel {
            background: rgba(18, 18, 18, 0.95); border: 1px solid #333;
            border-radius: 16px; padding: 2.5rem; width: 520px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            transform: translateY(30px); transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        .modal-overlay.active .glass-panel { transform: translateY(0); }
        
        .launch-icon { font-size: 3rem; color: var(--primary); margin-bottom: 1rem; text-align: center; display: block; }
        
        .form-label { display: block; color: var(--text-muted); margin-bottom: 0.6rem; font-size: 0.85rem; font-weight: 500; }
        .form-input { 
            width: 100%; background: #050505; border: 1px solid #333;
            padding: 0.9rem; border-radius: 8px; color: white; font-family: inherit; font-size: 0.9rem;
            transition: border-color 0.2s; margin-bottom: 1rem;
        }
        .form-input:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2); }
        
        .btn-launch {
            width: 100%; background: var(--primary); border: none; color: white;
            padding: 1rem; border-radius: 8px; font-weight: 600; cursor: pointer;
            transition: all 0.2s; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em;
            margin-top: 1rem; display: flex; justify-content: center; align-items: center; gap: 0.8rem;
        }
        .btn-launch:hover { background: #4f46e5; box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4); }
        .btn-secondary { background: #27272a; color: #fff; margin-top: 1rem; }
        .btn-secondary:hover { background: #3f3f46; }

        /* WIZARD */
        .wizard-step { display: none; animation: fadeIn 0.3s; }
        .wizard-step.active { display: block; }
        .wizard-progress { display: flex; gap: 5px; margin-bottom: 2rem; justify-content: center; }
        .wizard-dot { width: 40px; height: 4px; background: #333; border-radius: 2px; }
        .wizard-dot.active { background: var(--primary); }

        /* UTILS */
        .log-entry { margin-bottom: 6px; padding-left: 12px; border-left: 2px solid #333; animation: fadeIn 0.2s; }
        .log-ts { color: #52525b; font-size: 0.7em; margin-right: 8px; user-select: none; }
        .log-type { font-weight: bold; font-size: 0.75em; display: inline-block; width: 55px; }
        .type-INFO { color: var(--primary); }
        .type-SUCCESS { color: var(--success); }
        .type-WARNING { color: var(--warning); }
        .type-ERROR { color: var(--danger); }
        
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideUp { from { transform: translateY(40px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        
        #global-progress { 
            position: absolute; bottom: 0; left: 0; height: 2px; background: var(--primary); 
            box-shadow: 0 0 10px var(--primary); transition: width 0.4s ease;
            width: 0%;
        }
        
    </style>
</head>
<body>
    
    <!-- SETUP MODAL (WIZARD) -->
     <div id="setup-modal" class="modal-overlay active">
        <div class="glass-panel">
            
            <!-- STEP 1: Mission Data (Only Step) -->
            <div id="wiz-1" class="wizard-step active">
                <i class="fas fa-satellite-dish launch-icon"></i>
                <h2 style="font-size: 1.5rem; text-align: center; margin-bottom: 0.5rem;">Mission Data</h2>
                <div style="margin-bottom: 1.5rem;">
                    <label class="form-label">PARTICIPANT IDENTIFIER (or path)</label>
                    <input type="text" id="input-pid" class="form-input" placeholder="e.g. participant_ID6015951 or 6015951" value="">
                </div>
                
                <div>
                    <label class="form-label">TARGET PHENOTYPE</label>
                    <input type="text" id="input-target" class="form-input" placeholder="e.g. neuropsychiatric" value="neuropsychiatric">
                </div>
                
                <button id="btn-final-launch" class="btn-launch" onclick="launchPipeline()">
                    <span>Launch Pipeline</span>
                    <i class="fas fa-rocket"></i>
                </button>
            </div>

        </div>
    </div>

    <!-- MAIN HEADER -->
    <header>
        <div class="brand">
            <i class="fas fa-compass"></i>
            COMPASS <span>DASHBOARD</span>
        </div>
        
        <div class="stepper" id="stepper">
            <div class="stepper-item">Initializing...</div>
        </div>
        
        <div style="font-size: 0.75rem; color: var(--text-muted); display:flex; align-items:center; gap:0.5rem;">
            <i class="fas fa-circle" id="status-dot" style="font-size: 6px; color: var(--success);"></i>
            <span id="session-status">System Ready</span>
        </div>
        
        <div id="global-progress"></div>
    </header>

    <div class="container">
        <!-- MAIN PANEL -->
        <main class="main-panel" id="main-scroller">
            
            <!-- STATS CARDS -->
            <div class="stats-grid">
                <div class="stat-card">
                    <i class="fas fa-clock stat-icon"></i>
                    <div class="stat-title">Elapsed Time</div>
                    <div class="stat-value" id="timer">00:00</div>
                </div>
                <div class="stat-card">
                    <i class="fas fa-coins stat-icon"></i>
                    <div class="stat-title">Token Usage</div>
                    <div class="stat-value" id="token-display">0</div>
                </div>
                <!-- REMOVED MEMORY CARD -->
                <div class="stat-card">
                    <i class="fas fa-layer-group stat-icon"></i>
                    <div class="stat-title">Iteration</div>
                    <div class="stat-value"><span id="iter-display">1</span></div>
                </div>
            </div>

            <!-- TIMELINE -->
            <div class="timeline-wrapper">
                <div class="timeline-tabs">
                    <button class="tab-btn active" onclick="switchTab('timeline')"><i class="fas fa-stream"></i> Live Execution</button>
                    <button class="tab-btn" onclick="switchTab('results'); loadResults();"><i class="fas fa-file-alt"></i> Results</button>
                    <button class="tab-btn" onclick="switchTab('inspector'); loadInputs();"><i class="fas fa-code"></i> Data Inspector</button>
                </div>

                <!-- TAB: TIMELINE -->
                <div id="tab-timeline" class="tab-content active" style="padding:0;">
                    <div class="step-list" id="steps-container">
                        <div style="text-align:center; padding: 4rem; color: var(--text-muted); opacity: 0.5;">
                            <i class="fas fa-satellite-dish" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                            <p>Pipeline is ready. Launching mission...</p>
                        </div>
                    </div>
                </div>

                <!-- RESULTS TAB -->
                <div id="tab-results" class="tab-content" style="padding:0;">
                    <div class="results-container">
                        <div class="file-list-pane">
                             <div class="file-list-header">Generated Artifacts</div>
                             <div id="file-list-container">
                                 <!-- JS injected -->
                                 <div style="padding:1rem; color:#444; font-style:italic;">No files yet...</div>
                             </div>
                        </div>
                        <div class="file-preview-pane">
                            <div class="preview-header">
                                <span id="current-file-name">Preview</span>
                                <span style="opacity:0.5; font-size:0.75rem;">READ-ONLY</span>
                            </div>
                            <div id="file-preview-container" class="file-content-view">
                                <div style="color: #333; text-align: center; margin-top: 4rem;">
                                    <i class="fas fa-file-alt" style="font-size:2rem; margin-bottom:1rem; opacity:0.3;"></i>
                                    <p>Select a file from the list to inspect contents.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- TAB: INSPECTOR -->
                <!-- TAB: INSPECTOR -->
                <div id="tab-inspector" class="tab-content" style="padding:0;">
                    <div class="results-container">
                        <div class="file-list-pane">
                             <div class="file-list-header">Input Data Files</div>
                             <div id="input-list-container">
                                 <div style="padding:1rem; color:#444; font-style:italic;">Loading...</div>
                             </div>
                        </div>
                        <div class="file-preview-pane">
                            <div class="preview-header">
                                <span id="current-input-name">Preview</span>
                                <span style="opacity:0.5; font-size:0.75rem;">READ-ONLY</span>
                            </div>
                            <div id="input-preview-container" class="file-content-view">
                                <div style="color: #333; text-align: center; margin-top: 4rem;">
                                    <i class="fas fa-database" style="font-size:2rem; margin-bottom:1rem; opacity:0.3;"></i>
                                    <p>Select an input file to inspect source data.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PREDICTION HERO -->
            <div id="prediction-hero" class="prediction-hero" style="display:none; text-align:center; margin-top:20px; padding:20px; border:1px solid #333; border-radius:10px;">
                <span class="verdict-badge">Assessment Complete</span>
                <div class="verdict-line" id="final-verdict-text">--</div>
                <div style="font-size: 1.1rem; color: var(--text-muted);">
                    Confidence Level: <span style="color:white; font-weight:600;" id="final-confidence">--</span>
                </div>
            </div>

        </main>

        <!-- SIDEBAR LOGS -->
        <aside class="log-panel" id="log-panel">
            <div class="log-header">
                <span><i class="fas fa-terminal" style="margin-right:8px;"></i> System Logs</span>
                <div style="display:flex; gap:10px;">
                    <i class="fas fa-sync-alt" id="activity-spinner" style="opacity:0;"></i>
                    <button onclick="toggleLogScroll()" id="btn-log-scroll" style="background:none; border:none; color:var(--primary); cursor:pointer; font-size:0.7rem; font-weight:600;">AUTO: ON</button>
                </div>
            </div>
            <div class="log-content" id="log-container"></div>
        </aside>
    </div>

    <!-- LOGIC -->
    <script>
        let latestEventId = 0;
        let startTime = null;
        let autoScrollLogs = true;
        let isRunning = false;
        let currentStep = 1;

        // WIZARD LOGIC
        function nextStep() {
            document.getElementById('wiz-1').classList.remove('active');
            document.getElementById('wiz-2').classList.add('active');
            document.getElementById('dot-2').classList.add('active');
        }
        function prevStep() {
            document.getElementById('wiz-2').classList.remove('active');
            document.getElementById('wiz-1').classList.add('active');
            document.getElementById('dot-2').classList.remove('active');
        }
        function toggleModelInput() {
            const val = document.getElementById('input-backend').value;
            document.getElementById('group-model').style.display = val === 'local' ? 'block' : 'none';
        }

        async function launchPipeline() {
            const pid = document.getElementById('input-pid').value;
            const target = document.getElementById('input-target').value;
            
            // Hardcoded defaults as requested
            const backend = "openai";
            const model = "gpt-5"; 
            const tokens = 128000;

            if (!pid) return alert("Participant ID required");

            const btn = document.getElementById('btn-final-launch');
            btn.innerHTML = '<i class="fas fa-circle-notch spin"></i> Launching...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/launch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        id: pid, target: target, 
                        backend: backend, model: model, max_tokens: tokens
                    })
                });
                const data = await res.json();
                
                if (data.status === 'started') {
                    setTimeout(() => {
                         document.getElementById('setup-modal').classList.remove('active');
                         document.getElementById('session-status').textContent = "Pipeline Active";
                         document.getElementById('status-dot').classList.add('pulse');
                         isRunning = true;
                    }, 800);
                }
            } catch (e) {
                console.error(e);
                btn.innerHTML = 'Launch Failed';
                btn.disabled = false;
                alert("Connection failed.");
            }
        }

        function toggleLogScroll() {
            autoScrollLogs = !autoScrollLogs;
            document.getElementById('btn-log-scroll').textContent = autoScrollLogs ? 'AUTO: ON' : 'AUTO: OFF';
        }
        
        function switchTab(tab) {
             document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
             document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
             document.getElementById('tab-'+tab).classList.add('active');
             event.target.classList.add('active');
        }

        function renderSteps(steps, currentStage) {
            const container = document.getElementById('steps-container');
            
            // Special state for Orchestration (Stage 1) waiting
            if ((!steps || steps.length === 0) && currentStage === 1) {
                container.innerHTML = `
                    <div class="radar-spinner">
                        <div class="radar-ripple"></div>
                        <div class="radar-ripple"></div>
                        <div class="radar-core"></div>
                    </div>
                    <div class="waiting-status">Orchestrator creating execution plan...</div>
                `;
                return;
            }
            
            if (!steps || steps.length === 0) {
                 container.innerHTML = '';
                 return; 
            }

            const html = steps.map(step => `
                <div class="step ${step.status} ${step.status === 'running' ? 'active' : ''}" id="step-${step.id}">
                    <div class="step-marker">
                        <div class="step-circle"></div>
                        <div class="step-rail"></div>
                    </div>
                    <div class="step-card">
                        <div class="step-header">
                            <div class="step-tool"><i class="fas fa-wrench"></i> ${step.tool}</div>
                            <div class="step-meta">${step.duration ? step.duration + 's' : (step.status==='running' ? 'running' : 'pending')}</div>
                        </div>
                        <div class="step-desc">${step.desc}</div>
                        ${step.preview ? `<div class="step-preview">${step.preview}</div>` : ''}
                        ${step.error ? `<div class="step-preview" style="border-color:var(--danger); color:#fca5a5;">${step.error}</div>` : ''}
                    </div>
                </div>
            `).join('');
            container.innerHTML = html;
        }

        function renderJson(json) {
            return JSON.stringify(json, null, 2); 
        }

        // POLLING LOOP
        async function updateLoop() {
            try {
                const res = await fetch(`/api/snapshot?since_id=${latestEventId}`);
                const data = await res.json();
                const state = data.state;

                // Stats
                document.getElementById('token-display').textContent = state.total_tokens.toLocaleString();
                document.getElementById('iter-display').textContent = state.iteration || 1;
                
                // Timer
                if (state.start_time && !startTime) startTime = new Date();
                if (startTime && state.status !== "Pipeline Completed") {
                    const diff = Math.floor((new Date() - startTime) / 1000);
                    const m = Math.floor(diff / 60).toString().padStart(2, '0');
                    const s = (diff % 60).toString().padStart(2, '0');
                    document.getElementById('timer').textContent = `${m}:${s}`;
                }

                // Global Progress
                const pct = (state.progress / (state.max_steps || 1)) * 100;
                document.getElementById('global-progress').style.width = `${Math.min(pct, 100)}%`;

                // Stepper Header
                const stepper = document.getElementById('stepper');
                const curStage = state.current_stage || 0;
                if (state.stages) {
                    stepper.innerHTML = state.stages.map((st, i) => `
                        <div class="stepper-item ${i < curStage ? 'completed' : ''} ${i === curStage ? 'active' : ''}">
                            <div class="stepper-dot"></div>${st}
                        </div>
                        ${i < state.stages.length - 1 ? '<div class="stepper-line"></div>' : ''}
                    `).join('');
                }

                // Render Steps (Main Timeline)
                renderSteps(state.steps, state.current_stage);

                // Prediction Result
                if (state.prediction) {
                     document.getElementById('prediction-hero').style.display = 'block';
                     document.getElementById('final-verdict-text').textContent = state.prediction.result;
                     document.getElementById('final-confidence').textContent = (state.prediction.prob*100).toFixed(1) + "%";
                     document.getElementById('status-dot').style.color = state.prediction.result === 'CASE' ? 'var(--danger)' : 'var(--success)';
                }

                // Logs
                if (data.events.length > 0) {
                     const logCon = document.getElementById('log-container');
                     document.getElementById('activity-spinner').style.opacity = 1;
                     setTimeout(()=>document.getElementById('activity-spinner').style.opacity=0, 500);

                     data.events.forEach(e => {
                        if (e.id <= latestEventId) return;
                        latestEventId = e.id;
                        const div = document.createElement('div');
                        div.className = 'log-entry';
                        div.innerHTML = `<span class="log-ts">${e.time}</span><span class="log-type type-${e.type}">${e.type}</span> <span>${JSON.stringify(e.data).substring(0, 100)}</span>`;
                        logCon.appendChild(div);
                     });
                     if (autoScrollLogs) logCon.scrollTop = logCon.scrollHeight;
                }
                
                // Auto scroll main timeline
                if (isRunning && state.steps.length > 0) {
                     const activeStep = document.querySelector('.step.active');
                     if (activeStep && !document.getElementById('tab-results').classList.contains('active')) {
                         activeStep.scrollIntoView({behavior: "smooth", block: "nearest"});
                     }
                }

            } catch (err) { console.error(err); }
        }
        
        async function loadResults() {
            try {
                // Get participant ID from state or input
                // It is safer to ask API for output directory content
                const res = await fetch('/api/outputs');
                const files = await res.json();
                
                const container = document.getElementById('file-list-container');
                if (files.length === 0) {
                     container.innerHTML = '<div style="color:#666">No output files found.</div>';
                     return;
                }
                
                container.innerHTML = files.map(f => `
                    <div class="file-item" onclick="loadFile('${f}')">
                        <span><i class="fas fa-file-code"></i> ${f}</span>
                        <i class="fas fa-chevron-right" style="font-size:0.7rem; opacity:0.5;"></i>
                    </div>
                `).join('');
                
            } catch (e) { console.error(e); }
        }
        
        async function loadFile(filename) {
             document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
             event.currentTarget.classList.add('active');
             
             document.getElementById('current-file-name').textContent = filename;
             document.getElementById('file-preview-container').textContent = "Loading...";
             
             const res = await fetch(`/api/outputs/content?file=${filename}`);
             const text = await res.text();
             document.getElementById('file-preview-container').textContent = text;
        }

        async function loadInputs() {
            try {
                const res = await fetch('/api/inputs');
                const files = await res.json();
                
                const container = document.getElementById('input-list-container');
                if (files.length === 0) {
                     container.innerHTML = '<div style="padding:1rem; color:#666">No input files found.</div>';
                     return;
                }
                
                container.innerHTML = files.map(f => `
                    <div class="file-item" onclick="loadInputFile('${f}')">
                        <span><i class="fas fa-table"></i> ${f}</span>
                        <i class="fas fa-chevron-right" style="font-size:0.7rem; opacity:0.5;"></i>
                    </div>
                `).join('');
                
            } catch (e) { console.error(e); }
        }

        async function loadInputFile(filename) {
             document.querySelectorAll('#input-list-container .file-item').forEach(el => el.classList.remove('active'));
             event.currentTarget.classList.add('active');
             
             document.getElementById('current-input-name').textContent = filename;
             document.getElementById('input-preview-container').textContent = "Loading...";
             
             const res = await fetch(`/api/inputs/content?file=${filename}`);
             const text = await res.text();
             document.getElementById('input-preview-container').textContent = text;
        }

        function simpleMarkdown(text) {
             let html = text
                .replace(/^# (.*$)/gim, '<h1>$1</h1>')
                .replace(/^## (.*$)/gim, '<h2>$1</h2>')
                .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                .replace(/^\- (.*$)/gim, '• $1')
                .replace(/\*\*(.*)\*\*/gim, '<b>$1</b>');
             return html;
        }
        
        // Overwrite loadFile to use markdown
        async function loadFile(filename) {
             document.querySelectorAll('#file-list-container .file-item').forEach(el => el.classList.remove('active'));
             event.currentTarget.classList.add('active');
             
             document.getElementById('current-file-name').textContent = filename;
             document.getElementById('file-preview-container').innerHTML = "Loading...";
             
             const res = await fetch(`/api/outputs/content?file=${filename}`);
             const text = await res.text();
             
             if (filename.endsWith('.md')) {
                 document.getElementById('file-preview-container').innerHTML = simpleMarkdown(text);
             } else {
                  document.getElementById('file-preview-container').textContent = text;
             }
        }

        setInterval(updateLoop, 800);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

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
        "backend": data.get('backend'),
        "model": data.get('model'),
        "max_tokens": data.get('max_tokens')
    }
    
    if _launcher_callback:
        # Pass the full config dict to the callback
        threading.Thread(target=_launcher_callback, args=(config,), daemon=True).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "no_callback"}), 500

@app.route('/api/outputs')
def list_outputs():
    # List all files in results/participant_{id} if we can know participant ID
    # Or just list all recent outputs?
    # Better: List files for the CURRENT participant based on EventStore
    pid = _event_store.state.get("participant_id")
    if not pid or pid == "Unknown":
        return jsonify([])
        
    # Locate results dir
    # From settings: ../../results
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Navigate to results (sibling of project root in original structure, or ../results in new?)
    # Based on batch_run.py: PROJECT_ROOT.parent / "results"
    # Current file is in utils/, so root is ../
    # results is ../../results
    
    results_dir = os.path.join(os.path.dirname(base), "results")
    
    # Try finding folder
    target_dir = None
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

    def on_pipeline_start(self, participant_id, target, participant_dir=None, max_iterations=3):
        _event_store.add_event("INIT", {
            "participant_id": participant_id, 
            "target": target,
            "max_iterations": max_iterations
        })
        if participant_dir:
            _event_store.state["participant_dir"] = participant_dir
        
    def on_plan_created(self, plan_id, total_steps, domains):
        _event_store.add_event("PLAN", {"steps": total_steps, "domains": domains})
        
    def on_step_start(self, step_id, tool_name, description, parallel_with=None):
        _event_store.add_event("STEP_START", {"id": step_id, "tool": tool_name, "desc": description})
        
    def on_step_complete(self, step_id, tokens, duration_ms, preview=""):
        _event_store.add_event("STEP_COMPLETE", {"id": step_id, "tokens": tokens, "preview": preview})
        
    def on_step_failed(self, step_id, error):
        _event_store.add_event("STEP_FAIL", {"id": step_id, "error": error})
        
    def on_auto_repair(self, step_id, strategy):
        _event_store.add_event("REPAIR", {"id": step_id, "strategy": strategy})
        
    def on_fusion_complete(self, fusion_data):
        _event_store.add_event("FUSION", fusion_data)

    def on_prediction(self, classification, probability, confidence):
        _event_store.add_event("PREDICTION", {"result": classification, "prob": probability})
        
    def on_critic_verdict(self, verdict, confidence, checklist_passed, checklist_total):
        _event_store.add_event("CRITIC", {"verdict": verdict, "passed": checklist_passed})
        
    def on_pipeline_complete(self, result, probability, iterations, total_duration_secs, total_tokens):
        _event_store.add_event("COMPLETE", {"result": result})

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
