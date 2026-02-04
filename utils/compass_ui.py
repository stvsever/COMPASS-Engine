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
from flask import Flask, render_template_string, jsonify, request
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

# Disable flask logging for a cleaner terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class EventStore:
    """Thread-safe storage for pipeline events with optimized state tracking."""
    def __init__(self):
        self.reset()
        self._lock = threading.Lock()

    def reset(self):
        self.events = []
        self.state = {
            "participant_id": "Unknown",
            "target": "None",
            "status": "Ready to Launch",
            "start_time": None,
            "total_tokens": 0,
            "progress": 0,
            "max_steps": 1, 
            "steps": [],
            "prediction": None,
            "latest_update_id": 0,
            "current_stage": -1, # -1: Setup, 0:Init, 1:Plan, 2:Exec, 3:Predict, 4:Verify
            "stages": ["Initialization", "Orchestration", "Execution", "Prediction", "Verification"]
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

            elif event_type == "INIT":
                self.state["participant_id"] = data["participant_id"]
                self.state["target"] = data["target"]
                self.state["start_time"] = timestamp
                self.state["status"] = "Initializing Engine..."
                self.state["current_stage"] = 0
                
            elif event_type == "PLAN":
                self.state["max_steps"] = data["steps"]
                self.state["status"] = "Orchestrating Plan..."
                self.state["current_stage"] = 1
                
            elif event_type == "STEP_START":
                # Check if step exists vs new step
                existing = next((s for s in self.state["steps"] if s["id"] == data["id"]), None)
                if not existing:
                    self.state["steps"].append({
                        "id": data["id"],
                        "tool": data["tool"],
                        "desc": data["desc"],
                        "status": "running",
                        "tokens": 0,
                        "startTime": time.time(),
                        "duration": 0
                    })
                self.state["current_stage"] = 2 # Ensure we are in execution
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
                self.state["current_stage"] = 4 # Ready for verification
                
            elif event_type == "CRITIC":
                self.state["status"] = f"Critic Verdict: {data['verdict']}"
                self.state["current_stage"] = 4
                
            elif event_type == "COMPLETE":
                self.state["status"] = "Pipeline Completed"
                self.state["progress"] = self.state["max_steps"] # Force complete visual

    def get_snapshot(self, since_id=0):
        with self._lock:
            # Capture System Metrics (Optional)
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                cpu_pct = psutil.cpu_percent(interval=None) # Non-blocking
                
                self.state["system"] = {
                    "memory_mb": round(mem_mb, 1),
                    "cpu_percent": round(cpu_pct, 1)
                }
            except (ImportError, Exception):
                self.state["system"] = {
                    "memory_mb": 0,
                    "cpu_percent": 0
                }
            
            # Return full state but only new events to save bandwidth
            new_events = [e for e in self.events if e["id"] > int(since_id)]
            return {
                "state": self.state,
                "events": new_events
            }

# --- GLOBAL SINGLETONS ---
_event_store = EventStore()
_ui_instance = None
_launcher_callback: Optional[Callable[[str, str], None]] = None

# --- FLASK APP ---
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COMPASS | Intelligent Inference Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #09090b;
            --bg-card: #18181b;
            --bg-header: #18181b;
            --border: #27272a;
            --primary: #3b82f6;
            --primary-dim: rgba(59, 130, 246, 0.1);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #f4f4f5;
            --text-muted: #a1a1aa;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body { 
            background-color: var(--bg-body); 
            color: var(--text-main); 
            font-family: 'Inter', sans-serif; 
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* HEADER */
        header {
            background: var(--bg-header);
            border-bottom: 1px solid var(--border);
            padding: 0 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 70px;
        }
        
        .brand { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
        .brand span { color: var(--primary); }
        
        /* PIPELINE STEPPER */
        .stepper { display: flex; align-items: center; gap: 0.5rem; }
        .stepper-item { 
            display: flex; align-items: center; gap: 0.5rem; 
            font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: var(--border);
            position: relative;
        }
        .stepper-item.active { color: var(--primary); }
        .stepper-item.completed { color: var(--success); }
        .stepper-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
        .stepper-line { width: 30px; height: 2px; background: #27272a; }
        .stepper-item.completed + .stepper-line { background: var(--success); opacity: 0.3; }

        /* LAYOUT */
        .container {
            display: grid;
            grid-template-columns: 1fr 400px; /* Main Content + Log Sidebar */
            flex: 1;
            overflow: hidden; 
        }

        .main-panel {
            padding: 2rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            position: relative;
        }

        .log-panel {
            background: #0f0f12;
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }

        .log-header {
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .log-content {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }

        /* COMPONENTS */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
        }
        
        .stat-title { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem; letter-spacing: 0.05em; }
        .stat-value { font-size: 1.5rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

        .timeline-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .timeline-header {
            padding: 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .timeline-content {
            padding: 1.5rem;
            overflow-y: auto;
            flex: 1;
        }

        /* MODAL */
        .modal-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.8); backdrop-filter: blur(5px);
            z-index: 1000; display: flex; align-items: center; justify-content: center;
            opacity: 0; pointer-events: none; transition: opacity 0.3s;
        }
        .modal-overlay.active { opacity: 1; pointer-events: all; }
        
        .modal-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 12px; padding: 2rem; width: 450px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
            transform: translateY(20px); transition: transform 0.3s;
        }
        .modal-overlay.active .modal-card { transform: translateY(0); }
        
        .form-group { margin-bottom: 1.5rem; }
        .form-label { display: block; color: var(--text-muted); margin-bottom: 0.5rem; font-size: 0.875rem; }
        .form-input { 
            width: 100%; background: #09090b; border: 1px solid var(--border);
            padding: 0.75rem; border-radius: 6px; color: white; font-family: inherit;
        }
        .form-input:focus { outline: none; border-color: var(--primary); }
        
        .btn-primary {
            width: 100%; background: var(--primary); border: none; color: white;
            padding: 0.75rem; border-radius: 6px; font-weight: 600; cursor: pointer;
            transition: opacity 0.2s;
        }
        .btn-primary:hover { opacity: 0.9; }

        /* TABS */
        .tabs { display: flex; gap: 1rem; border-bottom: 1px solid var(--border); margin-bottom: 6px; }
        .tab-btn {
            background: none; border: none; color: var(--text-muted); padding: 0.8rem 1.2rem;
            cursor: pointer; font-family: inherit; font-size: 0.875rem; font-weight: 500;
            border-bottom: 2px solid transparent; transition: all 0.2s;
        }
        .tab-btn:hover { color: var(--text-main); }
        .tab-btn.active { color: var(--text-main); border-bottom-color: var(--primary); }

        .tab-content { display: none; padding-top: 1rem; height: 100%; overflow: hidden;}
        .tab-content.active { display: block; }
        
        /* DATA INSPECTOR & STEPS */
        .inspector-view { height: 100%; overflow-y: auto; padding-right: 10px; }
        .json-tree { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; line-height: 1.4; color: #a1a1aa; }
        .json-key { color: var(--primary); }
        .json-string { color: var(--success); }
        .json-number { color: var(--warning); }
        
        .step {
            display: flex; gap: 1.5rem; margin-bottom: 1.5rem;
            opacity: 0.7; transition: all 0.3s ease;
        }
        .step.active { opacity: 1; }
        
        .step-indicator { display: flex; flex-direction: column; align-items: center; }
        .step-dot { width: 12px; height: 12px; border-radius: 50%; background: var(--border); margin-bottom: 0.5rem; position: relative; z-index: 2; }
        .step-line { width: 2px; flex: 1; background: var(--border); min-height: 20px; }
        
        .step.running .step-dot { background: var(--primary); box-shadow: 0 0 10px var(--primary); }
        .step.complete .step-dot { background: var(--success); }
        .step.failed .step-dot { background: var(--danger); }
        .step.complete .step-line { background: #27272a; }

        .step-card {
            background: #27272a; border-radius: 6px; padding: 1rem; flex: 1; border: 1px solid transparent;
        }
        .step.running .step-card { border-color: var(--primary); background: rgba(59,130,246,0.05); }

        .step-title { font-weight: 600; margin-bottom: 0.25rem; display: flex; justify-content: space-between; }
        .step-desc { font-size: 0.875rem; color: var(--text-muted); line-height: 1.4; }
        .step-preview { 
            margin-top: 0.75rem; padding: 0.75rem; background: #18181b; 
            border-radius: 4px; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;
            border-left: 3px solid var(--primary);
        }

        /* PREDICTION OVERLAY */
        .prediction-box {
            margin-top: 2rem;
            background: linear-gradient(145deg, #18181b, #0f0f10);
            border: 1px solid var(--border); border-radius: 12px; padding: 2rem;
            text-align: center; display: none;
            animation: slideUp 0.5s ease;
        }
        
        .pred-verdict { font-size: 3rem; font-weight: 800; margin: 1rem 0; letter-spacing: -0.02em; }
        .verdict-CASE { color: var(--danger); text-shadow: 0 0 30px rgba(239, 68, 68, 0.2); }
        .verdict-CONTROL { color: var(--success); text-shadow: 0 0 30px rgba(16, 185, 129, 0.2); }
        
        /* LOGS */
        .log-entry { 
            margin-bottom: 0.5rem; border-left: 2px solid #333; padding-left: 10px; animation: fadeIn 0.2s;
        }
        .log-ts { color: #52525b; font-size: 0.7rem; margin-right: 8px; }
        .log-type { font-weight: bold; font-size: 0.75rem; display: inline-block; width: 60px; }
        .type-INFO { color: var(--primary); }
        .type-SUCCESS { color: var(--success); }
        .type-WARNING { color: var(--warning); }
        .type-ERROR { color: var(--danger); }

        @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes spin { to { transform: rotate(360deg); } }

        #global-progress { height: 4px; background: var(--primary); width: 0%; transition: width 0.5s ease; position: absolute; top: 0; left: 0; box-shadow: 0 0 10px var(--primary); }
    </style>
</head>
<body>
    <div id="setup-modal" class="modal-overlay active">
        <div class="modal-card">
            <h2 style="margin-bottom: 0.5rem;">Launch Mission</h2>
            <p style="color:var(--text-muted); margin-bottom: 1.5rem;">Enter participant details to begin inference.</p>
            
            <div class="form-group">
                <label class="form-label">Participant ID (Folder Name)</label>
                <input type="text" id="input-pid" class="form-input" placeholder="e.g. participant_ID6015951" value="participant_ID6015951">
            </div>
            
            <div class="form-group">
                <label class="form-label">Target Condition</label>
                <input type="text" id="input-target" class="form-input" placeholder="e.g. neuropsychiatric" value="neuropsychiatric">
            </div>
            
            <button class="btn-primary" onclick="launchPipeline()">INITIATE PIPELINE</button>
        </div>
    </div>

    <div id="global-progress"></div>

    <header>
        <div class="brand">COMPASS <span>DASHBOARD</span></div>
        <div class="stepper" id="stepper"></div>
        <div class="session-info">
             <div class="session-pill" id="status-display" style="color:var(--primary)">Initializing...</div>
        </div>
    </header>

    <div class="container">
        <!-- LEFT: MAIN VISUALIZATION -->
        <div class="main-panel">
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-title">Mission Time</div>
                    <div class="stat-value" id="timer">00:00</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Tokens Used</div>
                    <div class="stat-value" id="token-display">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">System Load</div>
                    <div class="stat-value" id="system-display" style="font-size: 1rem; margin-top:0.4rem;">...</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Current Phase</div>
                    <div class="stat-value" style="font-size: 1rem; margin-top:0.4rem;" id="phase-display">Waiting</div>
                </div>
            </div>

            <div class="timeline-card">
                <div class="timeline-header">
                    <div class="tabs">
                        <button class="tab-btn active" onclick="switchTab('timeline')">Execution Plan</button>
                        <button class="tab-btn" onclick="switchTab('inspector')">Predictor Input Inspector</button>
                    </div>
                </div>
                
                <div id="tab-timeline" class="tab-content active timeline-content">
                    <div id="steps-container">
                        <div id="empty-state" style="text-align:center; color: var(--text-muted); padding: 4rem 2rem; display: flex; flex-direction: column; align-items: center; gap: 1rem;">
                            <div style="width: 40px; height: 40px; border: 3px solid var(--border); border-top-color: var(--primary); border-radius: 50%; animation: spin 1s linear infinite;"></div>
                            <div id="detail-status">Waiting for launch...</div>
                        </div>
                    </div>
                </div>

                <div id="tab-inspector" class="tab-content timeline-content">
                    <div class="inspector-view" id="inspector-content">
                        <div style="text-align:center; color: var(--text-muted); padding: 2rem;">Data will appear here after Execution Phase...</div>
                    </div>
                </div>
            </div>

            <div id="final-verdict-area" class="prediction-box"></div>
        </div>

        <!-- RIGHT: LOGS -->
        <div class="log-panel">
            <div class="log-header">
                <span>SYSTEM LOGS</span>
                <button onclick="toggleScroll()" style="background:none; border:none; color:var(--primary); cursor:pointer; font-size:0.8rem;">Autoscroll: ON</button>
            </div>
            <div class="log-content" id="log-container"></div>
        </div>
    </div>

    <script>
        let latestEventId = 0;
        let startTime = null;
        let autoScroll = true;

        async function launchPipeline() {
            const pid = document.getElementById('input-pid').value;
            const target = document.getElementById('input-target').value;
            
            if (!pid) return alert("Participant ID required");
            
            // Hide modal
            document.getElementById('setup-modal').classList.remove('active');
            
            // Send API request
            try {
                const res = await fetch('/api/launch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({id: pid, target: target})
                });
                const data = await res.json();
                console.log(data.status);
            } catch (e) {
                alert("Failed to launch pipeline: " + e);
            }
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
        }

        function formatJson(json) {
            if (typeof json !== 'string') json = JSON.stringify(json, undefined, 2);
            json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
                var cls = 'json-number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) { cls = 'json-key'; } else { cls = 'json-string'; }
                } else if (/true|false/.test(match)) { cls = 'json-boolean'; } else if (/null/.test(match)) { cls = 'json-null'; }
                return '<span class="' + cls + '">' + match + '</span>';
            });
        }

        function formatTime(seconds) {
            const m = Math.floor(seconds / 60).toString().padStart(2, '0');
            const s = (seconds % 60).toString().padStart(2, '0');
            return `${m}:${s}`;
        }

        async function update() {
            try {
                const res = await fetch(`/api/snapshot?since_id=${latestEventId}`);
                const data = await res.json();
                const state = data.state;
                
                document.getElementById('status-display').textContent = state.status;
                document.getElementById('token-display').textContent = state.total_tokens.toLocaleString();
                if (state.system) document.getElementById('system-display').textContent = `RAM: ${state.system.memory_mb}MB`;
                
                const curStage = state.current_stage > -1 ? state.current_stage : 0;
                document.getElementById('phase-display').textContent = state.stages[curStage] || 'Waiting';
                
                const stepper = document.getElementById('stepper');
                stepper.innerHTML = state.stages.map((st, i) => {
                    let cls = "";
                    if (i < curStage) cls = "completed";
                    else if (i === curStage && state.current_stage > -1) cls = "active";
                    return `<div class="stepper-item ${cls}"><div class="stepper-dot"></div>${st}</div>${i < state.stages.length - 1 ? '<div class="stepper-line"></div>' : ''}`;
                }).join('');
                
                const pct = Math.min((state.progress / state.max_steps) * 100, 100);
                document.getElementById('global-progress').style.width = `${pct}%`;

                if (state.start_time && !startTime) startTime = new Date();
                if (startTime && state.status !== "Pipeline Completed") {
                    const diff = Math.floor((new Date() - startTime) / 1000);
                    document.getElementById('timer').textContent = formatTime(diff);
                }

                if (state.steps.length > 0) {
                    const stepContainer = document.getElementById('steps-container');
                    const stepsHtml = state.steps.map(step => `
                        <div class="step ${step.status} ${step.status === 'running' ? 'active' : ''}">
                            <div class="step-indicator"><div class="step-dot"></div><div class="step-line"></div></div>
                            <div class="step-card">
                                <div class="step-title"><span>${step.tool}</span><span style="font-size:0.75rem; opacity:0.6">${step.duration ? step.duration+'s' : ''}</span></div>
                                <div class="step-desc">${step.desc}</div>
                                ${step.preview ? `<div class="step-preview">${step.preview}</div>` : ''}
                                ${step.error ? `<div class="step-preview" style="border-color:var(--danger); color:var(--danger)">Error: ${step.error}</div>` : ''}
                            </div>
                        </div>
                    `).join('');
                    stepContainer.innerHTML = stepsHtml;
                } else {
                    const details = document.getElementById('detail-status');
                    if (details) details.textContent = state.status;
                }

                if (state.fusion_data && document.getElementById('inspector-content').innerHTML.includes("data will appear")) {
                    document.getElementById('inspector-content').innerHTML = `
                        <div style="margin-bottom:1rem; opacity:0.75">INPUT TO PREDICTOR AGENT:</div>
                        <pre class="json-tree">${formatJson(state.fusion_data)}</pre>
                    `;
                }

                if (state.prediction) {
                    const pArea = document.getElementById('final-verdict-area');
                    pArea.style.display = 'block';
                    pArea.innerHTML = `
                        <div style="text-transform:uppercase; letter-spacing:0.1em; color:var(--text-muted)">Final Analysis Complete</div>
                        <div class="pred-verdict verdict-${state.prediction.result}">${state.prediction.result}</div>
                        <div style="font-size:1.2rem">Confidence: ${(state.prediction.prob * 100).toFixed(1)}%</div>
                    `;
                }

                if (data.events.length > 0) {
                    const logContainer = document.getElementById('log-container');
                    data.events.forEach(e => {
                        if (e.id <= latestEventId) return;
                        latestEventId = e.id;
                        const el = document.createElement('div');
                        el.className = 'log-entry';
                        el.innerHTML = `<span class="log-ts">${e.time}</span><span class="log-type type-${e.type}">${e.type}</span><span>${JSON.stringify(e.data).substring(0, 150)}</span>`;
                        logContainer.appendChild(el);
                    });
                    if (autoScroll) requestAnimationFrame(() => { logContainer.scrollTop = logContainer.scrollHeight; });
                }

            } catch (err) { console.error("Polling error:", err); }
        }

        setInterval(update, 500);
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
    pid = data.get('id')
    target = data.get('target', 'neuropsychiatric')
    
    if _launcher_callback:
        threading.Thread(target=_launcher_callback, args=(pid, target), daemon=True).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "no_callback"}), 500

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

    def set_status(self, message, stage=None):
        data = {"message": message}
        if stage is not None:
            data["stage"] = stage
        _event_store.add_event("STATUS", data)

    def on_pipeline_start(self, participant_id, target, max_iterations=3):
        _event_store.add_event("INIT", {"participant_id": participant_id, "target": target})
        
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

def start_ui_loop(launcher_callback: Callable[[str, str], None]):
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
