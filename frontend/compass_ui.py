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
import ssl
from flask import Flask, render_template, jsonify, request, send_file, make_response
from flask.json.provider import DefaultJSONProvider
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

try:
    import certifi
except Exception:  # pragma: no cover - optional dependency
    certifi = None

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
            "iteration": 1,
            "deep_report_status": "idle",
            "deep_report_available": False,
            "deep_report_error": None,
            "deep_report_last_generated_at": None,
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
                self.state["deep_report_status"] = "idle"
                self.state["deep_report_available"] = False
                self.state["deep_report_error"] = None
                self.state["deep_report_last_generated_at"] = None
                
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
                else:
                    # Allow repeated STEP_START for the same step id to refresh live descriptions
                    # (used for dynamic fusion/chunking progress without creating extra timeline rows).
                    existing["tool"] = data.get("tool", existing.get("tool"))
                    existing["desc"] = data.get("desc", existing.get("desc"))
                    existing["status"] = "running"
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
            elif event_type == "DEEP_REPORT":
                status = str(data.get("status") or "").lower().strip()
                if status:
                    self.state["deep_report_status"] = status
                if "available" in data:
                    self.state["deep_report_available"] = bool(data.get("available"))
                if "error" in data:
                    self.state["deep_report_error"] = data.get("error")
                if status == "completed":
                    self.state["deep_report_last_generated_at"] = datetime.now().isoformat()

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
_deep_report_callback: Optional[Callable[[Dict], Dict[str, Any]]] = None

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
        "public_model": data.get('public_model'),
        "public_max_context_tokens": data.get('public_max_context_tokens'),
        "embedding_model": data.get('embedding_model'),
        "local_embedding_model": data.get('local_embedding_model'),
        "model": data.get('model'),
        "max_tokens": data.get('max_tokens'),
        "role_models": data.get('role_models'),
        "role_max_tokens": data.get('role_max_tokens'),
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


def _open_url_json(req: Request, timeout: int = 15) -> Dict[str, Any]:
    contexts = []
    if certifi is not None:
        try:
            contexts.append(ssl.create_default_context(cafile=certifi.where()))
        except Exception:
            pass
    contexts.append(ssl.create_default_context())
    last_error: Optional[Exception] = None
    for ctx in contexts:
        try:
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            reason = getattr(e, "reason", None)
            if isinstance(reason, ssl.SSLCertVerificationError):
                last_error = e
                continue
            raise
    if str(os.getenv("COMPASS_ALLOW_INSECURE_TLS", "0")).strip().lower() in {"1", "true", "yes"}:
        insecure_ctx = ssl._create_unverified_context()
        with urlopen(req, timeout=timeout, context=insecure_ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    if last_error is not None:
        raise last_error
    raise RuntimeError("tls_context_unavailable")


def _extract_context_length_candidates_from_mapping(data: Dict[str, Any]) -> List[int]:
    if not isinstance(data, dict):
        return []
    candidates = [
        data.get("max_position_embeddings"),
        data.get("n_positions"),
        data.get("max_seq_len"),
        data.get("max_sequence_length"),
        data.get("model_max_length"),
        data.get("max_seq_length"),
        data.get("seq_length"),
        data.get("context_length"),
        data.get("n_ctx"),
        data.get("max_context_length"),
        data.get("max_length"),
        data.get("max_seq_length_for_truncation"),
    ]
    values: List[int] = []
    for value in candidates:
        try:
            if value is None:
                continue
            parsed = int(value)
            if parsed > 0:
                values.append(parsed)
        except Exception:
            continue
    return values


def _choose_context_length(candidates: List[int]) -> Optional[int]:
    # Use a conservative, practical window: ignore tiny/sentinel values.
    sane = [v for v in candidates if 64 <= v <= 1_000_000]
    if sane:
        return min(sane)
    fallback = [v for v in candidates if 2 <= v <= 1_000_000]
    if fallback:
        return min(fallback)
    return None


def _fetch_hf_raw_json(model_id: str, filename: str) -> Optional[Dict[str, Any]]:
    url = f"https://huggingface.co/{model_id}/raw/main/{filename}"
    try:
        req = Request(url=url, method="GET")
        payload = _open_url_json(req, timeout=10)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _looks_like_embedding_id(model_id: str) -> bool:
    value = (model_id or "").lower()
    if not value:
        return False
    return any(token in value for token in ("embedding", "embed", "text-embedding", "/bge", "/e5", "/gte"))


def _is_embedding_model(
    model_id: str = "",
    pipeline_tag: str = "",
    tags: Optional[List[str]] = None,
    modality: str = "",
    supported_parameters: Optional[List[str]] = None,
) -> bool:
    pipeline = (pipeline_tag or "").lower()
    if pipeline in {"feature-extraction", "sentence-similarity"}:
        return True

    modality_l = (modality or "").lower()
    if "embedding" in modality_l or "vector" in modality_l:
        return True

    tags_l = [(tag or "").lower() for tag in (tags or [])]
    if any(tag in {"feature-extraction", "sentence-similarity", "embeddings", "embedding"} for tag in tags_l):
        return True

    params_l = [(param or "").lower() for param in (supported_parameters or [])]
    if any("embed" in param for param in params_l):
        return True

    return _looks_like_embedding_id(model_id)


def _is_hf_embedding_model(model_id: str = "", pipeline_tag: str = "", tags: Optional[List[str]] = None) -> bool:
    pipeline = (pipeline_tag or "").lower()
    tags_l = [(tag or "").lower() for tag in (tags or [])]

    if pipeline:
        return pipeline in {"feature-extraction", "sentence-similarity"}

    if any(tag in {"feature-extraction", "sentence-similarity", "embeddings", "embedding"} for tag in tags_l):
        return True

    name = (model_id or "").lower()
    if _looks_like_embedding_id(name):
        if any(token in name for token in ("instruct", "chat", "llm")):
            return False
        return True

    return False


@app.route('/api/openrouter/models')
def openrouter_models():
    try:
        try:
            from ..config.settings import get_settings
        except Exception:  # pragma: no cover - top-level fallback
            from config.settings import get_settings
        settings = get_settings()
    except Exception as e:
        return jsonify({"error": f"settings_unavailable: {e}"}), 500

    api_key = (settings.openrouter_api_key or "").strip()
    if not api_key:
        return jsonify({"error": "OPENROUTER_API_KEY is not configured"}), 400

    url = f"{settings.openrouter_base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name

    try:
        req = Request(url=url, headers=headers, method="GET")
        payload = _open_url_json(req, timeout=15)
        rows = []
        for item in (payload.get("data") or []):
            model_id = item.get("id")
            if not model_id:
                continue
            pricing = item.get("pricing") or {}
            architecture = item.get("architecture") or {}
            modality = architecture.get("modality") or ""
            supported_parameters = item.get("supported_parameters") or []
            prompt_price = pricing.get("prompt")
            completion_price = pricing.get("completion")
            rows.append(
                {
                    "id": model_id,
                    "context_length": int(item.get("context_length") or 0) or None,
                    "prompt_price": prompt_price,
                    "completion_price": completion_price,
                    "modality": modality,
                    "is_embedding": _is_embedding_model(
                        model_id=model_id,
                        modality=modality,
                        supported_parameters=supported_parameters,
                    ),
                }
            )
        rows.sort(key=lambda x: x["id"])
        return jsonify({"models": rows})
    except HTTPError as e:
        return jsonify({"error": f"openrouter_http_error_{e.code}"}), 502
    except URLError as e:
        reason = getattr(e, "reason", e)
        return jsonify({"error": f"openrouter_network_error: {reason}"}), 502
    except Exception as e:
        return jsonify({"error": f"openrouter_fetch_failed: {e}"}), 500


@app.route('/api/openrouter/embedding-models')
def openrouter_embedding_models():
    try:
        try:
            from ..config.settings import get_settings
        except Exception:  # pragma: no cover - top-level fallback
            from config.settings import get_settings
        settings = get_settings()
    except Exception as e:
        return jsonify({"error": f"settings_unavailable: {e}"}), 500

    api_key = (settings.openrouter_api_key or "").strip()
    if not api_key:
        return jsonify({"error": "OPENROUTER_API_KEY is not configured"}), 400

    base_url = settings.openrouter_base_url.rstrip("/")
    url = f"{base_url}/embeddings/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name

    try:
        req = Request(url=url, headers=headers, method="GET")
        payload = _open_url_json(req, timeout=15)
        raw_models = payload.get("data") if isinstance(payload, dict) else payload
        rows = []
        for item in (raw_models or []):
            model_id = item.get("id")
            if not model_id:
                continue
            pricing = item.get("pricing") or {}
            prompt_price = pricing.get("prompt") or item.get("prompt_price") or item.get("input_price")
            completion_price = pricing.get("completion") or item.get("completion_price") or item.get("output_price")
            context_length = item.get("context_length") or item.get("max_context_length")
            rows.append(
                {
                    "id": model_id,
                    "context_length": int(context_length or 0) or None,
                    "prompt_price": prompt_price,
                    "completion_price": completion_price,
                    "modality": "text->vector",
                    "is_embedding": True,
                }
            )
        rows.sort(key=lambda x: x["id"])
        return jsonify({"models": rows})
    except HTTPError as e:
        return jsonify({"error": f"openrouter_http_error_{e.code}"}), 502
    except URLError as e:
        reason = getattr(e, "reason", e)
        return jsonify({"error": f"openrouter_network_error: {reason}"}), 502
    except Exception as e:
        return jsonify({"error": f"openrouter_fetch_failed: {e}"}), 500


@app.route('/api/hf/models')
def hf_models():
    query = (request.args.get("q") or "").strip()
    task = (request.args.get("task") or "").strip().lower()
    limit = 200
    base_url = "https://huggingface.co/api/models"
    params = f"?limit={limit}&sort=downloads&direction=-1"
    if task == "embedding":
        params += "&pipeline_tag=feature-extraction"
    if query:
        params += f"&search={query}"
    url = f"{base_url}{params}"
    try:
        req = Request(url=url, method="GET")
        payload = _open_url_json(req, timeout=15)
        rows = []
        for item in payload or []:
            model_id = item.get("modelId") or item.get("id")
            if not model_id:
                continue
            pipeline_tag = item.get("pipeline_tag") or ""
            tags = item.get("tags") or []
            is_embedding = _is_hf_embedding_model(
                model_id=model_id,
                pipeline_tag=pipeline_tag,
                tags=tags,
            )
            if task == "embedding" and not is_embedding:
                continue
            if task == "llm" and is_embedding:
                continue
            rows.append(
                {
                    "id": model_id,
                    "downloads": item.get("downloads"),
                    "likes": item.get("likes"),
                    "pipeline_tag": pipeline_tag,
                    "is_embedding": is_embedding,
                }
            )
        return jsonify({"models": rows})
    except HTTPError as e:
        return jsonify({"error": f"hf_http_error_{e.code}"}), 502
    except URLError as e:
        reason = getattr(e, "reason", e)
        return jsonify({"error": f"hf_network_error: {reason}"}), 502
    except Exception as e:
        return jsonify({"error": f"hf_fetch_failed: {e}"}), 500


@app.route('/api/hf/model/<path:model_id>')
def hf_model_detail(model_id: str):
    model_id = (model_id or "").strip()
    if not model_id:
        return jsonify({"error": "missing_model_id"}), 400
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        req = Request(url=url, method="GET")
        payload = _open_url_json(req, timeout=15)
        config = payload.get("config") or {}
        transformers_info = payload.get("transformersInfo") or {}
        card_data = payload.get("cardData") or {}
        source_candidates: Dict[str, List[int]] = {
            "model_meta": [],
            "tokenizer_config": [],
            "config_json": [],
            "sentence_bert_config": [],
        }
        source_candidates["model_meta"].extend(_extract_context_length_candidates_from_mapping(config))
        source_candidates["model_meta"].extend(_extract_context_length_candidates_from_mapping(transformers_info))
        source_candidates["model_meta"].extend(_extract_context_length_candidates_from_mapping(card_data))

        tokenizer_cfg = _fetch_hf_raw_json(model_id, "tokenizer_config.json") or {}
        source_candidates["tokenizer_config"].extend(
            _extract_context_length_candidates_from_mapping(tokenizer_cfg)
        )
        raw_cfg = _fetch_hf_raw_json(model_id, "config.json") or {}
        source_candidates["config_json"].extend(
            _extract_context_length_candidates_from_mapping(raw_cfg)
        )
        sentence_cfg = _fetch_hf_raw_json(model_id, "sentence_bert_config.json") or {}
        source_candidates["sentence_bert_config"].extend(
            _extract_context_length_candidates_from_mapping(sentence_cfg)
        )

        all_candidates: List[int] = []
        for values in source_candidates.values():
            all_candidates.extend(values)
        context_len = _choose_context_length(all_candidates)

        context_source = "unknown"
        if context_len is not None:
            for source_name in ("tokenizer_config", "sentence_bert_config", "config_json", "model_meta"):
                if context_len in source_candidates.get(source_name, []):
                    context_source = source_name
                    break

        return jsonify(
            {
                "id": payload.get("modelId") or payload.get("id") or model_id,
                "context_length": context_len,
                "context_source": context_source,
                "downloads": payload.get("downloads"),
                "likes": payload.get("likes"),
                "library_name": payload.get("library_name"),
                "pipeline_tag": payload.get("pipeline_tag"),
                "is_embedding": _is_hf_embedding_model(
                    model_id=payload.get("modelId") or payload.get("id") or model_id,
                    pipeline_tag=payload.get("pipeline_tag"),
                    tags=payload.get("tags") or [],
                ),
            }
        )
    except HTTPError as e:
        return jsonify({"error": f"hf_http_error_{e.code}"}), 502
    except URLError as e:
        reason = getattr(e, "reason", e)
        return jsonify({"error": f"hf_network_error: {reason}"}), 502
    except Exception as e:
        return jsonify({"error": f"hf_fetch_failed: {e}"}), 500


@app.route('/api/deep_phenotype/generate', methods=['POST'])
def generate_deep_phenotype():
    data = request.json or {}
    if not _deep_report_callback:
        return jsonify({"status": "no_callback", "error": "Deep report callback not registered"}), 500

    payload = {
        "focus_modalities": data.get("focus_modalities", ""),
        "general_instruction": data.get("general_instruction", ""),
    }
    _event_store.add_event("DEEP_REPORT", {"status": "queued", "available": False, "error": None})

    def _worker():
        _event_store.add_event("DEEP_REPORT", {"status": "running", "available": False, "error": None})
        try:
            result = _deep_report_callback(payload)
            _event_store.add_event(
                "DEEP_REPORT",
                {
                    "status": "completed",
                    "available": True,
                    "error": None,
                    "path": (result or {}).get("path"),
                },
            )
        except Exception as e:
            _event_store.add_event(
                "DEEP_REPORT",
                {"status": "failed", "available": False, "error": str(e)},
            )

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"status": "started"})


@app.route('/api/deep_phenotype/status')
def deep_phenotype_status():
    state = _event_store.state
    return jsonify(
        {
            "status": state.get("deep_report_status", "idle"),
            "available": bool(state.get("deep_report_available")),
            "error": state.get("deep_report_error"),
            "last_generated_at": state.get("deep_report_last_generated_at"),
        }
    )

@app.route('/api/outputs')
def list_outputs():
    # List all files in results/participant_{id} if we can know participant ID
    pid = _event_store.state.get("participant_id")
    if not pid or pid == "Unknown":
        return jsonify([])

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(os.path.dirname(base), "results")
    p_dir = _event_store.state.get("participant_dir")
    pseudo_inputs = os.path.join(base, "data", "pseudo_data", "inputs")
    pseudo_outputs = os.path.join(base, "data", "pseudo_data", "outputs")
    output_root = results_dir
    try:
        if p_dir and os.path.commonpath([os.path.abspath(p_dir), pseudo_inputs]) == pseudo_inputs:
            output_root = pseudo_outputs
    except Exception:
        output_root = results_dir
    
    # Try finding folder
    target_dir = None
    if not os.path.exists(output_root):
         # Try creating it or looking at old path relative to project
         pass

    # Safe fallback if results_dir check fails
    possible_names = [f"participant_{pid}", f"participant_ID{pid}", pid, f"ID{pid}"]
    for name in possible_names:
        path = os.path.join(output_root, name)
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
    p_dir = _event_store.state.get("participant_dir")
    pseudo_inputs = os.path.join(base, "data", "pseudo_data", "inputs")
    pseudo_outputs = os.path.join(base, "data", "pseudo_data", "outputs")
    output_root = results_dir
    try:
        if p_dir and os.path.commonpath([os.path.abspath(p_dir), pseudo_inputs]) == pseudo_inputs:
            output_root = pseudo_outputs
    except Exception:
        output_root = results_dir
    
    # Same logic to find dir
    target_dir = None
    possible_names = [f"participant_{pid}", f"participant_ID{pid}", pid, f"ID{pid}"]
    for name in possible_names:
        path = os.path.join(output_root, name)
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

def start_ui_loop(
    launcher_callback: Callable[[Dict], None],
    deep_report_callback: Optional[Callable[[Dict], Dict[str, Any]]] = None
):
    """Start server and wait for user to launch via UI."""
    global _launcher_callback, _deep_report_callback
    _launcher_callback = launcher_callback
    _deep_report_callback = deep_report_callback
    
    ui = get_ui()
    ui.start_server()
    
    print("[*] Dashboard ready. Waiting for user input via Web UI...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
