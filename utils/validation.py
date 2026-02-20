"""
COMPASS Validation Utilities

Validation functions for inputs, outputs, and data structures.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..config.settings import get_settings
from ..data.models.prediction_task import PredictionTaskSpec, PredictionMode

logger = logging.getLogger("compass.validation")


def validate_participant_files(
    participant_dir: Path
) -> Tuple[bool, List[str], Dict[str, Path]]:
    """
    Validate that all required participant files exist.
    
    Args:
        participant_dir: Path to participant directory
    
    Returns:
        Tuple of (is_valid, error_messages, file_paths)
    """
    settings = get_settings()
    errors = []
    file_paths = {}
    
    if not participant_dir.exists():
        return False, [f"Participant directory not found: {participant_dir}"], {}
    
    if not participant_dir.is_dir():
        return False, [f"Path is not a directory: {participant_dir}"], {}
    
    expected_files = settings.get_participant_files(participant_dir)
    
    for file_key, file_path in expected_files.items():
        if not file_path.exists():
            errors.append(f"Missing required file: {file_path.name}")
        else:
            file_paths[file_key] = file_path
            
            # Validate JSON files
            if file_path.suffix == '.json':
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON in {file_path.name}: {e}")
    
    is_valid = len(errors) == 0
    return is_valid, errors, file_paths


def validate_target_condition(target: str) -> Tuple[bool, str]:
    """
    Validate prediction target label.
    
    Returns:
        Tuple of (is_valid, normalized_target_or_error)
    """
    normalized = str(target or "").strip()
    if not normalized:
        return False, "Invalid target label: value must be non-empty"
    return True, normalized


def validate_prediction_task_spec(task_spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate canonical prediction task specification.
    """
    try:
        PredictionTaskSpec(**(task_spec or {}))
    except Exception as exc:
        return False, [str(exc)]
    return True, []


def validate_data_overview(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate data_overview.json structure.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ["participant_id", "domain_coverage"]
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    if "domain_coverage" in data:
        domain_coverage = data["domain_coverage"]
        if not isinstance(domain_coverage, dict):
            errors.append("domain_coverage must be a dictionary")
        else:
            for domain_name, coverage in domain_coverage.items():
                if not isinstance(coverage, dict):
                    errors.append(f"Coverage for {domain_name} must be a dictionary")
                    continue
                
                required_coverage_keys = ["present_leaves", "total_leaves"]
                for key in required_coverage_keys:
                    if key not in coverage:
                        errors.append(f"Missing {key} in domain {domain_name}")
    
    return len(errors) == 0, errors


def validate_hierarchical_deviation(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate hierarchical_deviation_map.json structure.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if "participant_id" not in data:
        errors.append("Missing participant_id")
    
    if "root" not in data:
        errors.append("Missing root node")
    else:
        root = data["root"]
        if not _validate_deviation_node(root, errors, "root"):
            pass  # Errors already added
    
    return len(errors) == 0, errors


def _validate_deviation_node(
    node: Dict[str, Any],
    errors: List[str],
    path: str
) -> bool:
    """Recursively validate a deviation node."""
    required = ["node_id", "node_name"]
    
    for key in required:
        if key not in node:
            errors.append(f"Missing {key} in node at {path}")
    
    if "z_score" in node and node["z_score"] is not None:
        if not isinstance(node["z_score"], (int, float)):
            errors.append(f"z_score must be numeric at {path}")
    
    if "children" in node:
        if not isinstance(node["children"], list):
            errors.append(f"children must be a list at {path}")
        else:
            for i, child in enumerate(node["children"]):
                _validate_deviation_node(child, errors, f"{path}.children[{i}]")
    
    return True


def validate_prediction(prediction: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate prediction result structure.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Generalized schema path.
    if "root_prediction" in prediction or "prediction_task_spec" in prediction:
        root = prediction.get("root_prediction") or {}
        if not isinstance(root, dict):
            errors.append("root_prediction must be an object")
            return len(errors) == 0, errors

        mode = str(root.get("mode") or "").strip()
        if not mode:
            errors.append("root_prediction.mode is required")
        else:
            valid_modes = {m.value for m in PredictionMode}
            if mode not in valid_modes:
                errors.append(f"Invalid root_prediction.mode: {mode}")

        if mode in {PredictionMode.BINARY_CLASSIFICATION.value, PredictionMode.MULTICLASS_CLASSIFICATION.value}:
            cls_payload = root.get("classification")
            if not isinstance(cls_payload, dict):
                errors.append("classification output required for classification mode")
            else:
                probs = cls_payload.get("probabilities") or {}
                if probs:
                    if not isinstance(probs, dict):
                        errors.append("classification.probabilities must be an object")
                    else:
                        total = 0.0
                        for label, raw in probs.items():
                            try:
                                val = float(raw)
                            except Exception:
                                errors.append(f"classification probability for '{label}' is non-numeric")
                                continue
                            if val < 0.0 or val > 1.0:
                                errors.append(f"classification probability for '{label}' must be in [0,1], got {val}")
                            total += val
                        if abs(total - 1.0) > 0.05:
                            errors.append(f"classification probabilities should sum ~1.0, got {total:.4f}")
        elif mode in {PredictionMode.UNIVARIATE_REGRESSION.value, PredictionMode.MULTIVARIATE_REGRESSION.value}:
            reg_payload = root.get("regression")
            if not isinstance(reg_payload, dict):
                errors.append("regression output required for regression mode")
            else:
                values = reg_payload.get("values")
                if not isinstance(values, dict) or not values:
                    errors.append("regression.values must be a non-empty object")
                else:
                    for name, raw in values.items():
                        try:
                            val = float(raw)
                        except Exception:
                            errors.append(f"regression value for '{name}' is non-numeric")
                            continue
                        if not math.isfinite(val):
                            errors.append(f"regression value for '{name}' must be finite")

        return len(errors) == 0, errors

    # Legacy binary schema path.
    required_keys = ["binary_classification", "probability_score", "key_findings", "reasoning_chain"]
    for key in required_keys:
        if key not in prediction:
            errors.append(f"Missing required key: {key}")

    if "binary_classification" in prediction:
        valid_classes = ["CASE", "CONTROL"]
        if prediction["binary_classification"] not in valid_classes:
            errors.append(
                f"Invalid classification: {prediction['binary_classification']}. "
                f"Must be one of {valid_classes}"
            )

    if "probability_score" in prediction:
        prob = prediction["probability_score"]
        if not isinstance(prob, (int, float)):
            errors.append("probability_score must be numeric")
        elif prob < 0 or prob > 1:
            errors.append(f"probability_score must be between 0 and 1, got {prob}")

    if "binary_classification" in prediction and "probability_score" in prediction:
        classif = prediction["binary_classification"]
        prob = prediction["probability_score"]
        if classif == "CASE" and prob < 0.5:
            errors.append(f"Inconsistent: CASE classification with probability {prob} < 0.5")
        if classif == "CONTROL" and prob >= 0.5:
            errors.append(f"Inconsistent: CONTROL classification with probability {prob} >= 0.5")

    return len(errors) == 0, errors


def validate_execution_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate execution plan structure.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    required_keys = ["plan_id", "steps", "target_condition"]
    for key in required_keys:
        if key not in plan:
            errors.append(f"Missing required key: {key}")
    
    if "steps" in plan:
        if not isinstance(plan["steps"], list):
            errors.append("steps must be a list")
        elif len(plan["steps"]) == 0:
            errors.append("Execution plan has no steps")
        else:
            step_ids = set()
            for i, step in enumerate(plan["steps"]):
                if not isinstance(step, dict):
                    errors.append(f"Step {i} must be a dictionary")
                    continue
                
                if "step_id" not in step:
                    errors.append(f"Step {i} missing step_id")
                else:
                    if step["step_id"] in step_ids:
                        errors.append(f"Duplicate step_id: {step['step_id']}")
                    step_ids.add(step["step_id"])
                
                if "tool_name" not in step:
                    errors.append(f"Step {i} missing tool_name")
    
    return len(errors) == 0, errors


def check_token_budget(
    estimated_tokens: int,
    budget: int
) -> Tuple[bool, str]:
    """
    Check if estimated tokens fit within budget.
    
    Returns:
        Tuple of (within_budget, message)
    """
    if estimated_tokens <= budget:
        return True, f"Token estimate ({estimated_tokens}) within budget ({budget})"
    
    overage = estimated_tokens - budget
    overage_pct = (overage / budget) * 100
    return False, (
        f"Token estimate ({estimated_tokens}) exceeds budget ({budget}) "
        f"by {overage} tokens ({overage_pct:.1f}%)"
    )
