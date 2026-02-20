"""
COMPASS Prediction Task Models

Canonical task specification for binary/multiclass/regression/hierarchical
deep phenotype prediction.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, root_validator, validator


class PredictionMode(str, Enum):
    """Supported prediction objective modes."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    UNIVARIATE_REGRESSION = "univariate_regression"
    MULTIVARIATE_REGRESSION = "multivariate_regression"


class PredictionTaskNode(BaseModel):
    """A recursive task node in a phenotype prediction tree."""

    node_id: str
    display_name: str
    mode: PredictionMode
    class_labels: List[str] = Field(default_factory=list)
    regression_outputs: List[str] = Field(default_factory=list)
    unit_by_output: Dict[str, str] = Field(default_factory=dict)
    required: bool = True
    children: List["PredictionTaskNode"] = Field(default_factory=list)

    @validator("node_id", "display_name")
    def _validate_non_empty(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value must be non-empty")
        return text

    @validator("class_labels", pre=True, always=True)
    def _normalize_class_labels(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        labels = [str(x).strip() for x in list(value or []) if str(x).strip()]
        dedup: List[str] = []
        for label in labels:
            if label not in dedup:
                dedup.append(label)
        return dedup

    @validator("regression_outputs", pre=True, always=True)
    def _normalize_regression_outputs(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        outputs = [str(x).strip() for x in list(value or []) if str(x).strip()]
        dedup: List[str] = []
        for output in outputs:
            if output not in dedup:
                dedup.append(output)
        return dedup

    @validator("children", pre=True, always=True)
    def _normalize_children(cls, value: Any) -> List["PredictionTaskNode"]:
        return list(value or [])

    @root_validator(skip_on_failure=True)
    def _validate_mode_contract(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        mode: PredictionMode = values.get("mode")
        class_labels: List[str] = list(values.get("class_labels") or [])
        regression_outputs: List[str] = list(values.get("regression_outputs") or [])

        if mode == PredictionMode.BINARY_CLASSIFICATION:
            if len(class_labels) != 2:
                raise ValueError("binary_classification requires exactly 2 class_labels")
            if regression_outputs:
                raise ValueError("binary_classification cannot define regression_outputs")
        elif mode == PredictionMode.MULTICLASS_CLASSIFICATION:
            if len(class_labels) < 3:
                raise ValueError("multiclass_classification requires >=3 class_labels")
            if regression_outputs:
                raise ValueError("multiclass_classification cannot define regression_outputs")
        elif mode == PredictionMode.UNIVARIATE_REGRESSION:
            if len(regression_outputs) != 1:
                raise ValueError("univariate_regression requires exactly 1 regression_output")
            if class_labels:
                raise ValueError("univariate_regression cannot define class_labels")
        elif mode == PredictionMode.MULTIVARIATE_REGRESSION:
            if len(regression_outputs) < 2:
                raise ValueError("multivariate_regression requires >=2 regression_outputs")
            if class_labels:
                raise ValueError("multivariate_regression cannot define class_labels")

        child_ids: List[str] = []
        for child in list(values.get("children") or []):
            cid = str(getattr(child, "node_id", "")).strip()
            if cid in child_ids:
                raise ValueError(f"duplicate child node_id under node '{values.get('node_id')}': {cid}")
            child_ids.append(cid)
        return values

    def walk(self) -> List["PredictionTaskNode"]:
        nodes: List["PredictionTaskNode"] = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return nodes


PredictionTaskNode.model_rebuild()


class PredictionTaskSpec(BaseModel):
    """
    Canonical task specification.

    Designed as hierarchical-first and supports mixed mode children.
    """

    schema_version: str = "1.0"
    task_id: Optional[str] = None
    root: PredictionTaskNode

    def node_index(self) -> Dict[str, PredictionTaskNode]:
        return {node.node_id: node for node in self.root.walk()}

    def is_pure_binary_root(self) -> bool:
        return self.root.mode == PredictionMode.BINARY_CLASSIFICATION and len(self.root.children) == 0

    def to_brief_dict(self) -> Dict[str, Any]:
        node_count = len(self.root.walk())
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "root_node_id": self.root.node_id,
            "root_mode": self.root.mode.value,
            "node_count": node_count,
        }

    def legacy_target_control(self) -> Tuple[str, str]:
        """
        Derive backward-compatible target/control labels for legacy components.
        """
        root_name = str(self.root.display_name or self.root.node_id)
        if self.root.mode == PredictionMode.BINARY_CLASSIFICATION and len(self.root.class_labels) == 2:
            case_label = str(self.root.class_labels[0])
            control_label = str(self.root.class_labels[1])
            return case_label or root_name, control_label or "CONTROL"
        return root_name, ""


def parse_csv_list(raw: Optional[str]) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def build_binary_task_spec(
    *,
    target_label: str,
    control_label: str = "CONTROL",
    class_labels: Optional[List[str]] = None,
) -> PredictionTaskSpec:
    labels = [str(x).strip() for x in (class_labels or []) if str(x).strip()]
    if len(labels) != 2:
        labels = [str(target_label or "CASE").strip() or "CASE", str(control_label or "CONTROL").strip() or "CONTROL"]
    return PredictionTaskSpec(
        root=PredictionTaskNode(
            node_id="root",
            display_name=str(target_label or "Target Phenotype").strip() or "Target Phenotype",
            mode=PredictionMode.BINARY_CLASSIFICATION,
            class_labels=labels,
            children=[],
        )
    )


def build_task_spec_from_flat_args(
    *,
    prediction_type: str,
    target_label: str,
    control_label: str = "CONTROL",
    class_labels: Optional[List[str]] = None,
    regression_outputs: Optional[List[str]] = None,
) -> PredictionTaskSpec:
    pred_type = str(prediction_type or "binary").strip().lower()
    classes = [str(x).strip() for x in (class_labels or []) if str(x).strip()]
    outputs = [str(x).strip() for x in (regression_outputs or []) if str(x).strip()]

    if pred_type == "binary":
        return build_binary_task_spec(
            target_label=target_label,
            control_label=control_label,
            class_labels=classes if classes else None,
        )
    if pred_type == "multiclass":
        if len(classes) < 3:
            raise ValueError("--class_labels requires at least 3 labels for multiclass")
        return PredictionTaskSpec(
            root=PredictionTaskNode(
                node_id="root",
                display_name=str(target_label or "Target Phenotype").strip() or "Target Phenotype",
                mode=PredictionMode.MULTICLASS_CLASSIFICATION,
                class_labels=classes,
            )
        )
    if pred_type == "regression_univariate":
        if any("," in str(name or "") for name in outputs):
            raise ValueError("Univariate regression output must be a single name without commas")
        if len(outputs) != 1:
            raise ValueError("Univariate regression requires exactly 1 output (--regression_output or one --regression_outputs value)")
        return PredictionTaskSpec(
            root=PredictionTaskNode(
                node_id="root",
                display_name=str(target_label or "Target Phenotype").strip() or "Target Phenotype",
                mode=PredictionMode.UNIVARIATE_REGRESSION,
                regression_outputs=outputs,
            )
        )
    if pred_type == "regression_multivariate":
        if len(outputs) < 2:
            raise ValueError("--regression_outputs requires >=2 outputs for multivariate regression")
        return PredictionTaskSpec(
            root=PredictionTaskNode(
                node_id="root",
                display_name=str(target_label or "Target Phenotype").strip() or "Target Phenotype",
                mode=PredictionMode.MULTIVARIATE_REGRESSION,
                regression_outputs=outputs,
            )
        )
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")
