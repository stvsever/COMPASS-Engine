"""
COMPASS Patient Report Generator

Generates per-patient final reports with predictions and execution logs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ...config.settings import get_settings
from ...data.models.prediction_result import PredictionResult, CriticEvaluation


class PatientReportGenerator:
    """
    Generates comprehensive patient reports.
    
    Includes:
    - Prediction details
    - Key findings
    - Execution log summary
    - Decision trace
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def generate(
        self,
        participant_id: str,
        prediction: PredictionResult,
        evaluation: CriticEvaluation,
        execution_summary: Dict[str, Any],
        decision_trace: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete patient report.
        
        Returns:
            Report as dictionary
        """
        report = {
            "report_id": f"RPT_{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "participant_id": participant_id,
            "generated_at": datetime.now().isoformat(),
            
            "prediction": {
                "prediction_type": self._prediction_type_from_prediction(prediction),
                "classification": self._classification_label_from_prediction(prediction),
                "primary_output": self._primary_output_from_prediction(prediction),
                "probability": prediction.probability_score,
                "confidence": prediction.confidence_level.value,
                "target_condition": prediction.target_condition,
                "control_condition": prediction.control_condition,
                "prediction_task_spec": (
                    prediction.prediction_task_spec.model_dump()
                    if prediction.prediction_task_spec is not None and hasattr(prediction.prediction_task_spec, "model_dump")
                    else (prediction.prediction_task_spec.dict() if prediction.prediction_task_spec is not None else None)
                ),
                "root_prediction": (
                    prediction.root_prediction.model_dump()
                    if prediction.root_prediction is not None and hasattr(prediction.root_prediction, "model_dump")
                    else (prediction.root_prediction.dict() if prediction.root_prediction is not None else None)
                ),
            },
            
            "evaluation": {
                "verdict": evaluation.verdict.value,
                "confidence_in_verdict": evaluation.confidence_in_verdict,
                "checklist_passed": evaluation.checklist.pass_count,
                "checklist_total": evaluation.checklist.total_count
            },
            
            "key_findings": [
                {
                    "domain": f.domain,
                    "finding": f.finding,
                    "direction": f.direction
                }
                for f in prediction.key_findings[:10]
            ],
            
            "reasoning": prediction.reasoning_chain,
            "clinical_summary": prediction.clinical_summary,
            
            "execution": {
                "iterations": execution_summary.get("iterations", 1),
                "selected_iteration": execution_summary.get("selected_iteration", 1),
                "selection_reason": execution_summary.get("selection_reason", ""),
                "coverage_summary": execution_summary.get("coverage_summary", {}),
                "dataflow_summary": execution_summary.get("dataflow_summary", {}),
                "tokens_used": execution_summary.get("tokens_used", 0),
                "domains_processed": execution_summary.get("domains_processed", []),
                "detailed_logs": execution_summary.get("detailed_logs", [])
            },
            
            "decision_trace": decision_trace or []
        }
        
        return report
    
    def save(
        self,
        report: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> Path:
        """Save report to JSON file."""
        if output_dir is None:
            output_dir = self.settings.paths.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        participant_id = report.get("participant_id", "unknown")
        filename = f"report_{participant_id}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[Report] Saved to: {output_path}")
        return output_path
    
    def to_markdown(self, report: Dict[str, Any]) -> str:
        """Generate markdown version of report."""
        pred = report.get("prediction", {})
        eval_data = report.get("evaluation", {})
        
        classification = pred.get('classification', 'N/A')
        prediction_type = pred.get('prediction_type', 'unknown')
        target = pred.get('target_condition', 'N/A')
        control = pred.get('control_condition', 'N/A')
        primary_output = pred.get("primary_output", "N/A")
        is_classification_mode = str(prediction_type).endswith("classification")
        
        display_prediction = primary_output
        if is_classification_mode:
            display_prediction = classification

        lines = [
            f"# Patient Report: {report.get('participant_id', 'Unknown')}",
            f"\n**Generated**: {report.get('generated_at', '')}",
            f"\n## Prediction",
            f"- **Prediction Type**: {prediction_type}",
            f"- **Primary Output**: {display_prediction}",
            (
                f"- **Probability / Root Confidence**: {pred.get('probability', 0):.1%}"
                if isinstance(pred.get("probability"), (int, float))
                else "- **Probability / Root Confidence**: N/A"
            ),
            f"- **Confidence**: {pred.get('confidence', 'N/A')}",
            f"- **Target Label Context**: {target}",
        ]
        if is_classification_mode and str(control or "").strip():
            lines.append(f"- **Comparator Label Context**: {control}")
        lines.extend(
            [
                f"\n## Evaluation",
                f"- **Verdict**: {eval_data.get('verdict', 'N/A')}",
                f"- **Checklist**: {eval_data.get('checklist_passed', 0)}/{eval_data.get('checklist_total', 7)} passed",
                f"\n## Key Findings",
            ]
        )
        
        for i, finding in enumerate(report.get("key_findings", [])[:5], 1):
            lines.append(f"{i}. **[{finding.get('domain', '')}]** {finding.get('finding', '')}")

        lines.extend([
            f"\n## Clinical Summary",
            report.get("clinical_summary", "No summary available"),
            
            f"\n## Reasoning Chain"
        ])
        
        for i, step in enumerate(report.get("reasoning", [])[:5], 1):
            lines.append(f"{i}. {step}")
        
        exec_data = report.get("execution", {})
        lines.extend([
            f"\n## Execution Details",
            f"- **Iterations**: {exec_data.get('iterations', 1)}",
            f"- **Selected Iteration**: {exec_data.get('selected_iteration', 1)}",
            f"- **Selection Reason**: {exec_data.get('selection_reason', 'N/A')}",
            f"- **Tokens Used**: {exec_data.get('tokens_used', 0):,}",
            f"- **Domains Processed**: {', '.join(exec_data.get('domains_processed', []))}"
        ])
        
        return "\n".join(lines)
    
    def save_markdown(
        self,
        report: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> Path:
        """Save markdown version of report."""
        if output_dir is None:
            output_dir = self.settings.paths.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        participant_id = report.get("participant_id", "unknown")
        filename = f"report_{participant_id}.md"
        output_path = output_dir / filename
        
        markdown = self.to_markdown(report)
        with open(output_path, 'w') as f:
            f.write(markdown)
        
        print(f"[Report] Markdown saved to: {output_path}")
        return output_path

    def _primary_output_from_prediction(self, prediction: PredictionResult) -> str:
        root = prediction.root_prediction
        if root is not None:
            if root.classification is not None:
                return str(root.classification.predicted_label or root.node_id)
            if root.regression is not None and root.regression.values:
                values = list(root.regression.values.items())
                if len(values) == 1:
                    key, value = values[0]
                    try:
                        return f"{key}: {float(value):.3f}"
                    except Exception:
                        return f"{key}: {value}"
                shown = []
                for key, value in values[:4]:
                    try:
                        shown.append(f"{key}: {float(value):.3f}")
                    except Exception:
                        shown.append(f"{key}: {value}")
                extra = f" (+{len(values) - 4} more)" if len(values) > 4 else ""
                return f"{', '.join(shown)}{extra}"
            return str(root.node_id)
        if prediction.binary_classification is not None:
            return prediction.binary_classification.value
        return "NON_BINARY"

    def _prediction_type_from_prediction(self, prediction: PredictionResult) -> str:
        root = prediction.root_prediction
        if root is not None and getattr(root, "mode", None) is not None:
            return str(root.mode.value)
        if prediction.binary_classification is not None:
            return "binary_classification"
        return "unknown"

    def _classification_label_from_prediction(self, prediction: PredictionResult) -> Optional[str]:
        root = prediction.root_prediction
        if root is not None and getattr(root, "mode", None) is not None:
            mode_value = str(root.mode.value)
            if mode_value.endswith("classification"):
                cls = getattr(root, "classification", None)
                if cls is not None:
                    return str(getattr(cls, "predicted_label", "") or "").strip() or None
        if prediction.binary_classification is not None:
            return prediction.binary_classification.value
        return None
