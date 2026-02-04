"""
COMPASS Patient Report Generator

Generates per-patient final reports with predictions and execution logs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ...config.settings import get_settings
from ...models.prediction_result import PredictionResult, CriticEvaluation


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
                "classification": prediction.binary_classification.value,
                "probability": prediction.probability_score,
                "confidence": prediction.confidence_level.value,
                "target_condition": prediction.target_condition
            },
            
            "evaluation": {
                "verdict": evaluation.verdict.value,
                "confidence_in_verdict": evaluation.confidence_in_verdict,
                "checklist_passed": evaluation.checklist.pass_count,
                "checklist_total": 7
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
        target = pred.get('target_condition', 'N/A')
        
        display_classification = classification
        # Re-inject target for CASE results to match user's exact specification
        if "CASE" in classification:
            display_classification = f"CASE (Phenotype match found for: {target})"

        lines = [
            f"# Patient Report: {report.get('participant_id', 'Unknown')}",
            f"\n**Generated**: {report.get('generated_at', '')}",
            f"\n## Prediction",
            f"- **Classification**: {display_classification}",
            f"- **Probability**: {pred.get('probability', 0):.1%}",
            f"- **Confidence**: {pred.get('confidence', 'N/A')}",
            f"- **Target Condition**: {target}",
            
            f"\n## Evaluation",
            f"- **Verdict**: {eval_data.get('verdict', 'N/A')}",
            f"- **Checklist**: {eval_data.get('checklist_passed', 0)}/{eval_data.get('checklist_total', 7)} passed",
            
            f"\n## Key Findings"
        ]
        
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
