"""
COMPASS Decision Trace

Records decision paths and reasoning throughout the pipeline.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DecisionPoint:
    """A single decision point in the pipeline."""
    decision_id: str
    component: str
    decision_type: str
    input_summary: str
    output_summary: str
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionTrace:
    """
    Records and tracks decision paths through the COMPASS pipeline.
    
    Enables:
    - Auditability of predictions
    - Debugging of unexpected outcomes
    - Transparency for clinical review
    """
    
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.decisions: List[DecisionPoint] = []
        self.start_time = datetime.now()
        self._decision_counter = 0
    
    def record(
        self,
        component: str,
        decision_type: str,
        input_summary: str,
        output_summary: str,
        reasoning: str = "",
        **metadata
    ) -> DecisionPoint:
        """
        Record a decision point.
        
        Args:
            component: Which component made the decision
            decision_type: Type of decision (e.g., PLAN_GENERATION, TOOL_SELECTION)
            input_summary: Brief summary of inputs
            output_summary: Brief summary of outputs
            reasoning: Explanation of decision logic
            **metadata: Additional key-value metadata
        
        Returns:
            The recorded DecisionPoint
        """
        self._decision_counter += 1
        
        decision = DecisionPoint(
            decision_id=f"D{self._decision_counter:03d}",
            component=component,
            decision_type=decision_type,
            input_summary=input_summary[:500],  # Truncate for storage
            output_summary=output_summary[:500],
            reasoning=reasoning[:1000],
            metadata=metadata
        )
        
        self.decisions.append(decision)
        return decision
    
    def record_orchestrator_plan(
        self,
        domains: List[str],
        num_steps: int,
        reasoning: str
    ):
        """Record orchestrator planning decision."""
        self.record(
            component="Orchestrator",
            decision_type="PLAN_GENERATION",
            input_summary=f"Domains: {', '.join(domains)}",
            output_summary=f"Generated {num_steps}-step plan",
            reasoning=reasoning
        )
    
    def record_tool_selection(
        self,
        tool_name: str,
        input_domains: List[str],
        reasoning: str
    ):
        """Record tool selection decision."""
        self.record(
            component="Executor",
            decision_type="TOOL_SELECTION",
            input_summary=f"Processing: {', '.join(input_domains)}",
            output_summary=f"Selected tool: {tool_name}",
            reasoning=reasoning
        )
    
    def record_prediction(
        self,
        classification: str,
        probability: float,
        key_findings: List[str],
        reasoning: str
    ):
        """Record prediction decision."""
        self.record(
            component="Predictor",
            decision_type="PREDICTION_OUTPUT",
            input_summary=f"Key findings: {', '.join(key_findings[:3])}...",
            output_summary=f"{classification} (confidence={probability:.3f})",
            reasoning=reasoning,
            probability=probability,
            classification=classification
        )
    
    def record_critic_verdict(
        self,
        verdict: str,
        checklist_passed: int,
        checklist_total: int,
        reasoning: str
    ):
        """Record critic evaluation decision."""
        self.record(
            component="Critic",
            decision_type="EVALUATION",
            input_summary=f"Checklist: {checklist_passed}/{checklist_total} passed",
            output_summary=verdict,
            reasoning=reasoning
        )
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get full decision trace as dictionaries."""
        return [
            {
                "decision_id": d.decision_id,
                "component": d.component,
                "decision_type": d.decision_type,
                "input_summary": d.input_summary,
                "output_summary": d.output_summary,
                "reasoning": d.reasoning,
                "timestamp": d.timestamp.isoformat(),
                "metadata": d.metadata
            }
            for d in self.decisions
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of decision trace."""
        components = {}
        for d in self.decisions:
            if d.component not in components:
                components[d.component] = 0
            components[d.component] += 1
        
        return {
            "participant_id": self.participant_id,
            "total_decisions": len(self.decisions),
            "decisions_by_component": components,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def to_markdown(self) -> str:
        """Generate markdown representation of decision trace."""
        lines = [
            f"# Decision Trace: {self.participant_id}",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"\nTotal decisions: {len(self.decisions)}",
            "\n---\n"
        ]
        
        for d in self.decisions:
            lines.extend([
                f"## {d.decision_id}: {d.decision_type}",
                f"**Component**: {d.component}",
                f"**Time**: {d.timestamp.strftime('%H:%M:%S')}",
                f"\n**Input**: {d.input_summary}",
                f"\n**Output**: {d.output_summary}",
                f"\n**Reasoning**: {d.reasoning}",
                "\n---\n"
            ])
        
        return "\n".join(lines)
