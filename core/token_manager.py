"""
COMPASS Token Manager

Tracks and enforces token budgets across the pipeline.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from ..config.settings import get_settings

logger = logging.getLogger("compass.token_manager")


@dataclass
class TokenUsage:
    """Token usage for a single component or step."""
    component: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    step_tool: Optional[str] = None
    step_reasoning: Optional[str] = None
    
    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ... (skipping TokenBudget and TokenManager init)




@dataclass
class TokenBudget:
    """Budget allocation and tracking for a component."""
    component: str
    allocated: int
    used: int = 0
    
    @property
    def remaining(self) -> int:
        return max(0, self.allocated - self.used)
    
    @property
    def utilization(self) -> float:
        if self.allocated == 0:
            return 0.0
        return (self.used / self.allocated) * 100
    
    @property
    def is_exceeded(self) -> bool:
        return self.used > self.allocated


class TokenManager:
    """
    Manages token budgets across the COMPASS pipeline.
    
    Features:
    - Budget allocation per component
    - Real-time usage tracking
    - Warnings when approaching limits
    - Hard stops when budgets exceeded
    """
    
    def __init__(self, total_budget: Optional[int] = None):
        self.settings = get_settings()
        self.total_budget = total_budget or self.settings.token_budget.total_budget
        
        # Initialize budgets per component
        self.budgets: Dict[str, TokenBudget] = {
            "orchestrator": TokenBudget(
                component="orchestrator",
                allocated=self.settings.token_budget.orchestrator_budget
            ),
            "executor": TokenBudget(
                component="executor",
                allocated=self.settings.token_budget.executor_budget_per_step * 10  # Estimate 10 steps
            ),
            "fusion": TokenBudget(
                component="fusion",
                allocated=self.settings.token_budget.fusion_budget
            ),
            "predictor": TokenBudget(
                component="predictor",
                allocated=self.settings.token_budget.predictor_budget
            ),
            "critic": TokenBudget(
                component="critic",
                allocated=self.settings.token_budget.critic_budget
            ),
        }
        
        # Track usage history
        self.usage_history: list = []
        
        # Warning thresholds (percentage)
        self.warning_threshold = 80
        self.critical_threshold = 95
        
        logger.info(f"TokenManager initialized with total budget: {self.total_budget}")
        print(f"[TokenManager] Total budget: {self.total_budget:,} tokens")
    
    def record_usage(
        self,
        component: str,
        prompt_tokens: int,
        completion_tokens: int,
        step_id: Optional[int] = None,
        step_tool: Optional[str] = None,
        step_reasoning: Optional[str] = None
    ):
        """
        Record token usage for a component.
        
        Args:
            component: Name of component (orchestrator, executor, etc.)
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            step_id: Optional step ID for executor tracking
            step_tool: Optional tool name
            step_reasoning: Optional rationale
        """
        total = prompt_tokens + completion_tokens
        
        if component not in self.budgets:
            logger.warning(f"Unknown component: {component}")
            self.budgets[component] = TokenBudget(
                component=component,
                allocated=1000000  # Default allocation
            )
        
        self.budgets[component].used += total
        
        # Record to history
        self.usage_history.append(TokenUsage(
            component=component,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            step_tool=step_tool,
            step_reasoning=step_reasoning
        ))
        
        # Check thresholds
        utilization = self.budgets[component].utilization
        
        if utilization >= self.critical_threshold:
            logger.warning(
                f"CRITICAL: {component} token usage at {utilization:.1f}% of budget"
            )
            print(f"[TokenManager] ⚠️ CRITICAL: {component} at {utilization:.1f}% of budget")
        elif utilization >= self.warning_threshold:
            logger.info(
                f"WARNING: {component} token usage at {utilization:.1f}% of budget"
            )
        
        # Log usage
        logger.debug(
            f"Token usage recorded: {component} +{total} tokens "
            f"(now {self.budgets[component].used}/{self.budgets[component].allocated})"
        )
    
    def check_budget(self, component: str, estimated_tokens: int) -> bool:
        """
        Check if a component has enough budget for an operation.
        
        Args:
            component: Component name
            estimated_tokens: Estimated tokens for the operation
        
        Returns:
            True if budget is available, False otherwise
        """
        if component not in self.budgets:
            return True  # Unknown components aren't blocked
        
        budget = self.budgets[component]
        return (budget.used + estimated_tokens) <= budget.allocated
    
    def get_remaining(self, component: str) -> int:
        """Get remaining budget for a component."""
        if component not in self.budgets:
            return self.total_budget  # No specific budget
        return self.budgets[component].remaining
    
    @property
    def total_used(self) -> int:
        """Get total tokens used across all components."""
        return sum(b.used for b in self.budgets.values())
    
    @property
    def total_remaining(self) -> int:
        """Get total remaining budget."""
        return max(0, self.total_budget - self.total_used)
    
    @property
    def total_utilization(self) -> float:
        """Get overall budget utilization percentage."""
        if self.total_budget == 0:
            return 0.0
        return (self.total_used / self.total_budget) * 100
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all token usage."""
        return {
            "total_budget": self.total_budget,
            "total_used": self.total_used,
            "total_remaining": self.total_remaining,
            "utilization_percent": round(self.total_utilization, 1),
            "by_component": {
                name: {
                    "allocated": b.allocated,
                    "used": b.used,
                    "remaining": b.remaining,
                    "utilization": round(b.utilization, 1)
                }
                for name, b in self.budgets.items()
            },
            "call_count": len(self.usage_history)
        }
    
    def get_detailed_usage(self) -> Dict[str, Any]:
        """Get detailed usage report for output files."""
        total_prompt = sum(u.prompt_tokens for u in self.usage_history)
        total_completion = sum(u.completion_tokens for u in self.usage_history)
        
        return {
            "total_tokens": self.total_used,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "calls": [
                {
                    "component": u.component,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total": u.total,
                    "step_tool": u.step_tool,
                    "step_reasoning": u.step_reasoning
                }
                for u in self.usage_history
            ]
        }
    
    def print_summary(self):
        """Print a formatted summary of token usage."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("TOKEN USAGE SUMMARY")
        print("=" * 50)
        print(f"Total: {summary['total_used']:,} / {summary['total_budget']:,} "
              f"({summary['utilization_percent']}%)")
        print("-" * 50)
        
        for component, stats in summary["by_component"].items():
            status = "✓" if stats["used"] <= stats["allocated"] else "✗"
            print(f"{status} {component:12}: {stats['used']:6,} / {stats['allocated']:6,} "
                  f"({stats['utilization']:5.1f}%)")
        
        print("-" * 50)
        print(f"Total API calls: {summary['call_count']}")
        print("=" * 50 + "\n")
    
    def reset(self):
        """Reset all usage tracking (for new participant)."""
        for budget in self.budgets.values():
            budget.used = 0
        self.usage_history = []
        logger.info("TokenManager reset for new participant")
