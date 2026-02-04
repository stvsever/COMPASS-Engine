"""
COMPASS: Cognitive Orchestrated Multi-modal Predictive Agent System for Stratification

A hierarchical multi-agent LLM system for binary neuropsychiatric/neurologic 
disorder prediction using UK Biobank multimodal data.

Architecture:
    Orchestrator (GPT-5) → Executor → Tools (GPT-5-nano) → Fusion → Predictor → Critic (GPT-5)
    └── If unsatisfactory ──────────────────────────────────────────────────────────────┘
"""

__version__ = "1.0.0"
__author__ = "COMPASS Development Team"
__description__ = "Cognitive Orchestrated Multi-modal Predictive Agent System for Stratification"

from .main import run_compass_pipeline

__all__ = ["run_compass_pipeline"]
