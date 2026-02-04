"""
COMPASS - Clinical Orchestrated Multi-modal Predictive Agentic Support System

Global Configuration Settings

Centralizes all configuration including model names, token limits, 
retry settings, and file paths.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# System branding
COMPASS_FULL_NAME = "Clinical Orchestrated Multi-modal Predictive Agentic Support System"
COMPASS_VERSION = "1.0.0"


from enum import Enum

class LLMBackend(Enum):
    OPENAI = "openai"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """
    Configuration for LLM models.
    
    NOTE: For TESTING, all models use gpt-5-nano for cost efficiency.
    For PRODUCTION runs, set orchestrator/critic/predictor to gpt-5 for best reasoning.
    Tools always use gpt-5-nano regardless of environment.
    """
    # Backend Selection
    backend: LLMBackend = LLMBackend.OPENAI
    local_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Corrected ID for Qwen2.5 0.5B Instruct
    local_max_tokens: int = 2048
    
    # TESTING MODE: All gpt-5-nano
    # PRODUCTION: Change orchestrator/critic/predictor to "gpt-5"
    orchestrator_model: str = "gpt-5"       # Production: "gpt-5"
    critic_model: str = "gpt-5"             # Production: "gpt-5"
    predictor_model: str = "gpt-5"          # Production: "gpt-5"
    integrator_model: str = "gpt-5"         # Production: "gpt-5"
    tool_model: str = "gpt-5-nano"          # Always gpt-5-nano
    
    # Token limits per model
    orchestrator_max_tokens: int = 16384
    critic_max_tokens: int = 16384
    predictor_max_tokens: int = 16384
    integrator_max_tokens: int = 24000
    tool_max_tokens: int = 16384
    
    # Temperature settings
    orchestrator_temperature: float = 0.3
    critic_temperature: float = 0.2
    predictor_temperature: float = 0.2  # Reduced to avoid verbose loops
    integrator_temperature: float = 0.3
    tool_temperature: float = 0.5


@dataclass
class RetryConfig:
    """Configuration for retry and auto-repair logic."""
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    auto_repair_enabled: bool = True
    max_critic_iterations: int = 3


@dataclass
class TokenBudgetConfig:
    """Token budget constraints for processing."""
    total_budget: int = 1500000
    orchestrator_budget: int = 50000
    executor_budget_per_step: int = 30000
    fusion_budget: int = 90000  # Reduced for gpt-5-nano safety
    integrator_budget: int = 90000 
    predictor_budget: int = 100000
    critic_budget: int = 50000


@dataclass
class PathConfig:
    """File and directory paths configuration."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    # New prompt locations
    agent_prompts_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "agents" / "prompts")
    tool_prompts_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "tools" / "prompts")
    
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "results")
    overview_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "overview")
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.overview_dir.mkdir(parents=True, exist_ok=True)
        self.agent_prompts_dir.mkdir(parents=True, exist_ok=True)
        self.tool_prompts_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """
    Master settings class combining all configuration sections.
    
    Usage:
        settings = get_settings()
        model = settings.models.orchestrator_model
    """
    models: ModelConfig = field(default_factory=ModelConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # API Configuration
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # Logging settings
    log_level: str = "INFO"
    verbose_logging: bool = True
    save_intermediate_outputs: bool = True
    detailed_tool_logging: bool = False  # New flag for capturing raw I/O

    
    # Prediction targets - Removed strict enum to allow dynamic phenotype strings
    # valid_targets: tuple = ("neuropsychiatric", "neurologic")
    
    # Data file names (expected in participant directory)
    data_overview_file: str = "data_overview.json"
    multimodal_data_file: str = "multimodal_data.json"
    non_numerical_data_file: str = "non_numerical_data.txt"
    hierarchical_deviation_file: str = "hierarchical_deviation_map.json"
    
    def validate(self) -> bool:
        """Validate that required settings are present."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return True
    
    def get_participant_files(self, participant_dir: Path) -> Dict[str, Path]:
        """Get paths to all expected participant files."""
        return {
            "data_overview": participant_dir / self.data_overview_file,
            "multimodal_data": participant_dir / self.multimodal_data_file,
            "non_numerical_data": participant_dir / self.non_numerical_data_file,
            "hierarchical_deviation": participant_dir / self.hierarchical_deviation_file,
        }


# Singleton pattern for settings
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reload_settings() -> Settings:
    """Force reload of settings (useful for testing)."""
    global _settings_instance
    _settings_instance = Settings()
    return _settings_instance


# Print configuration on import if verbose
if __name__ == "__main__":
    settings = get_settings()
    print("=" * 60)
    print("COMPASS Configuration")
    print("=" * 60)
    print(f"Orchestrator Model: {settings.models.orchestrator_model}")
    print(f"Tool Model: {settings.models.tool_model}")
    print(f"Total Token Budget: {settings.token_budget.total_budget}")
    print(f"Max Critic Iterations: {settings.retry.max_critic_iterations}")
    print(f"API Key Present: {'Yes' if settings.openai_api_key else 'No'}")
    print("=" * 60)
