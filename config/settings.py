"""
COMPASS - Clinical Ontology-driven Multi-modal Predictive Agentic Support System

Global Configuration Settings

Centralizes all configuration including model names, token limits, 
retry settings, and file paths.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# System branding
COMPASS_FULL_NAME = "Clinical Ontology-driven Multi-modal Predictive Agentic Support System"
COMPASS_VERSION = "1.0.0"


from enum import Enum

class LLMBackend(Enum):
    OPENROUTER = "openrouter"
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
    backend: LLMBackend = LLMBackend.OPENROUTER
    public_model_name: str = "gpt-5-nano"
    public_max_context_tokens: int = 128000
    embedding_model: str = "text-embedding-3-large"
    local_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Corrected ID for Qwen2.5 0.5B Instruct
    local_max_tokens: int = 2048
    # Local backend advanced configuration
    local_backend_type: str = "auto"  # auto|vllm|transformers
    local_dtype: str = "auto"  # auto|float16|bfloat16|float32|fp8
    local_quantization: Optional[str] = None  # e.g., awq|gptq|4bit|8bit|fp8
    local_tensor_parallel_size: int = 1
    local_pipeline_parallel_size: int = 1
    local_gpu_memory_utilization: float = 0.9
    local_max_model_len: int = 0  # 0 = use local_max_tokens / backend default
    local_kv_cache_dtype: Optional[str] = None  # e.g., fp8_e4m3|fp8_e5m2
    local_enforce_eager: bool = False
    local_trust_remote_code: bool = True
    local_attn_implementation: str = "auto"  # transformers: auto|flash_attention_2|sdpa|eager
    
    # TESTING MODE: All gpt-5-nano
    # PRODUCTION: Change orchestrator/critic/predictor/integrator/communicator to "gpt-5"
    orchestrator_model: str = "gpt-5-nano"       # Production: "gpt-5"
    critic_model: str = "gpt-5-nano"             # Production: "gpt-5"
    predictor_model: str = "gpt-5-nano"     # Production: "gpt-5"
    integrator_model: str = "gpt-5-nano"         # Production: "gpt-5"
    communicator_model: str = "gpt-5-nano"       # Production: "gpt-5"
    tool_model: str = "gpt-5-nano"               # Always gpt-5-nano
    
    orchestrator_max_tokens: int = 64000
    critic_max_tokens: int = 64000
    predictor_max_tokens: int = 64000
    integrator_max_tokens: int = 64000
    communicator_max_tokens: int = 64000
    tool_max_tokens: int = 24000
    
    # Temperature settings
    orchestrator_temperature: float = 0.3
    critic_temperature: float = 0.2
    predictor_temperature: float = 0.2  # Reduced to avoid verbose loops
    integrator_temperature: float = 0.3
    communicator_temperature: float = 0.2
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
    fusion_budget: int = 90000  # Reduced for gpt-5(-nano) safety
    integrator_budget: int = 90000 
    predictor_budget: int = 100000
    critic_budget: int = 50000
    
    # Granular controls
    max_agent_input_tokens: int = 20000   # Max context window for agent prompts
    max_agent_output_tokens: int = 4096   # Max tokens for agent generation
    max_tool_input_tokens: int = 20000    # Max input size passed to tools
    max_tool_output_tokens: int = 15000   # Max output size for tools/compressors


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
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    openrouter_site_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", ""))
    openrouter_app_name: str = field(default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "COMPASS"))
    
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

    def _normalize_model_name(self, model_name: Optional[str]) -> str:
        if not model_name:
            return ""
        normalized = str(model_name).strip().lower()
        normalized = re.sub(r"^[a-z0-9_\-]+/", "", normalized)
        return normalized

    def effective_context_window(self, model_name: Optional[str] = None) -> int:
        if self.models.backend == LLMBackend.LOCAL:
            local_len = int(getattr(self.models, "local_max_model_len", 0) or 0)
            if local_len > 0:
                return local_len
            return max(1024, int(getattr(self.models, "local_max_tokens", 2048) or 2048))

        public_ctx = int(getattr(self.models, "public_max_context_tokens", 0) or 0)
        normalized_public = self._normalize_model_name(self.models.public_model_name)
        normalized_requested = self._normalize_model_name(model_name or self.models.public_model_name)
        if public_ctx > 0 and (
            not model_name or normalized_requested == normalized_public
        ):
            return public_ctx

        normalized = normalized_requested
        known_ctx = {
            "gpt-5": 128000,
            "gpt-5-mini": 128000,
            "gpt-5-nano": 128000,
            "gpt-4.1": 128000,
            "gpt-4.1-mini": 128000,
            "gpt-4.1-nano": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }
        if normalized in known_ctx:
            return known_ctx[normalized]
        return max(8192, int(self.models.public_max_context_tokens or 128000))

    def auto_output_token_limit(self, model_name: Optional[str] = None) -> int:
        ctx = self.effective_context_window(model_name=model_name)
        return max(1024, min(64000, int(ctx * 0.5)))
    
    def validate(self) -> bool:
        """Validate that required settings are present."""
        if self.models.backend == LLMBackend.LOCAL:
            return True
        if self.models.backend == LLMBackend.OPENROUTER:
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            return True
        if self.models.backend == LLMBackend.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            return True
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
    api_ok = settings.openrouter_api_key if settings.models.backend == LLMBackend.OPENROUTER else settings.openai_api_key
    print(f"API Key Present: {'Yes' if api_ok else 'No'}")
    print("=" * 60)
