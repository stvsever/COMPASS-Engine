"""Tools module for COMPASS multi-agent system."""

from .base_tool import BaseTool, ToolOutput, get_tool
from .unimodal_compressor import UnimodalCompressor
from .multimodal_narrative import MultimodalNarrativeCreator
from .hypothesis_generator import HypothesisGenerator
from .code_executor import CodeExecutor
from .feature_synthesizer import FeatureSynthesizer
from .clinical_ranker import ClinicalRelevanceRanker
from .anomaly_narrative import AnomalyNarrativeBuilder
from .phenotype_representation import PhenotypeRepresentation
from .differential_diagnosis import DifferentialDiagnosis
from .chunk_evidence_extractor import ChunkEvidenceExtractor

# Tool registry
TOOL_REGISTRY = {
    "UnimodalCompressor": UnimodalCompressor,
    "MultimodalNarrativeCreator": MultimodalNarrativeCreator,
    "HypothesisGenerator": HypothesisGenerator,
    "CodeExecutor": CodeExecutor,
    "FeatureSynthesizer": FeatureSynthesizer,
    "ClinicalRelevanceRanker": ClinicalRelevanceRanker,
    "AnomalyNarrativeBuilder": AnomalyNarrativeBuilder,
    "PhenotypeRepresentation": PhenotypeRepresentation,
    "DifferentialDiagnosis": DifferentialDiagnosis,
    "ChunkEvidenceExtractor": ChunkEvidenceExtractor,
}

__all__ = [
    "BaseTool",
    "ToolOutput",
    "get_tool",
    "UnimodalCompressor",
    "MultimodalNarrativeCreator",
    "HypothesisGenerator",
    "CodeExecutor",
    "FeatureSynthesizer",
    "ClinicalRelevanceRanker",
    "AnomalyNarrativeBuilder",
    "PhenotypeRepresentation",
    "DifferentialDiagnosis",
    "ChunkEvidenceExtractor",
    "TOOL_REGISTRY",
]
