"""
COMPASS Data Schemas

Pydantic models defining the structure of all data objects used in the system.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class TargetCondition(str, Enum):
    """
    Broad categorization of prediction targets.
    Specific phenotypes (e.g., ANXIETY_DISORDERS) are also valid as strings in the system.
    """
    NEUROPSYCHIATRIC = "neuropsychiatric"
    NEUROLOGIC = "neurologic"


class DomainName(str, Enum):
    """Valid data domain names from UK Biobank ontology."""
    BIOLOGICAL_ASSAY = "BIOLOGICAL_ASSAY"
    BRAIN = "BRAIN"
    COGNITION = "COGNITION"
    DEMOGRAPHICS = "DEMOGRAPHICS"
    GENOMICS = "GENOMICS"
    LIFESTYLE_ENVIRONMENT = "LIFESTYLE_ENVIRONMENT"


class AbnormalityDirection(str, Enum):
    """Direction of abnormality for a feature."""
    ELEVATED = "ELEVATED"
    REDUCED = "REDUCED"
    HIGH = "HIGH"      # Alias for ELEVATED
    LOW = "LOW"        # Alias for REDUCED  
    NORMAL = "NORMAL"
    MIXED = "MIXED"


class SeverityLevel(str, Enum):
    """Severity classification for abnormalities."""
    SEVERE = "SEVERE"
    MODERATE = "MODERATE"
    MILD = "MILD"
    NORMAL = "NORMAL"


class ConfidenceLevel(str, Enum):
    """Confidence level for predictions and assessments."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ============================================================================
# Data Overview Schemas
# ============================================================================

class DomainCoverage(BaseModel):
    """Coverage statistics for a single domain."""
    domain_name: str
    present_leaves: int = Field(..., description="Number of leaf nodes with data")
    total_leaves: int = Field(..., description="Total possible leaf nodes")
    coverage_percentage: float = Field(..., ge=0, le=100)
    missing_count: int
    total_tokens: Optional[int] = Field(None, description="Estimated tokens for this domain")
    
    @property
    def is_available(self) -> bool:
        """Check if domain has any data."""
        return self.present_leaves > 0


class DataOverview(BaseModel):
    """
    High-level overview of available data for a participant.
    Corresponds to data_overview.json input file.
    """
    participant_id: str
    domain_coverage: Dict[str, DomainCoverage]
    total_tokens: int = Field(..., description="Total estimated tokens across all data")
    available_domains: List[str] = Field(default_factory=list)
    prediction_target: Optional[Union[TargetCondition, str]] = None
    token_budget: Optional[int] = None
    
    @validator('available_domains', pre=True, always=True)
    def set_available_domains(cls, v, values):
        if not v and 'domain_coverage' in values:
            return [
                name for name, cov in values['domain_coverage'].items()
                if cov.present_leaves > 0
            ]
        return v
    
    def get_domains_by_coverage(self, min_coverage: float = 0.0) -> List[str]:
        """Get domains sorted by coverage percentage."""
        filtered = [
            (name, cov) for name, cov in self.domain_coverage.items()
            if cov.coverage_percentage >= min_coverage
        ]
        return [name for name, _ in sorted(filtered, key=lambda x: -x[1].coverage_percentage)]


# ============================================================================
# Hierarchical Deviation Schemas
# ============================================================================

class DeviationNode(BaseModel):
    """A single node in the hierarchical deviation tree."""
    node_id: str
    node_name: str
    level: int = Field(..., description="Hierarchy level (0 = root)")
    z_score: Optional[float] = Field(None, description="Age-sex normalized z-score")
    raw_value: Optional[Union[float, int, str]] = None  # Can be numeric or categorical
    reference_mean: Optional[float] = None
    reference_std: Optional[float] = None
    direction: Optional[AbnormalityDirection] = None
    children: List["DeviationNode"] = Field(default_factory=list)
    is_leaf: bool = False
    unit: Optional[str] = None  # Added unit field
    severity: Optional[str] = None  # Pre-computed severity
    
    @property
    def is_abnormal(self) -> bool:
        """Check if this node shows abnormal deviation."""
        if self.z_score is None:
            return False
        return abs(self.z_score) > 1.5
    
    @property
    def computed_severity(self) -> SeverityLevel:
        """Classify severity based on z-score."""
        if self.z_score is None:
            return SeverityLevel.NORMAL
        abs_z = abs(self.z_score)
        if abs_z > 3.0:
            return SeverityLevel.SEVERE
        elif abs_z > 2.0:
            return SeverityLevel.MODERATE
        elif abs_z > 1.5:
            return SeverityLevel.MILD
        return SeverityLevel.NORMAL


# Enable self-referencing
DeviationNode.model_rebuild()


class HierarchicalDeviation(BaseModel):
    """
    Complete hierarchical deviation map for a participant.
    Corresponds to hierarchical_deviation_map.json input file.
    """
    participant_id: str
    root: DeviationNode
    domain_summaries: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Pre-computed summaries per domain"
    )
    
    def get_abnormal_nodes(self, min_z: float = 1.5) -> List[DeviationNode]:
        """Get all nodes with abnormal deviations."""
        abnormal = []
        
        def traverse(node: DeviationNode):
            if node.z_score is not None and abs(node.z_score) > min_z:
                abnormal.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(self.root)
        return abnormal
    
    def get_domain_nodes(self, domain: str) -> List[DeviationNode]:
        """Get all nodes belonging to a specific domain."""
        for child in self.root.children:
            if child.node_name.upper() == domain.upper():
                return [child] + self._get_all_descendants(child)
        return []
    
    def _get_all_descendants(self, node: DeviationNode) -> List[DeviationNode]:
        """Recursively get all descendant nodes."""
        descendants = []
        for child in node.children:
            descendants.append(child)
            descendants.extend(self._get_all_descendants(child))
        return descendants


# ============================================================================
# Multimodal Data Schema
# ============================================================================

class FeatureValue(BaseModel):
    """Single feature value with metadata."""
    feature_id: str
    field_name: str
    value: Union[float, int, str, None]
    z_score: Optional[float] = None
    unit: Optional[str] = None
    domain: str
    path_in_hierarchy: List[str] = Field(default_factory=list)


class MultimodalData(BaseModel):
    """
    Container for multimodal data.
    Corresponds to multi_modal_data.json input file.
    """
    participant_id: str
    features: Dict[str, List[FeatureValue]] = Field(
        default_factory=dict,
        description="Features grouped by domain"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_domain_features(self, domain: str) -> List[FeatureValue]:
        """Get all features for a specific domain."""
        return self.features.get(domain, [])
    
    def get_abnormal_features(self, min_z: float = 1.5) -> List[FeatureValue]:
        """Get features with abnormal z-scores across all domains."""
        abnormal = []
        for domain_features in self.features.values():
            for feature in domain_features:
                if feature.z_score is not None and abs(feature.z_score) > min_z:
                    abnormal.append(feature)
        return abnormal


# ============================================================================
# Non-Numerical Data Schema
# ============================================================================

class NonNumericalData(BaseModel):
    """
    Container for non-numerical/textual data.
    Corresponds to non_numerical_data.txt input file.
    """
    participant_id: str
    raw_text: str
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Parsed sections of non-numerical data"
    )
    token_count: Optional[int] = None
    
    @classmethod
    def from_text(cls, participant_id: str, text: str) -> "NonNumericalData":
        """Parse raw text into structured sections."""
        sections = {}
        current_section = "general"
        current_content = []
        
        for line in text.split('\n'):
            if line.startswith('##') or line.startswith('**'):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip('#* ')
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return cls(
            participant_id=participant_id,
            raw_text=text,
            sections=sections
        )


# ============================================================================
# Tool Output Schemas
# ============================================================================

class ToolOutput(BaseModel):
    """Base class for tool outputs."""
    tool_name: str
    execution_time_ms: int
    success: bool
    error: Optional[str] = None


class CompressorOutput(ToolOutput):
    """Output from UnimodalCompressor tool."""
    domain: str
    original_token_count: int
    compressed_token_count: int
    key_abnormalities: List[Dict[str, Any]]
    clinical_narrative: str
    confidence: ConfidenceLevel


class NarrativeOutput(ToolOutput):
    """Output from MultimodalNarrativeCreator tool."""
    domains_integrated: List[str]
    cross_modal_patterns: List[Dict[str, Any]]
    integrated_narrative: str
    evidence_direction: str


class HypothesisOutput(ToolOutput):
    """Output from HypothesisGenerator tool."""
    primary_hypothesis: Dict[str, Any]
    alternative_hypotheses: List[Dict[str, Any]]
    differential_considerations: List[str]


class FeatureSynthesisOutput(ToolOutput):
    """Output from FeatureSynthesizer tool."""
    top_features: List[Dict[str, Any]]
    domain_importance: List[Dict[str, Any]]
    predictive_features: Dict[str, List[str]]


class ClinicalRankingOutput(ToolOutput):
    """Output from ClinicalRelevanceRanker tool."""
    ranked_features: List[Dict[str, Any]]
    clinical_summary: str
    missing_key_features: List[str]


class AnomalyNarrativeOutput(ToolOutput):
    """Output from AnomalyNarrativeBuilder tool."""
    overall_severity: SeverityLevel
    domain_narratives: List[Dict[str, Any]]
    integrated_narrative: str
    clinical_highlights: List[str]
