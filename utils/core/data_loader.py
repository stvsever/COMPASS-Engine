"""
COMPASS Data Loader

Loads and parses participant data files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from ...config.settings import get_settings
from ...data.models.schemas import (
    DataOverview,
    DomainCoverage,
    HierarchicalDeviation,
    DeviationNode,
    MultimodalData,
    NonNumericalData,
)
from ..validation import validate_participant_files

logger = logging.getLogger("compass.data_loader")


@dataclass
class ParticipantData:
    """Container for all participant data."""
    participant_id: str
    data_overview: DataOverview
    hierarchical_deviation: HierarchicalDeviation
    multimodal_data: MultimodalData
    non_numerical_data: NonNumericalData
    raw_files: Dict[str, Path]
    
    def get_token_estimate(self) -> int:
        """Estimate total tokens across all data."""
        total = 0
        
        # From data overview
        if hasattr(self.data_overview, 'total_tokens'):
            total += self.data_overview.total_tokens or 0
        
        # From non-numerical data
        if hasattr(self.non_numerical_data, 'token_count'):
            total += self.non_numerical_data.token_count or 0
        
        return total
    
    def get_available_domains(self) -> list:
        """Get list of domains with data."""
        return [
            name for name, cov in self.data_overview.domain_coverage.items()
            if cov.is_available
        ]


class DataLoader:
    """
    Loads and parses all participant data files.
    
    Expected files in participant directory:
    - data_overview.json
    - multi_modal_data.json
    - non_numerical_data.txt
    - hierarchical_deviation_map.json
    """
    
    def __init__(self):
        self.settings = get_settings()
        logger.info("DataLoader initialized")
    
    def load(self, participant_dir: Path) -> ParticipantData:
        """
        Load all participant data.
        
        Args:
            participant_dir: Path to participant directory
        
        Returns:
            ParticipantData containing all loaded data
        
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If data is invalid
        """
        participant_dir = Path(participant_dir)
        logger.info(f"Loading participant data from: {participant_dir}")
        
        # Validate files exist
        is_valid, errors, file_paths = validate_participant_files(participant_dir)
        if not is_valid:
            error_msg = f"Validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        print(f"[DataLoader] Loading participant data from {participant_dir}")
        
        # Load each file
        data_overview = self._load_data_overview(file_paths["data_overview"])
        print(f"[DataLoader] Loaded data_overview.json - {len(data_overview.domain_coverage)} domains")
        
        hierarchical_deviation = self._load_hierarchical_deviation(
            file_paths["hierarchical_deviation"]
        )
        print(f"[DataLoader] Loaded hierarchical_deviation_map.json")
        
        multimodal_data = self._load_multimodal_data(file_paths["multimodal_data"])
        print(f"[DataLoader] Loaded multi_modal_data.json")
        
        non_numerical_data = self._load_non_numerical_data(
            file_paths["non_numerical_data"],
            data_overview.participant_id
        )
        print(f"[DataLoader] Loaded non_numerical_data.txt")
        
        participant_data = ParticipantData(
            participant_id=data_overview.participant_id,
            data_overview=data_overview,
            hierarchical_deviation=hierarchical_deviation,
            multimodal_data=multimodal_data,
            non_numerical_data=non_numerical_data,
            raw_files=file_paths
        )
        
        logger.info(
            f"Successfully loaded data for participant {data_overview.participant_id}"
        )
        print(f"[DataLoader] ✓ All data loaded for participant {data_overview.participant_id}")
        
        return participant_data
    
    def _load_data_overview(self, file_path: Path) -> DataOverview:
        """Load and parse data_overview.json."""
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        # Parse domain coverage
        domain_coverage = {}
        for domain_name, coverage_data in raw_data.get("domain_coverage", {}).items():
            domain_coverage[domain_name] = DomainCoverage(
                domain_name=domain_name,
                present_leaves=coverage_data.get("present_leaves", 0),
                total_leaves=coverage_data.get("total_leaves", 0),
                coverage_percentage=coverage_data.get("coverage_percentage", 0.0),
                missing_count=coverage_data.get("missing_count", 0),
                total_tokens=coverage_data.get("total_tokens")
            )
        
        return DataOverview(
            participant_id=raw_data.get("participant_id", "unknown"),
            domain_coverage=domain_coverage,
            total_tokens=raw_data.get("total_tokens", 0),
            available_domains=raw_data.get("available_domains", []),
            token_budget=raw_data.get("token_budget")
        )
    
    def _load_hierarchical_deviation(self, file_path: Path) -> HierarchicalDeviation:
        """Load and parse hierarchical_deviation_map.json.
        
        Supports two formats:
        1. Legacy: {"root": {...}, "children": [...]}
        2. UKB nested: {"BRAIN": {"_stats": {...}, "Morphologics": {...}}, ...}
        """
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        # Detect format type
        if "root" in raw_data:
            # Legacy format with explicit tree structure
            root_node = self._parse_deviation_node(raw_data.get("root", {}))
            participant_id = raw_data.get("participant_id", "unknown")
            domain_summaries = raw_data.get("domain_summaries", {})
        else:
            # UKB nested dict format - convert to tree structure
            root_node, domain_summaries = self._parse_ukb_deviation_format(raw_data)
            participant_id = "unknown"  # Not stored in UKB format
        
        return HierarchicalDeviation(
            participant_id=participant_id,
            root=root_node,
            domain_summaries=domain_summaries
        )
    
    def _parse_ukb_deviation_format(
        self, 
        data: Dict[str, Any],
        level: int = 0,
        parent_name: str = "ROOT"
    ) -> tuple:
        """Parse UKB nested dict format into tree structure.
        
        UKB format: {"DOMAIN": {"_stats": {...}, "subdomain": {...}}, ...}
        """
        children = []
        domain_summaries = {}
        
        for key, value in data.items():
            if key == "_stats":
                continue  # Skip stats at this level
            
            if isinstance(value, dict):
                # Get stats for this node
                stats = value.get("_stats", {})
                mean_score = stats.get("mean_abs_score", 0.0)
                n_leaves = stats.get("n_leaves", 0)
                
                # Recursively parse children
                child_node = self._parse_ukb_node(
                    key, value, level + 1
                )
                children.append(child_node)
                
                # Track domain-level summaries
                if level == 0:
                    domain_summaries[key] = {
                        "mean_abs_score": mean_score,
                        "n_leaves": n_leaves
                    }
        
        # Create root node
        root_node = DeviationNode(
            node_id="ROOT",
            node_name="ROOT",
            level=0,
            z_score=None,
            children=children,
            is_leaf=False
        )
        
        return root_node, domain_summaries
    
    def _parse_ukb_node(
        self,
        name: str,
        data: Dict[str, Any],
        level: int
    ) -> DeviationNode:
        """Parse a single UKB deviation node and its children."""
        stats = data.get("_stats", {})
        mean_score = stats.get("mean_abs_score")
        n_leaves = stats.get("n_leaves", 0)
        
        # Parse children recursively
        children = []
        for key, value in data.items():
            if key == "_stats":
                continue
            if isinstance(value, dict) and (
                "_stats" in value or any(
                    isinstance(v, dict) for v in value.values()
                )
            ):
                child_node = self._parse_ukb_node(key, value, level + 1)
                children.append(child_node)
        
        # Determine if leaf (has _stats but no child nodes with _stats)
        is_leaf = len(children) == 0 and "_stats" in data
        
        return DeviationNode(
            node_id=f"{level}_{name}".replace(" ", "_").replace("/", "_"),
            node_name=name,
            level=level,
            z_score=mean_score,  # Using mean_abs_score as representative z_score
            children=children,
            is_leaf=is_leaf
        )
    
    def _parse_deviation_node(self, node_data: Dict[str, Any]) -> DeviationNode:
        """Recursively parse legacy deviation nodes (root/children format)."""
        children = []
        for child_data in node_data.get("children", []):
            children.append(self._parse_deviation_node(child_data))
        
        return DeviationNode(
            node_id=node_data.get("node_id", ""),
            node_name=node_data.get("node_name", ""),
            level=node_data.get("level", 0),
            z_score=node_data.get("z_score"),
            raw_value=node_data.get("raw_value"),
            reference_mean=node_data.get("reference_mean"),
            reference_std=node_data.get("reference_std"),
            direction=node_data.get("direction"),
            children=children,
            is_leaf=node_data.get("is_leaf", len(children) == 0)
        )
    
    def _load_multimodal_data(self, file_path: Path) -> MultimodalData:
        """Load and parse multimodal_data.json.
        
        Supports two formats:
        1. Legacy: {"participant_id": "...", "features": {...}, "metadata": {...}}
        2. UKB nested: {"BRAIN": {"Morphologics": {...}}, "BIOLOGICAL ASSAY": {...}, ...}
        """
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        # Detect format type
        if "participant_id" in raw_data:
            # Legacy format
            return MultimodalData(
                participant_id=raw_data.get("participant_id", "unknown"),
                features=raw_data.get("features", {}),
                metadata=raw_data.get("metadata", {})
            )
        else:
            # UKB format - domain names as top-level keys
            # Convert to features format
            features = {}
            for domain_name, domain_data in raw_data.items():
                if isinstance(domain_data, dict):
                    features[domain_name] = self._flatten_ukb_domain(
                        domain_data,
                        domain_name
                    )
            
            return MultimodalData(
                participant_id="unknown",  # Not stored in UKB format
                features=features,
                metadata={"format": "ukb_nested"}
            )
    
    def _flatten_ukb_domain(
        self, 
        domain_data: Dict[str, Any], 
        domain_name: str,
        path: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Flatten UKB domain data, extracting leaves as feature dicts."""
        if path is None:
            path = []
            
        results = []
        
        for key, value in domain_data.items():
            if key == "_leaves":
                # Found leaf nodes
                for leaf in value:
                    feature_name = leaf.get("feature", "unknown")
                    z_score = leaf.get("z_score")
                    
                    # Create feature dict matching FeatureValue schema
                    # Generate a simple ID from feature name
                    feature_id = feature_name.replace(" ", "_").replace("(", "").replace(")", "").lower()[:50]
                    
                    feature_dict = {
                        "feature_id": feature_id,
                        "field_name": feature_name,
                        "value": z_score,  # Using z-score as value since raw value missing
                        "z_score": z_score,
                        "unit": None,
                        "domain": domain_name,
                        "path_in_hierarchy": path
                    }
                    results.append(feature_dict)
                    
            elif isinstance(value, dict):
                # Recurse into subdomain
                if key != "_stats":
                    child_results = self._flatten_ukb_domain(
                        value, 
                        domain_name,
                        path + [key]
                    )
                    results.extend(child_results)
        
        return results
    
    def _load_non_numerical_data(
        self,
        file_path: Path,
        participant_id: str
    ) -> NonNumericalData:
        """Load and parse non_numerical_data.txt."""
        with open(file_path, 'r') as f:
            raw_text = f.read()
        
        return NonNumericalData.from_text(participant_id, raw_text)


def load_participant_data(participant_dir: Path) -> ParticipantData:
    """
    Convenience function to load participant data.
    
    Args:
        participant_dir: Path to participant directory
    
    Returns:
        ParticipantData object
    """
    loader = DataLoader()
    return loader.load(participant_dir)
