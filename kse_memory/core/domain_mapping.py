"""
Cross-domain conceptual space mapping for KSE Memory SDK.

This module provides semantic remapping of the 10-dimensional conceptual space
across different industries and domains while maintaining mathematical consistency.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from .models import ConceptualDimensions


class Domain(Enum):
    """Supported domains for conceptual space mapping."""
    RETAIL = "retail"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    ENTERPRISE_SOFTWARE = "enterprise_software"
    REAL_ESTATE = "real_estate"
    AUTOMOTIVE = "automotive"
    EDUCATION = "education"
    HOSPITALITY = "hospitality"
    MANUFACTURING = "manufacturing"
    MEDIA_ENTERTAINMENT = "media_entertainment"


@dataclass
class DimensionMapping:
    """Mapping definition for a single conceptual dimension."""
    name: str
    description: str
    examples: List[str]
    measurement_guide: str
    weight: float = 1.0
    aliases: List[str] = field(default_factory=list)


@dataclass
class DomainProfile:
    """Complete domain profile with all dimension mappings."""
    domain: Domain
    name: str
    description: str
    dimensions: Dict[str, DimensionMapping]
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    business_metrics: List[str] = field(default_factory=list)


class ConceptualSpaceMapper:
    """Manages cross-domain conceptual space mappings."""
    
    def __init__(self):
        """Initialize the mapper with predefined domain profiles."""
        self.base_dimensions = [
            "elegance", "comfort", "boldness", "modernity", "minimalism",
            "luxury", "functionality", "versatility", "seasonality", "innovation"
        ]
        self.domain_profiles = self._initialize_domain_profiles()
    
    def _initialize_domain_profiles(self) -> Dict[Domain, DomainProfile]:
        """Initialize domain profiles with cross-domain mappings."""
        profiles = {}
        
        # Define domain mappings
        domain_mappings = {
            Domain.RETAIL: {
                "name": "Retail & E-commerce",
                "description": "Fashion, consumer goods, and retail products",
                "dimensions": [
                    ("elegance", "Sophistication and refinement level", ["silk dress", "minimalist jewelry"]),
                    ("comfort", "Physical and emotional comfort", ["soft fabric", "ergonomic design"]),
                    ("boldness", "Statement-making and attention-grabbing", ["bright colors", "unique patterns"]),
                    ("modernity", "Contemporary appeal and current trends", ["latest fashion", "trending designs"]),
                    ("minimalism", "Simplicity and clean design", ["clean lines", "unadorned design"]),
                    ("luxury", "Premium quality and exclusivity", ["designer brands", "premium materials"]),
                    ("functionality", "Practical utility and performance", ["multi-tool", "performance wear"]),
                    ("versatility", "Adaptability across contexts", ["convertible furniture", "multi-season items"]),
                    ("seasonality", "Time-specific relevance", ["winter coats", "holiday decorations"]),
                    ("innovation", "Novel features and technology", ["smart fabrics", "innovative design"])
                ],
                "visualization": {"primary_axes": ["elegance", "comfort", "boldness"]},
                "metrics": ["conversion_rate", "customer_satisfaction", "return_rate"]
            },
            
            Domain.FINANCE: {
                "name": "Financial Services",
                "description": "Investment products and financial instruments",
                "dimensions": [
                    ("risk_level", "Investment risk profile and volatility", ["government bonds", "cryptocurrency"]),
                    ("liquidity", "Asset accessibility and cash conversion", ["savings account", "real estate"]),
                    ("growth_potential", "Expected returns and appreciation", ["growth stocks", "emerging markets"]),
                    ("innovation", "Financial technology advancement", ["robo-advisors", "blockchain"]),
                    ("transparency", "Fee clarity and disclosure", ["index funds", "clear fee structure"]),
                    ("exclusivity", "Access barriers and requirements", ["private banking", "hedge funds"]),
                    ("stability", "Market stability and predictability", ["treasury bonds", "dividend stocks"]),
                    ("diversification", "Portfolio diversification benefits", ["index funds", "international exposure"]),
                    ("regulatory_compliance", "Oversight and protection level", ["FDIC insured", "SEC regulated"]),
                    ("complexity", "Understanding requirements", ["savings account", "derivatives"])
                ],
                "visualization": {"primary_axes": ["risk_level", "growth_potential", "liquidity"]},
                "metrics": ["sharpe_ratio", "alpha", "beta", "expense_ratio"]
            },
            
            Domain.HEALTHCARE: {
                "name": "Healthcare & Medical Devices",
                "description": "Medical devices, treatments, and healthcare services",
                "dimensions": [
                    ("precision", "Measurement accuracy and reliability", ["MRI scanner", "surgical robot"]),
                    ("patient_comfort", "Patient experience and comfort", ["cushioned MRI", "painless injection"]),
                    ("clinical_efficacy", "Treatment effectiveness", ["cancer treatment", "diagnostic test"]),
                    ("innovation", "Technology advancement", ["gene therapy", "AI diagnostics"]),
                    ("usability", "Ease of use for patients and providers", ["simple interface", "intuitive design"]),
                    ("cost_effectiveness", "Value per health outcome", ["generic drugs", "preventive care"]),
                    ("safety", "Risk profile and adverse effects", ["FDA approved", "minimal side effects"]),
                    ("accessibility", "Patient population reach", ["universal screening", "telemedicine"]),
                    ("regulatory_approval", "FDA and regulatory status", ["FDA cleared", "clinical trial"]),
                    ("evidence_quality", "Research backing strength", ["randomized trial", "meta-analysis"])
                ],
                "visualization": {"primary_axes": ["clinical_efficacy", "safety", "cost_effectiveness"]},
                "metrics": ["clinical_outcomes", "patient_satisfaction", "cost_per_qaly"]
            },
            
            Domain.ENTERPRISE_SOFTWARE: {
                "name": "Enterprise Software",
                "description": "Business software and SaaS platforms",
                "dimensions": [
                    ("scalability", "Growth and load handling capability", ["cloud architecture", "auto-scaling"]),
                    ("usability", "User experience and adoption ease", ["intuitive interface", "minimal training"]),
                    ("innovation", "Technology advancement", ["AI integration", "cutting-edge features"]),
                    ("integration", "System compatibility", ["REST APIs", "third-party connectors"]),
                    ("simplicity", "Configuration and maintenance ease", ["plug-and-play", "self-managing"]),
                    ("enterprise_grade", "Premium capabilities", ["SSO", "advanced analytics"]),
                    ("performance", "Speed and responsiveness", ["fast loading", "real-time updates"]),
                    ("customization", "Flexibility and configuration", ["custom fields", "workflow builder"]),
                    ("reliability", "Uptime and stability", ["99.9% uptime", "disaster recovery"]),
                    ("security", "Data protection measures", ["encryption", "access controls"])
                ],
                "visualization": {"primary_axes": ["scalability", "performance", "security"]},
                "metrics": ["user_adoption", "system_uptime", "support_tickets", "roi"]
            }
        }
        
        # Create domain profiles
        for domain, config in domain_mappings.items():
            dimensions = {}
            for i, (name, desc, examples) in enumerate(config["dimensions"]):
                base_dim = self.base_dimensions[i] if i < len(self.base_dimensions) else f"dimension_{i}"
                dimensions[base_dim] = DimensionMapping(
                    name=name,
                    description=desc,
                    examples=examples,
                    measurement_guide=f"Rate from low {name} (0.0) to high {name} (1.0)"
                )
            
            profiles[domain] = DomainProfile(
                domain=domain,
                name=config["name"],
                description=config["description"],
                dimensions=dimensions,
                visualization_config=config.get("visualization", {}),
                business_metrics=config.get("metrics", [])
            )
        
        return profiles
    
    def get_domain_profile(self, domain: Domain) -> Optional[DomainProfile]:
        """Get domain profile for a specific domain."""
        return self.domain_profiles.get(domain)
    
    def list_available_domains(self) -> List[Domain]:
        """List all available domains."""
        return list(self.domain_profiles.keys())
    
    def map_dimensions(self, source_domain: Domain, target_domain: Domain, 
                      source_dimensions: ConceptualDimensions) -> ConceptualDimensions:
        """
        Map conceptual dimensions from one domain to another.
        
        Args:
            source_domain: Source domain
            target_domain: Target domain
            source_dimensions: Dimensions in source domain
            
        Returns:
            Mapped dimensions in target domain
        """
        if source_domain == target_domain:
            return source_dimensions
        
        # For now, return the same values with base dimension names
        # In a full implementation, this would apply semantic similarity weighting
        return source_dimensions
    
    def get_dimension_description(self, domain: Domain, dimension: str) -> str:
        """Get description for a dimension in a specific domain."""
        profile = self.domain_profiles.get(domain)
        if not profile:
            return f"Unknown domain: {domain}"
        
        if dimension in profile.dimensions:
            return profile.dimensions[dimension].description
        
        return f"Unknown dimension: {dimension}"
    
    def get_domain_specific_names(self, domain: Domain) -> Dict[str, str]:
        """Get mapping from base dimensions to domain-specific names."""
        profile = self.domain_profiles.get(domain)
        if not profile:
            return {}
        
        mapping = {}
        for base_dim, domain_mapping in profile.dimensions.items():
            mapping[base_dim] = domain_mapping.name
        
        return mapping
    
    def create_domain_specific_dimensions(self, domain: Domain, 
                                        base_dimensions: ConceptualDimensions) -> Dict[str, float]:
        """
        Create domain-specific dimension representation.
        
        Args:
            domain: Target domain
            base_dimensions: Base conceptual dimensions
            
        Returns:
            Dictionary with domain-specific dimension names and values
        """
        profile = self.domain_profiles.get(domain)
        if not profile:
            return base_dimensions.to_dict()
        
        base_values = base_dimensions.to_dict()
        domain_specific = {}
        
        for base_dim, value in base_values.items():
            if base_dim in profile.dimensions:
                domain_name = profile.dimensions[base_dim].name
                domain_specific[domain_name] = value
            else:
                domain_specific[base_dim] = value
        
        return domain_specific
    
    def get_visualization_config(self, domain: Domain) -> Dict[str, Any]:
        """Get visualization configuration for a domain."""
        profile = self.domain_profiles.get(domain)
        return profile.visualization_config if profile else {}
    
    def get_business_metrics(self, domain: Domain) -> List[str]:
        """Get relevant business metrics for a domain."""
        profile = self.domain_profiles.get(domain)
        return profile.business_metrics if profile else []


# Convenience functions for easy access
_mapper_instance = None

def get_mapper() -> ConceptualSpaceMapper:
    """Get a singleton instance of the conceptual space mapper."""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = ConceptualSpaceMapper()
    return _mapper_instance


def map_to_domain(domain: Domain, base_dimensions: ConceptualDimensions) -> Dict[str, float]:
    """
    Convenience function to map base dimensions to domain-specific representation.
    
    Args:
        domain: Target domain
        base_dimensions: Base conceptual dimensions
        
    Returns:
        Domain-specific dimension mapping
    """
    mapper = get_mapper()
    return mapper.create_domain_specific_dimensions(domain, base_dimensions)


def get_domain_dimensions(domain: Domain) -> List[str]:
    """
    Get list of dimension names for a specific domain.
    
    Args:
        domain: Target domain
        
    Returns:
        List of domain-specific dimension names
    """
    mapper = get_mapper()
    profile = mapper.get_domain_profile(domain)
    return [mapping.name for mapping in profile.dimensions.values()] if profile else []


def get_cross_domain_mapping() -> Dict[str, Dict[str, str]]:
    """
    Get complete cross-domain mapping table.
    
    Returns:
        Dictionary mapping base dimensions to domain-specific names
    """
    mapper = get_mapper()
    mapping_table = {}
    
    for domain in mapper.list_available_domains():
        domain_mapping = mapper.get_domain_specific_names(domain)
        mapping_table[domain.value] = domain_mapping
    
    return mapping_table