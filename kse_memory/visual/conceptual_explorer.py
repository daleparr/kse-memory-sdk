"""
KSE Memory SDK - Conceptual Space Explorer

Interactive 3D visualization of conceptual spaces with
domain-specific adaptations for different industries.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..core.memory import KSEMemory
from ..core.models import Product, ConceptualDimensions


@dataclass
class ConceptualMapping:
    """Mapping configuration for domain-specific conceptual spaces."""
    domain: str
    dimensions: List[str]
    dimension_labels: Dict[str, str]
    color_scheme: Dict[str, str]
    clustering_rules: Dict[str, Any]
    visualization_config: Dict[str, Any]


class ConceptualSpaceExplorer:
    """
    Interactive 3D explorer for conceptual spaces across domains.
    
    Adapts visualization and interaction patterns based on:
    - Industry domain (retail, finance, healthcare, etc.)
    - Product types and characteristics
    - User interaction patterns
    - Business intelligence requirements
    
    Example:
        explorer = ConceptualSpaceExplorer(kse_memory)
        
        # Retail fashion visualization
        retail_data = await explorer.get_space_data(
            domain="retail_fashion",
            focus_dimensions=["elegance", "comfort", "boldness"]
        )
        
        # Healthcare device visualization  
        healthcare_data = await explorer.get_space_data(
            domain="healthcare_devices",
            focus_dimensions=["precision", "safety", "innovation"]
        )
    """
    
    def __init__(self, kse_memory: KSEMemory):
        """Initialize conceptual space explorer."""
        self.kse_memory = kse_memory
        self.domain_mappings = self._initialize_domain_mappings()
    
    def _initialize_domain_mappings(self) -> Dict[str, ConceptualMapping]:
        """Initialize domain-specific conceptual mappings."""
        return {
            "retail_fashion": ConceptualMapping(
                domain="retail_fashion",
                dimensions=["elegance", "comfort", "boldness", "modernity", "minimalism", "luxury", "functionality", "versatility", "seasonality", "innovation"],
                dimension_labels={
                    "elegance": "Sophistication & Refinement",
                    "comfort": "Physical & Emotional Comfort", 
                    "boldness": "Statement & Attention-Grabbing",
                    "modernity": "Contemporary & Cutting-Edge",
                    "minimalism": "Simplicity & Clean Design",
                    "luxury": "Premium Quality & Exclusivity",
                    "functionality": "Practical Utility & Performance",
                    "versatility": "Adaptability Across Contexts",
                    "seasonality": "Time-Specific Relevance",
                    "innovation": "Novel Features & Technology"
                },
                color_scheme={
                    "elegance": "#8B5CF6",  # Purple
                    "comfort": "#10B981",   # Green
                    "boldness": "#F59E0B",  # Orange
                    "modernity": "#3B82F6", # Blue
                    "minimalism": "#6B7280", # Gray
                    "luxury": "#F59E0B",    # Gold
                    "functionality": "#EF4444", # Red
                    "versatility": "#8B5CF6", # Purple
                    "seasonality": "#06B6D4", # Cyan
                    "innovation": "#EC4899"  # Pink
                },
                clustering_rules={
                    "primary_clusters": ["elegance", "comfort", "boldness"],
                    "secondary_clusters": ["functionality", "luxury"],
                    "cluster_threshold": 0.7
                },
                visualization_config={
                    "default_view": "3d_scatter",
                    "interaction_mode": "explore",
                    "animation_speed": 1.0,
                    "point_size_range": [2, 8]
                }
            ),
            
            "finance_products": ConceptualMapping(
                domain="finance_products",
                dimensions=["risk_level", "liquidity", "growth_potential", "stability", "complexity", "accessibility", "regulatory_compliance", "innovation", "transparency", "diversification"],
                dimension_labels={
                    "risk_level": "Investment Risk Profile",
                    "liquidity": "Asset Liquidity & Accessibility",
                    "growth_potential": "Expected Growth & Returns",
                    "stability": "Market Stability & Predictability",
                    "complexity": "Product Complexity & Understanding",
                    "accessibility": "Minimum Investment & Barriers",
                    "regulatory_compliance": "Regulatory Oversight & Protection",
                    "innovation": "Financial Innovation & Technology",
                    "transparency": "Fee Transparency & Disclosure",
                    "diversification": "Portfolio Diversification Value"
                },
                color_scheme={
                    "risk_level": "#EF4444",    # Red (high risk)
                    "liquidity": "#10B981",     # Green (liquid)
                    "growth_potential": "#F59E0B", # Orange (growth)
                    "stability": "#3B82F6",     # Blue (stable)
                    "complexity": "#8B5CF6",    # Purple (complex)
                    "accessibility": "#06B6D4", # Cyan (accessible)
                    "regulatory_compliance": "#6B7280", # Gray (regulated)
                    "innovation": "#EC4899",    # Pink (innovative)
                    "transparency": "#10B981",  # Green (transparent)
                    "diversification": "#F59E0B" # Orange (diversified)
                },
                clustering_rules={
                    "primary_clusters": ["risk_level", "growth_potential", "stability"],
                    "secondary_clusters": ["liquidity", "accessibility"],
                    "cluster_threshold": 0.6
                },
                visualization_config={
                    "default_view": "risk_return_matrix",
                    "interaction_mode": "analyze",
                    "animation_speed": 0.8,
                    "point_size_range": [3, 10]
                }
            ),
            
            "healthcare_devices": ConceptualMapping(
                domain="healthcare_devices",
                dimensions=["precision", "safety", "usability", "portability", "cost_effectiveness", "regulatory_approval", "innovation", "reliability", "patient_comfort", "clinical_efficacy"],
                dimension_labels={
                    "precision": "Measurement Precision & Accuracy",
                    "safety": "Patient & Operator Safety",
                    "usability": "Ease of Use & Training Requirements",
                    "portability": "Mobility & Space Requirements",
                    "cost_effectiveness": "Cost vs Clinical Value",
                    "regulatory_approval": "FDA/CE Approval Status",
                    "innovation": "Technological Innovation",
                    "reliability": "Device Reliability & Uptime",
                    "patient_comfort": "Patient Experience & Comfort",
                    "clinical_efficacy": "Clinical Outcomes & Effectiveness"
                },
                color_scheme={
                    "precision": "#3B82F6",     # Blue (precise)
                    "safety": "#10B981",       # Green (safe)
                    "usability": "#F59E0B",    # Orange (usable)
                    "portability": "#06B6D4",  # Cyan (portable)
                    "cost_effectiveness": "#8B5CF6", # Purple (cost-effective)
                    "regulatory_approval": "#6B7280", # Gray (approved)
                    "innovation": "#EC4899",   # Pink (innovative)
                    "reliability": "#10B981",  # Green (reliable)
                    "patient_comfort": "#F59E0B", # Orange (comfortable)
                    "clinical_efficacy": "#EF4444" # Red (effective)
                },
                clustering_rules={
                    "primary_clusters": ["precision", "safety", "clinical_efficacy"],
                    "secondary_clusters": ["usability", "cost_effectiveness"],
                    "cluster_threshold": 0.8
                },
                visualization_config={
                    "default_view": "clinical_matrix",
                    "interaction_mode": "evaluate",
                    "animation_speed": 0.6,
                    "point_size_range": [4, 12]
                }
            ),
            
            "enterprise_software": ConceptualMapping(
                domain="enterprise_software",
                dimensions=["scalability", "security", "usability", "integration", "performance", "cost_efficiency", "support_quality", "innovation", "compliance", "customization"],
                dimension_labels={
                    "scalability": "System Scalability & Growth",
                    "security": "Data Security & Privacy",
                    "usability": "User Experience & Adoption",
                    "integration": "System Integration Capabilities",
                    "performance": "Speed & Reliability",
                    "cost_efficiency": "Total Cost of Ownership",
                    "support_quality": "Vendor Support & Documentation",
                    "innovation": "Feature Innovation & Roadmap",
                    "compliance": "Regulatory & Industry Compliance",
                    "customization": "Customization & Flexibility"
                },
                color_scheme={
                    "scalability": "#10B981",   # Green (scalable)
                    "security": "#EF4444",     # Red (secure)
                    "usability": "#F59E0B",    # Orange (usable)
                    "integration": "#3B82F6",  # Blue (integrated)
                    "performance": "#EC4899",  # Pink (performant)
                    "cost_efficiency": "#8B5CF6", # Purple (cost-efficient)
                    "support_quality": "#06B6D4", # Cyan (supported)
                    "innovation": "#F59E0B",   # Orange (innovative)
                    "compliance": "#6B7280",   # Gray (compliant)
                    "customization": "#10B981" # Green (customizable)
                },
                clustering_rules={
                    "primary_clusters": ["scalability", "security", "performance"],
                    "secondary_clusters": ["usability", "integration"],
                    "cluster_threshold": 0.65
                },
                visualization_config={
                    "default_view": "enterprise_matrix",
                    "interaction_mode": "compare",
                    "animation_speed": 0.7,
                    "point_size_range": [3, 9]
                }
            ),
            
            "real_estate": ConceptualMapping(
                domain="real_estate",
                dimensions=["location_quality", "value_appreciation", "rental_yield", "property_condition", "amenities", "accessibility", "neighborhood_safety", "investment_potential", "maintenance_requirements", "market_liquidity"],
                dimension_labels={
                    "location_quality": "Location Desirability & Prestige",
                    "value_appreciation": "Historical & Projected Appreciation",
                    "rental_yield": "Rental Income Potential",
                    "property_condition": "Physical Condition & Age",
                    "amenities": "Property & Community Amenities",
                    "accessibility": "Transportation & Connectivity",
                    "neighborhood_safety": "Safety & Crime Statistics",
                    "investment_potential": "Long-term Investment Value",
                    "maintenance_requirements": "Upkeep & Maintenance Costs",
                    "market_liquidity": "Ease of Sale & Market Activity"
                },
                color_scheme={
                    "location_quality": "#F59E0B",    # Gold (premium location)
                    "value_appreciation": "#10B981",  # Green (appreciating)
                    "rental_yield": "#3B82F6",       # Blue (income)
                    "property_condition": "#8B5CF6",  # Purple (condition)
                    "amenities": "#EC4899",          # Pink (amenities)
                    "accessibility": "#06B6D4",      # Cyan (accessible)
                    "neighborhood_safety": "#10B981", # Green (safe)
                    "investment_potential": "#F59E0B", # Orange (potential)
                    "maintenance_requirements": "#EF4444", # Red (maintenance)
                    "market_liquidity": "#3B82F6"    # Blue (liquid)
                },
                clustering_rules={
                    "primary_clusters": ["location_quality", "value_appreciation", "investment_potential"],
                    "secondary_clusters": ["rental_yield", "accessibility"],
                    "cluster_threshold": 0.7
                },
                visualization_config={
                    "default_view": "investment_matrix",
                    "interaction_mode": "invest",
                    "animation_speed": 0.9,
                    "point_size_range": [4, 10]
                }
            )
        }
    
    async def get_space_data(
        self,
        domain: str = "retail_fashion",
        focus_dimensions: Optional[List[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        max_products: int = 1000
    ) -> Dict[str, Any]:
        """
        Get conceptual space data for visualization.
        
        Args:
            domain: Domain type for specialized visualization
            focus_dimensions: Specific dimensions to highlight
            filter_criteria: Product filtering criteria
            max_products: Maximum number of products to include
            
        Returns:
            Visualization data structure
        """
        # Get domain mapping
        mapping = self.domain_mappings.get(domain, self.domain_mappings["retail_fashion"])
        
        # Get products from KSE Memory
        products = await self._get_products_for_domain(domain, filter_criteria, max_products)
        
        # Extract conceptual data
        conceptual_data = self._extract_conceptual_data(products, mapping, focus_dimensions)
        
        # Generate clusters
        clusters = self._generate_clusters(conceptual_data, mapping)
        
        # Create visualization configuration
        viz_config = self._create_visualization_config(mapping, focus_dimensions)
        
        return {
            "domain": domain,
            "mapping": {
                "dimensions": mapping.dimensions,
                "labels": mapping.dimension_labels,
                "colors": mapping.color_scheme
            },
            "products": conceptual_data,
            "clusters": clusters,
            "visualization": viz_config,
            "statistics": self._calculate_space_statistics(conceptual_data),
            "interactions": self._get_interaction_config(mapping),
            "metadata": {
                "total_products": len(products),
                "focus_dimensions": focus_dimensions or mapping.dimensions[:3],
                "generated_at": "2024-01-01T00:00:00Z"
            }
        }
    
    async def _get_products_for_domain(
        self,
        domain: str,
        filter_criteria: Optional[Dict[str, Any]],
        max_products: int
    ) -> List[Product]:
        """Get products relevant to the specified domain."""
        # In a real implementation, this would query KSE Memory
        # For now, return sample products based on domain
        
        if domain == "retail_fashion":
            return self._generate_sample_fashion_products(max_products)
        elif domain == "finance_products":
            return self._generate_sample_finance_products(max_products)
        elif domain == "healthcare_devices":
            return self._generate_sample_healthcare_products(max_products)
        elif domain == "enterprise_software":
            return self._generate_sample_software_products(max_products)
        elif domain == "real_estate":
            return self._generate_sample_real_estate_products(max_products)
        else:
            return self._generate_sample_fashion_products(max_products)
    
    def _generate_sample_fashion_products(self, count: int) -> List[Product]:
        """Generate sample fashion products with conceptual dimensions."""
        products = []
        
        fashion_items = [
            ("Premium Running Shoes", "Athletic footwear with responsive cushioning", [0.6, 0.9, 0.4, 0.8, 0.7, 0.3, 0.95, 0.8, 0.5, 0.7]),
            ("Elegant Evening Dress", "Sophisticated silk dress for formal occasions", [0.95, 0.6, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6, 0.3, 0.2]),
            ("Minimalist T-Shirt", "Essential cotton t-shirt for everyday wear", [0.5, 0.8, 0.2, 0.6, 0.95, 0.2, 0.9, 0.95, 0.5, 0.3]),
            ("Bold Geometric Jacket", "Statement bomber with vibrant patterns", [0.4, 0.7, 0.95, 0.9, 0.1, 0.6, 0.7, 0.5, 0.7, 0.8]),
            ("Luxury Leather Handbag", "Handcrafted Italian leather bag", [0.9, 0.5, 0.4, 0.6, 0.7, 0.95, 0.8, 0.7, 0.3, 0.4]),
        ]
        
        dimension_names = ["elegance", "comfort", "boldness", "modernity", "minimalism", "luxury", "functionality", "versatility", "seasonality", "innovation"]
        
        for i, (title, desc, dims) in enumerate(fashion_items * (count // len(fashion_items) + 1)):
            if len(products) >= count:
                break
                
            # Add some variation
            varied_dims = [max(0, min(1, d + np.random.normal(0, 0.1))) for d in dims]
            
            conceptual_dims = ConceptualDimensions(**dict(zip(dimension_names, varied_dims)))
            
            product = Product(
                id=f"fashion_{i+1:03d}",
                title=f"{title} {i+1}",
                description=desc,
                price=np.random.uniform(29.99, 299.99),
                category="Fashion",
                tags=["fashion", "apparel"],
                conceptual_dimensions=conceptual_dims
            )
            products.append(product)
        
        return products[:count]
    
    def _generate_sample_finance_products(self, count: int) -> List[Product]:
        """Generate sample financial products."""
        products = []
        
        finance_items = [
            ("High-Yield Savings", "Competitive savings with 4.5% APY", [0.2, 0.9, 0.3, 0.8, 0.2, 0.9, 0.9, 0.6, 0.9, 0.3]),
            ("Growth ETF Portfolio", "Diversified growth-focused ETFs", [0.7, 0.6, 0.8, 0.6, 0.6, 0.7, 0.8, 0.9, 0.7, 0.8]),
            ("Corporate Bonds", "Investment-grade corporate bonds", [0.3, 0.8, 0.4, 0.9, 0.4, 0.8, 0.9, 0.5, 0.9, 0.2]),
            ("Cryptocurrency Fund", "Diversified crypto investment fund", [0.9, 0.3, 0.6, 0.4, 0.8, 0.4, 0.6, 0.7, 0.6, 0.9]),
            ("Real Estate REIT", "Commercial real estate investment trust", [0.5, 0.7, 0.5, 0.7, 0.5, 0.6, 0.8, 0.8, 0.8, 0.4]),
        ]
        
        dimension_names = ["risk_level", "liquidity", "growth_potential", "stability", "complexity", "accessibility", "regulatory_compliance", "innovation", "transparency", "diversification"]
        
        for i, (title, desc, dims) in enumerate(finance_items * (count // len(finance_items) + 1)):
            if len(products) >= count:
                break
                
            varied_dims = [max(0, min(1, d + np.random.normal(0, 0.1))) for d in dims]
            conceptual_dims = ConceptualDimensions(**dict(zip(dimension_names, varied_dims)))
            
            product = Product(
                id=f"finance_{i+1:03d}",
                title=f"{title} {i+1}",
                description=desc,
                price=np.random.uniform(0, 10000),
                category="Financial Product",
                tags=["finance", "investment"],
                conceptual_dimensions=conceptual_dims
            )
            products.append(product)
        
        return products[:count]
    
    def _generate_sample_healthcare_products(self, count: int) -> List[Product]:
        """Generate sample healthcare products."""
        products = []
        
        healthcare_items = [
            ("Digital Blood Pressure Monitor", "Clinically validated BP monitor", [0.9, 0.8, 0.8, 0.7, 0.6, 0.9, 0.7, 0.8, 0.8, 0.8]),
            ("MRI Imaging System", "High-resolution 3T MRI system", [0.95, 0.7, 0.6, 0.3, 0.4, 0.9, 0.95, 0.8, 0.6, 0.95]),
            ("Surgical Instrument Set", "Precision titanium surgical tools", [0.95, 0.8, 0.4, 0.6, 0.5, 0.9, 0.9, 0.8, 0.7, 0.7]),
            ("Patient Monitoring System", "Continuous vital signs monitoring", [0.8, 0.9, 0.7, 0.8, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8]),
            ("Portable Ultrasound", "Handheld ultrasound device", [0.7, 0.8, 0.8, 0.9, 0.7, 0.6, 0.8, 0.7, 0.8, 0.8]),
        ]
        
        dimension_names = ["precision", "safety", "usability", "portability", "cost_effectiveness", "regulatory_approval", "innovation", "reliability", "patient_comfort", "clinical_efficacy"]
        
        for i, (title, desc, dims) in enumerate(healthcare_items * (count // len(healthcare_items) + 1)):
            if len(products) >= count:
                break
                
            varied_dims = [max(0, min(1, d + np.random.normal(0, 0.05))) for d in dims]
            conceptual_dims = ConceptualDimensions(**dict(zip(dimension_names, varied_dims)))
            
            product = Product(
                id=f"healthcare_{i+1:03d}",
                title=f"{title} {i+1}",
                description=desc,
                price=np.random.uniform(99.99, 2500000),
                category="Medical Device",
                tags=["healthcare", "medical"],
                conceptual_dimensions=conceptual_dims
            )
            products.append(product)
        
        return products[:count]
    
    def _generate_sample_software_products(self, count: int) -> List[Product]:
        """Generate sample enterprise software products."""
        # Implementation similar to above but for software
        return []
    
    def _generate_sample_real_estate_products(self, count: int) -> List[Product]:
        """Generate sample real estate products."""
        # Implementation similar to above but for real estate
        return []
    
    def _extract_conceptual_data(
        self,
        products: List[Product],
        mapping: ConceptualMapping,
        focus_dimensions: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Extract conceptual data for visualization."""
        conceptual_data = []
        
        for product in products:
            if not product.conceptual_dimensions:
                continue
            
            dims_dict = product.conceptual_dimensions.to_dict()
            
            # Map to domain-specific dimensions
            mapped_dims = {}
            for dim in mapping.dimensions:
                mapped_dims[dim] = dims_dict.get(dim, 0.5)  # Default to neutral
            
            # Calculate 3D coordinates for primary visualization
            if focus_dimensions and len(focus_dimensions) >= 3:
                coords = [mapped_dims[dim] for dim in focus_dimensions[:3]]
            else:
                coords = [mapped_dims[dim] for dim in mapping.dimensions[:3]]
            
            conceptual_data.append({
                "id": product.id,
                "title": product.title,
                "description": product.description,
                "price": product.price,
                "category": product.category,
                "dimensions": mapped_dims,
                "coordinates": coords,
                "color": self._calculate_product_color(mapped_dims, mapping),
                "size": self._calculate_product_size(mapped_dims, mapping),
                "metadata": product.metadata or {}
            })
        
        return conceptual_data
    
    def _calculate_product_color(self, dimensions: Dict[str, float], mapping: ConceptualMapping) -> str:
        """Calculate product color based on dominant dimensions."""
        # Find the dimension with highest value
        max_dim = max(dimensions.items(), key=lambda x: x[1])
        return mapping.color_scheme.get(max_dim[0], "#6B7280")
    
    def _calculate_product_size(self, dimensions: Dict[str, float], mapping: ConceptualMapping) -> float:
        """Calculate product size based on overall conceptual strength."""
        # Use average of all dimensions to determine size
        avg_strength = sum(dimensions.values()) / len(dimensions)
        size_range = mapping.visualization_config["point_size_range"]
        return size_range[0] + (size_range[1] - size_range[0]) * avg_strength
    
    def _generate_clusters(self, conceptual_data: List[Dict[str, Any]], mapping: ConceptualMapping) -> List[Dict[str, Any]]:
        """Generate conceptual clusters for visualization."""
        clusters = []
        
        # Simple clustering based on primary dimensions
        primary_dims = mapping.clustering_rules["primary_clusters"]
        threshold = mapping.clustering_rules["cluster_threshold"]
        
        # Group products by similarity in primary dimensions
        cluster_centers = []
        for i, product in enumerate(conceptual_data):
            primary_values = [product["dimensions"][dim] for dim in primary_dims]
            
            # Find nearest cluster or create new one
            assigned = False
            for j, center in enumerate(cluster_centers):
                distance = np.linalg.norm(np.array(primary_values) - np.array(center["values"]))
                if distance < (1 - threshold):
                    center["products"].append(product["id"])
                    assigned = True
                    break
            
            if not assigned:
                cluster_centers.append({
                    "id": f"cluster_{len(cluster_centers)}",
                    "values": primary_values,
                    "products": [product["id"]],
                    "center": primary_values,
                    "dimensions": primary_dims
                })
        
        # Convert to visualization format
        for i, center in enumerate(cluster_centers):
            if len(center["products"]) >= 2:  # Only include clusters with multiple products
                clusters.append({
                    "id": center["id"],
                    "name": f"Cluster {i+1}",
                    "center": center["center"],
                    "products": center["products"],
                    "size": len(center["products"]),
                    "color": f"rgba(99, 179, 237, 0.3)",  # Semi-transparent blue
                    "dimensions": center["dimensions"]
                })
        
        return clusters
    
    def _create_visualization_config(self, mapping: ConceptualMapping, focus_dimensions: Optional[List[str]]) -> Dict[str, Any]:
        """Create visualization configuration."""
        return {
            "type": mapping.visualization_config["default_view"],
            "interaction_mode": mapping.visualization_config["interaction_mode"],
            "animation": {
                "enabled": True,
                "speed": mapping.visualization_config["animation_speed"],
                "transitions": True
            },
            "axes": {
                "x": {
                    "dimension": focus_dimensions[0] if focus_dimensions else mapping.dimensions[0],
                    "label": mapping.dimension_labels.get(focus_dimensions[0] if focus_dimensions else mapping.dimensions[0]),
                    "color": mapping.color_scheme.get(focus_dimensions[0] if focus_dimensions else mapping.dimensions[0])
                },
                "y": {
                    "dimension": focus_dimensions[1] if focus_dimensions and len(focus_dimensions) > 1 else mapping.dimensions[1],
                    "label": mapping.dimension_labels.get(focus_dimensions[1] if focus_dimensions and len(focus_dimensions) > 1 else mapping.dimensions[1]),
                    "color": mapping.color_scheme.get(focus_dimensions[1] if focus_dimensions and len(focus_dimensions) > 1 else mapping.dimensions[1])
                },
                "z": {
                    "dimension": focus_dimensions[2] if focus_dimensions and len(focus_dimensions) > 2 else mapping.dimensions[2],
                    "label": mapping.dimension_labels.get(focus_dimensions[2] if focus_dimensions and len(focus_dimensions) > 2 else mapping.dimensions[2]),
                    "color": mapping.color_scheme.get(focus_dimensions[2] if focus_dimensions and len(focus_dimensions) > 2 else mapping.dimensions[2])
                }
            },
            "controls": {
                "rotation": True,
                "zoom": True,
                "pan": True,
                "filter": True,
                "dimension_selector": True
            },
            "legend": {
                "show": True,
                "position": "top-right",
                "dimensions": mapping.dimensions,
                "colors": mapping.color_scheme
            }
        }
    
    def _calculate_space_statistics(self, conceptual_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about the conceptual space."""
        if not conceptual_data:
            return {}
        
        # Calculate dimension statistics
        dimension_stats = {}
        all_dimensions = conceptual_data[0]["dimensions"].keys()
        
        for dim in all_dimensions:
            values = [product["dimensions"][dim] for product in conceptual_data]
            dimension_stats[dim] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "distribution": "normal"  # Could calculate actual distribution
            }
        
        return {
            "total_products": len(conceptual_data),
            "dimension_count": len(all_dimensions),
            "dimension_statistics": dimension_stats,
            "space_density": len(conceptual_data) / (len(all_dimensions) ** 3),  # Rough density measure
            "coverage": {
                "min_coverage": min(min(product["dimensions"].values()) for product in conceptual_data),
                "max_coverage": max(max(product["dimensions"].values()) for product in conceptual_data),
                "avg_coverage": np.mean([np.mean(list(product["dimensions"].values())) for product in conceptual_data])
            }
        }
    
    def _get_interaction_config(self, mapping: ConceptualMapping) -> Dict[str, Any]:
        """Get interaction configuration for the domain."""
        base_interactions = {
            "hover": {
                "enabled": True,
                "show_details": True,
                "highlight_similar": True
            },
            "click": {
                "enabled": True,
                "action": "select",
                "show_connections": True
            },
            "search": {
                "enabled": True,
                "highlight_results": True,
                "animate_to_results": True
            }
        }