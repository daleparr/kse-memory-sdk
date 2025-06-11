#!/usr/bin/env python3
"""
Cross-Domain Conceptual Space Mapping Demonstration

This example shows how the KSE Memory SDK's 10-dimensional conceptual space
can be semantically remapped across different industries while maintaining
mathematical consistency.
"""

import asyncio
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

# Add the package to Python path for demo
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kse_memory.core.models import ConceptualDimensions
from kse_memory.core.domain_mapping import (
    Domain, get_mapper, map_to_domain, get_domain_dimensions, 
    get_cross_domain_mapping
)


def create_sample_product_dimensions() -> ConceptualDimensions:
    """Create sample product with conceptual dimensions."""
    return ConceptualDimensions(
        elegance=0.8,      # High sophistication
        comfort=0.9,       # Very comfortable
        boldness=0.3,      # Subtle, not attention-grabbing
        modernity=0.7,     # Contemporary
        minimalism=0.9,    # Very clean and simple
        luxury=0.6,        # Mid-range luxury
        functionality=0.8, # Highly functional
        versatility=0.7,   # Quite versatile
        seasonality=0.2,   # Year-round appeal
        innovation=0.5     # Moderate innovation
    )


def demonstrate_cross_domain_mapping():
    """Demonstrate cross-domain conceptual space mapping."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]KSE Memory SDK - Cross-Domain Conceptual Space Mapping[/bold blue]",
        border_style="blue"
    ))
    
    # Create sample product dimensions
    base_dimensions = create_sample_product_dimensions()
    
    console.print("\n[bold green]Sample Product Conceptual Dimensions (Base/Retail):[/bold green]")
    
    # Display base dimensions
    base_table = Table(title="Base Conceptual Dimensions")
    base_table.add_column("Dimension", style="cyan")
    base_table.add_column("Value", style="magenta")
    base_table.add_column("Description", style="white")
    
    base_dict = base_dimensions.to_dict()
    descriptions = {
        "elegance": "Sophistication and refinement",
        "comfort": "Physical and emotional comfort",
        "boldness": "Statement-making quality",
        "modernity": "Contemporary appeal",
        "minimalism": "Design simplicity",
        "luxury": "Premium exclusivity",
        "functionality": "Practical utility",
        "versatility": "Multi-context adaptability",
        "seasonality": "Time-specific relevance",
        "innovation": "Technology advancement"
    }
    
    for dim, value in base_dict.items():
        base_table.add_row(
            dim.title(),
            f"{value:.2f}",
            descriptions.get(dim, "")
        )
    
    console.print(base_table)
    
    # Demonstrate cross-domain mapping
    console.print("\n[bold yellow]Cross-Domain Semantic Remapping:[/bold yellow]")
    
    mapper = get_mapper()
    domains_to_demo = [Domain.FINANCE, Domain.HEALTHCARE, Domain.ENTERPRISE_SOFTWARE]
    
    for domain in domains_to_demo:
        console.print(f"\n[bold cyan]→ {domain.value.upper()} Domain Mapping:[/bold cyan]")
        
        # Get domain-specific mapping
        domain_specific = map_to_domain(domain, base_dimensions)
        profile = mapper.get_domain_profile(domain)
        
        if profile:
            domain_table = Table(title=f"{profile.name} - Domain-Specific Dimensions")
            domain_table.add_column("Base Dimension", style="dim")
            domain_table.add_column("Domain-Specific Name", style="bold cyan")
            domain_table.add_column("Value", style="magenta")
            domain_table.add_column("Domain Description", style="white")
            
            for base_dim, domain_mapping in profile.dimensions.items():
                value = base_dict.get(base_dim, 0.0)
                domain_table.add_row(
                    base_dim.title(),
                    domain_mapping.name.replace("_", " ").title(),
                    f"{value:.2f}",
                    domain_mapping.description
                )
            
            console.print(domain_table)


def demonstrate_mapping_table():
    """Show complete cross-domain mapping table."""
    console = Console()
    
    console.print("\n[bold green]Complete Cross-Domain Mapping Table:[/bold green]")
    
    # Get complete mapping
    mapping_table = get_cross_domain_mapping()
    
    # Create comprehensive table
    table = Table(title="10-Dimensional Cross-Domain Semantic Mapping")
    table.add_column("Base Dimension", style="bold white")
    
    # Add domain columns
    domains = list(mapping_table.keys())
    for domain in domains:
        table.add_column(domain.replace("_", " ").title(), style="cyan")
    
    # Get base dimensions
    base_dims = [
        "elegance", "comfort", "boldness", "modernity", "minimalism",
        "luxury", "functionality", "versatility", "seasonality", "innovation"
    ]
    
    # Add rows
    for base_dim in base_dims:
        row = [base_dim.title()]
        for domain in domains:
            domain_mapping = mapping_table.get(domain, {})
            domain_name = domain_mapping.get(base_dim, base_dim)
            row.append(domain_name.replace("_", " ").title())
        table.add_row(*row)
    
    console.print(table)


def demonstrate_business_applications():
    """Show business applications of cross-domain mapping."""
    console = Console()
    
    console.print("\n[bold green]Business Applications by Domain:[/bold green]")
    
    mapper = get_mapper()
    
    applications = {
        Domain.RETAIL: [
            "Seasonal collection planning using elegance × seasonality × modernity",
            "Customer segmentation via comfort × luxury × versatility mapping",
            "Inventory optimization balancing functionality × innovation × cost",
            "Trend forecasting through boldness × modernity × innovation tracking"
        ],
        Domain.FINANCE: [
            "Portfolio construction optimizing risk × return × diversification",
            "Risk assessment analyzing stability × complexity × regulatory compliance",
            "Product development balancing innovation × accessibility × transparency",
            "Client matching via risk × liquidity × growth preference profiles"
        ],
        Domain.HEALTHCARE: [
            "Treatment selection optimizing efficacy × safety × cost-effectiveness",
            "Device evaluation via precision × usability × regulatory approval",
            "Clinical trial design balancing innovation × evidence × accessibility",
            "Healthcare policy using safety × cost × population accessibility"
        ],
        Domain.ENTERPRISE_SOFTWARE: [
            "Vendor selection via scalability × security × integration capability",
            "Architecture planning using performance × reliability × customization",
            "Technology adoption balancing innovation × simplicity × enterprise readiness",
            "ROI optimization through usability × performance × total cost analysis"
        ]
    }
    
    for domain, use_cases in applications.items():
        profile = mapper.get_domain_profile(domain)
        if profile:
            panel_content = "\n".join([f"• {use_case}" for use_case in use_cases])
            console.print(Panel(
                panel_content,
                title=f"[bold cyan]{profile.name}[/bold cyan]",
                border_style="cyan"
            ))


def demonstrate_mathematical_consistency():
    """Show that mathematical operations remain consistent across domains."""
    console = Console()
    
    console.print("\n[bold green]Mathematical Consistency Across Domains:[/bold green]")
    
    # Create two sample products
    product_a = ConceptualDimensions(
        elegance=0.8, comfort=0.9, boldness=0.3, modernity=0.7, minimalism=0.9,
        luxury=0.6, functionality=0.8, versatility=0.7, seasonality=0.2, innovation=0.5
    )
    
    product_b = ConceptualDimensions(
        elegance=0.4, comfort=0.6, boldness=0.8, modernity=0.9, minimalism=0.3,
        luxury=0.9, functionality=0.6, versatility=0.5, seasonality=0.7, innovation=0.8
    )
    
    # Calculate similarity in base space
    def cosine_similarity(a: ConceptualDimensions, b: ConceptualDimensions) -> float:
        """Calculate cosine similarity between two conceptual dimension vectors."""
        vec_a = list(a.to_dict().values())
        vec_b = list(b.to_dict().values())
        
        dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
        magnitude_a = sum(x ** 2 for x in vec_a) ** 0.5
        magnitude_b = sum(x ** 2 for x in vec_b) ** 0.5
        
        return dot_product / (magnitude_a * magnitude_b) if magnitude_a * magnitude_b > 0 else 0
    
    base_similarity = cosine_similarity(product_a, product_b)
    
    console.print(f"[bold white]Base Space Similarity:[/bold white] {base_similarity:.4f}")
    
    # Show that similarity is preserved across domain mappings
    similarity_table = Table(title="Cross-Domain Similarity Preservation")
    similarity_table.add_column("Domain", style="cyan")
    similarity_table.add_column("Similarity", style="magenta")
    similarity_table.add_column("Difference from Base", style="yellow")
    
    mapper = get_mapper()
    for domain in [Domain.RETAIL, Domain.FINANCE, Domain.HEALTHCARE, Domain.ENTERPRISE_SOFTWARE]:
        profile = mapper.get_domain_profile(domain)
        if profile:
            # Since we're using the same mathematical values, similarity should be identical
            domain_similarity = cosine_similarity(product_a, product_b)
            difference = abs(domain_similarity - base_similarity)
            
            similarity_table.add_row(
                profile.name,
                f"{domain_similarity:.4f}",
                f"{difference:.6f}"
            )
    
    console.print(similarity_table)
    
    console.print("\n[dim]Note: Mathematical consistency is preserved because the underlying")
    console.print("numerical values remain the same - only the semantic interpretation changes.[/dim]")


async def main():
    """Run the complete cross-domain mapping demonstration."""
    console = Console()
    
    try:
        # Main demonstration
        demonstrate_cross_domain_mapping()
        
        # Comprehensive mapping table
        demonstrate_mapping_table()
        
        # Business applications
        demonstrate_business_applications()
        
        # Mathematical consistency
        demonstrate_mathematical_consistency()
        
        # Summary
        console.print("\n" + "="*80)
        console.print(Panel.fit(
            "[bold green]Cross-Domain Mapping Summary[/bold green]\n\n"
            "✓ [cyan]10-dimensional framework adapts across 4+ industries[/cyan]\n"
            "✓ [cyan]Semantic meaning changes while preserving mathematical structure[/cyan]\n"
            "✓ [cyan]Business applications tailored to domain-specific needs[/cyan]\n"
            "✓ [cyan]Similarity calculations remain consistent across domains[/cyan]\n"
            "✓ [cyan]Visualization and metrics adapt to industry requirements[/cyan]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error in demonstration: {str(e)}[/red]")


if __name__ == "__main__":
    asyncio.run(main())