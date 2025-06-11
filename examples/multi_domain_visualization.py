"""
KSE Memory SDK - Multi-Domain Visualization Example

Demonstrates how KSE Memory's visual tools adapt across
different industries and use cases, showcasing the flexibility
of the hybrid AI substrate for product intelligence.
"""

import asyncio
from typing import Dict, Any

from kse_memory.core.memory import KSEMemory
from kse_memory.core.config import KSEConfig
from kse_memory.visual.conceptual_explorer import ConceptualSpaceExplorer
from kse_memory.visual.dashboard import KSEDashboard


async def demonstrate_retail_fashion():
    """Demonstrate KSE Memory for retail fashion intelligence."""
    print("👗 Retail Fashion Intelligence")
    print("=" * 50)
    
    # Initialize KSE Memory for fashion
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    # Initialize conceptual explorer
    explorer = ConceptualSpaceExplorer(kse)
    
    # Get fashion conceptual space
    fashion_space = await explorer.get_space_data(
        domain="retail_fashion",
        focus_dimensions=["elegance", "comfort", "boldness"],
        max_products=100
    )
    
    print(f"✅ Fashion Space Generated:")
    print(f"   Products: {fashion_space['metadata']['total_products']}")
    print(f"   Dimensions: {len(fashion_space['mapping']['dimensions'])}")
    print(f"   Clusters: {len(fashion_space['clusters'])}")
    print(f"   Focus: {', '.join(fashion_space['metadata']['focus_dimensions'])}")
    
    # Show key insights
    stats = fashion_space['statistics']
    print(f"\n📊 Fashion Intelligence Insights:")
    print(f"   Average Elegance: {stats['dimension_statistics']['elegance']['mean']:.2f}")
    print(f"   Average Comfort: {stats['dimension_statistics']['comfort']['mean']:.2f}")
    print(f"   Average Boldness: {stats['dimension_statistics']['boldness']['mean']:.2f}")
    print(f"   Space Density: {stats['space_density']:.3f}")
    
    await kse.disconnect()
    return fashion_space


async def demonstrate_financial_products():
    """Demonstrate KSE Memory for financial product intelligence."""
    print("\n💰 Financial Product Intelligence")
    print("=" * 50)
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    explorer = ConceptualSpaceExplorer(kse)
    
    # Get financial product space
    finance_space = await explorer.get_space_data(
        domain="finance_products",
        focus_dimensions=["risk_level", "growth_potential", "stability"],
        max_products=50
    )
    
    print(f"✅ Finance Space Generated:")
    print(f"   Products: {finance_space['metadata']['total_products']}")
    print(f"   Risk-Return Matrix: {finance_space['visualization']['type']}")
    print(f"   Clusters: {len(finance_space['clusters'])}")
    
    # Show financial insights
    stats = finance_space['statistics']
    print(f"\n📈 Financial Intelligence Insights:")
    print(f"   Average Risk Level: {stats['dimension_statistics']['risk_level']['mean']:.2f}")
    print(f"   Average Growth Potential: {stats['dimension_statistics']['growth_potential']['mean']:.2f}")
    print(f"   Average Stability: {stats['dimension_statistics']['stability']['mean']:.2f}")
    print(f"   Portfolio Diversification: {stats['dimension_statistics']['diversification']['mean']:.2f}")
    
    await kse.disconnect()
    return finance_space


async def demonstrate_healthcare_devices():
    """Demonstrate KSE Memory for healthcare device intelligence."""
    print("\n🏥 Healthcare Device Intelligence")
    print("=" * 50)
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    explorer = ConceptualSpaceExplorer(kse)
    
    # Get healthcare device space
    healthcare_space = await explorer.get_space_data(
        domain="healthcare_devices",
        focus_dimensions=["precision", "safety", "clinical_efficacy"],
        max_products=30
    )
    
    print(f"✅ Healthcare Space Generated:")
    print(f"   Devices: {healthcare_space['metadata']['total_products']}")
    print(f"   Clinical Matrix: {healthcare_space['visualization']['type']}")
    print(f"   Safety Clusters: {len(healthcare_space['clusters'])}")
    
    # Show healthcare insights
    stats = healthcare_space['statistics']
    print(f"\n🔬 Healthcare Intelligence Insights:")
    print(f"   Average Precision: {stats['dimension_statistics']['precision']['mean']:.2f}")
    print(f"   Average Safety: {stats['dimension_statistics']['safety']['mean']:.2f}")
    print(f"   Average Clinical Efficacy: {stats['dimension_statistics']['clinical_efficacy']['mean']:.2f}")
    print(f"   Regulatory Compliance: {stats['dimension_statistics']['regulatory_approval']['mean']:.2f}")
    
    await kse.disconnect()
    return healthcare_space


async def demonstrate_enterprise_software():
    """Demonstrate KSE Memory for enterprise software intelligence."""
    print("\n💼 Enterprise Software Intelligence")
    print("=" * 50)
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    explorer = ConceptualSpaceExplorer(kse)
    
    # Get enterprise software space
    software_space = await explorer.get_space_data(
        domain="enterprise_software",
        focus_dimensions=["scalability", "security", "performance"],
        max_products=40
    )
    
    print(f"✅ Software Space Generated:")
    print(f"   Solutions: {software_space['metadata']['total_products']}")
    print(f"   Enterprise Matrix: {software_space['visualization']['type']}")
    print(f"   Architecture Clusters: {len(software_space['clusters'])}")
    
    # Show software insights
    stats = software_space['statistics']
    print(f"\n⚡ Software Intelligence Insights:")
    print(f"   Average Scalability: {stats['dimension_statistics']['scalability']['mean']:.2f}")
    print(f"   Average Security: {stats['dimension_statistics']['security']['mean']:.2f}")
    print(f"   Average Performance: {stats['dimension_statistics']['performance']['mean']:.2f}")
    print(f"   Integration Capability: {stats['dimension_statistics']['integration']['mean']:.2f}")
    
    await kse.disconnect()
    return software_space


async def demonstrate_real_estate():
    """Demonstrate KSE Memory for real estate intelligence."""
    print("\n🏠 Real Estate Intelligence")
    print("=" * 50)
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    explorer = ConceptualSpaceExplorer(kse)
    
    # Get real estate space
    real_estate_space = await explorer.get_space_data(
        domain="real_estate",
        focus_dimensions=["location_quality", "value_appreciation", "investment_potential"],
        max_products=60
    )
    
    print(f"✅ Real Estate Space Generated:")
    print(f"   Properties: {real_estate_space['metadata']['total_products']}")
    print(f"   Investment Matrix: {real_estate_space['visualization']['type']}")
    print(f"   Location Clusters: {len(real_estate_space['clusters'])}")
    
    # Show real estate insights
    stats = real_estate_space['statistics']
    print(f"\n🏘️ Real Estate Intelligence Insights:")
    print(f"   Average Location Quality: {stats['dimension_statistics']['location_quality']['mean']:.2f}")
    print(f"   Average Value Appreciation: {stats['dimension_statistics']['value_appreciation']['mean']:.2f}")
    print(f"   Average Investment Potential: {stats['dimension_statistics']['investment_potential']['mean']:.2f}")
    print(f"   Market Liquidity: {stats['dimension_statistics']['market_liquidity']['mean']:.2f}")
    
    await kse.disconnect()
    return real_estate_space


async def demonstrate_cross_domain_analysis():
    """Demonstrate cross-domain analysis capabilities."""
    print("\n🔄 Cross-Domain Intelligence Analysis")
    print("=" * 50)
    
    # Run all domain demonstrations
    fashion_space = await demonstrate_retail_fashion()
    finance_space = await demonstrate_financial_products()
    healthcare_space = await demonstrate_healthcare_devices()
    software_space = await demonstrate_enterprise_software()
    real_estate_space = await demonstrate_real_estate()
    
    # Analyze cross-domain patterns
    print("\n🧠 Cross-Domain Pattern Analysis:")
    
    # Compare space densities
    densities = {
        "Fashion": fashion_space['statistics']['space_density'],
        "Finance": finance_space['statistics']['space_density'],
        "Healthcare": healthcare_space['statistics']['space_density'],
        "Software": software_space['statistics']['space_density'],
        "Real Estate": real_estate_space['statistics']['space_density']
    }
    
    print(f"   Space Density Comparison:")
    for domain, density in sorted(densities.items(), key=lambda x: x[1], reverse=True):
        print(f"     {domain}: {density:.3f}")
    
    # Compare cluster counts
    clusters = {
        "Fashion": len(fashion_space['clusters']),
        "Finance": len(finance_space['clusters']),
        "Healthcare": len(healthcare_space['clusters']),
        "Software": len(software_space['clusters']),
        "Real Estate": len(real_estate_space['clusters'])
    }
    
    print(f"\n   Clustering Patterns:")
    for domain, count in sorted(clusters.items(), key=lambda x: x[1], reverse=True):
        print(f"     {domain}: {count} clusters")
    
    # Analyze dimensional complexity
    print(f"\n   Dimensional Complexity:")
    print(f"     Fashion: 10 dimensions (elegance, comfort, boldness, etc.)")
    print(f"     Finance: 10 dimensions (risk, liquidity, growth, etc.)")
    print(f"     Healthcare: 10 dimensions (precision, safety, efficacy, etc.)")
    print(f"     Software: 10 dimensions (scalability, security, performance, etc.)")
    print(f"     Real Estate: 10 dimensions (location, appreciation, yield, etc.)")


async def demonstrate_adaptive_visualization():
    """Demonstrate how visualizations adapt to different domains."""
    print("\n🎨 Adaptive Visualization Capabilities")
    print("=" * 50)
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    explorer = ConceptualSpaceExplorer(kse)
    
    # Show different visualization types for different domains
    domains = [
        ("retail_fashion", "3d_scatter", "explore"),
        ("finance_products", "risk_return_matrix", "analyze"),
        ("healthcare_devices", "clinical_matrix", "evaluate"),
        ("enterprise_software", "enterprise_matrix", "compare"),
        ("real_estate", "investment_matrix", "invest")
    ]
    
    print("📊 Domain-Specific Visualization Adaptations:")
    
    for domain, viz_type, interaction_mode in domains:
        space_data = await explorer.get_space_data(domain=domain, max_products=10)
        viz_config = space_data['visualization']
        
        print(f"\n   {domain.replace('_', ' ').title()}:")
        print(f"     Visualization Type: {viz_config['type']}")
        print(f"     Interaction Mode: {viz_config['interaction_mode']}")
        print(f"     Primary Axes: {viz_config['axes']['x']['dimension']} × {viz_config['axes']['y']['dimension']} × {viz_config['axes']['z']['dimension']}")
        print(f"     Color Scheme: Domain-optimized")
        print(f"     Controls: {'Enabled' if viz_config['controls']['rotation'] else 'Disabled'}")
    
    await kse.disconnect()


async def demonstrate_business_intelligence_scenarios():
    """Demonstrate real-world business intelligence scenarios."""
    print("\n💼 Business Intelligence Scenarios")
    print("=" * 50)
    
    scenarios = [
        {
            "title": "Fashion Retailer: Seasonal Collection Planning",
            "domain": "retail_fashion",
            "focus": ["seasonality", "modernity", "versatility"],
            "use_case": "Plan seasonal collections based on conceptual trends"
        },
        {
            "title": "Investment Firm: Portfolio Risk Assessment",
            "domain": "finance_products",
            "focus": ["risk_level", "stability", "diversification"],
            "use_case": "Assess portfolio risk across multiple dimensions"
        },
        {
            "title": "Hospital: Medical Equipment Procurement",
            "domain": "healthcare_devices",
            "focus": ["clinical_efficacy", "cost_effectiveness", "safety"],
            "use_case": "Select optimal medical equipment for patient outcomes"
        },
        {
            "title": "Enterprise: Software Vendor Selection",
            "domain": "enterprise_software",
            "focus": ["scalability", "integration", "support_quality"],
            "use_case": "Choose enterprise software based on technical requirements"
        },
        {
            "title": "Real Estate Fund: Investment Opportunity Analysis",
            "domain": "real_estate",
            "focus": ["investment_potential", "location_quality", "market_liquidity"],
            "use_case": "Identify high-potential real estate investments"
        }
    ]
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    explorer = ConceptualSpaceExplorer(kse)
    
    for scenario in scenarios:
        print(f"\n🎯 {scenario['title']}")
        print(f"   Use Case: {scenario['use_case']}")
        print(f"   Focus Dimensions: {', '.join(scenario['focus'])}")
        
        # Get scenario-specific data
        space_data = await explorer.get_space_data(
            domain=scenario['domain'],
            focus_dimensions=scenario['focus'],
            max_products=20
        )
        
        # Show insights
        stats = space_data['statistics']
        focus_stats = {dim: stats['dimension_statistics'][dim]['mean'] 
                      for dim in scenario['focus']}
        
        print(f"   Key Metrics:")
        for dim, value in focus_stats.items():
            print(f"     {dim.replace('_', ' ').title()}: {value:.2f}")
        
        print(f"   Clusters Found: {len(space_data['clusters'])}")
        print(f"   Visualization: {space_data['visualization']['type']}")
    
    await kse.disconnect()


async def main():
    """Run comprehensive multi-domain visualization demonstration."""
    print("🧠 KSE Memory SDK - Multi-Domain Product Intelligence")
    print("=" * 70)
    print("Demonstrating hybrid AI substrate flexibility across industries")
    print("=" * 70)
    
    try:
        # Run all demonstrations
        await demonstrate_cross_domain_analysis()
        await demonstrate_adaptive_visualization()
        await demonstrate_business_intelligence_scenarios()
        
        print("\n" + "=" * 70)
        print("🎉 Multi-Domain Demonstration Complete!")
        print("=" * 70)
        
        print("\n🌟 Key Takeaways:")
        print("✅ KSE Memory adapts to any product domain")
        print("✅ Conceptual dimensions customize to industry needs")
        print("✅ Visualizations optimize for domain-specific insights")
        print("✅ Business intelligence scales across use cases")
        print("✅ Hybrid AI substrate provides universal product intelligence")
        
        print("\n🚀 Ready for Production:")
        print("• Fashion retailers can optimize collections")
        print("• Financial firms can assess portfolio risk")
        print("• Hospitals can select optimal equipment")
        print("• Enterprises can choose best software")
        print("• Real estate funds can identify opportunities")
        
        print("\n💡 Next Steps:")
        print("1. Choose your domain and customize dimensions")
        print("2. Integrate KSE Memory with your product data")
        print("3. Launch visual dashboard for stakeholders")
        print("4. Scale across multiple product categories")
        print("5. Build domain-specific intelligence applications")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        print("This is expected in a mock environment.")


if __name__ == "__main__":
    asyncio.run(main())