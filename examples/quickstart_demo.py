"""
KSE Memory SDK - Quickstart Demo Example

This example demonstrates the zero-configuration quickstart
experience that showcases hybrid AI capabilities.
"""

import asyncio
from kse_memory.quickstart.demo import QuickstartDemo


async def main():
    """Run the quickstart demo."""
    print("🚀 KSE Memory SDK - Quickstart Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = QuickstartDemo()
    
    try:
        # Run retail demo
        print("\n📱 Running Retail Demo...")
        retail_results = await demo.run(
            demo_type="retail",
            open_browser=False  # Skip browser for this example
        )
        
        print(f"✅ Retail demo completed!")
        print(f"   Products loaded: {retail_results['products_loaded']}")
        print(f"   Search queries: {len(retail_results['search_results'])}")
        print(f"   Performance improvement: +{retail_results['benchmark_results']['improvement_percentage']:.1f}%")
        
        # Run finance demo
        print("\n💰 Running Finance Demo...")
        finance_results = await demo.run(
            demo_type="finance",
            open_browser=False
        )
        
        print(f"✅ Finance demo completed!")
        print(f"   Products loaded: {finance_results['products_loaded']}")
        print(f"   Performance improvement: +{finance_results['benchmark_results']['improvement_percentage']:.1f}%")
        
        # Run healthcare demo
        print("\n🏥 Running Healthcare Demo...")
        healthcare_results = await demo.run(
            demo_type="healthcare",
            open_browser=False
        )
        
        print(f"✅ Healthcare demo completed!")
        print(f"   Products loaded: {healthcare_results['products_loaded']}")
        print(f"   Performance improvement: +{healthcare_results['benchmark_results']['improvement_percentage']:.1f}%")
        
        print("\n🎉 All demos completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("• Hybrid AI combines knowledge graphs + conceptual spaces + embeddings")
        print("• Better relevance through multi-dimensional similarity")
        print("• Fast performance with sub-100ms query times")
        print("• Zero configuration - works out of the box")
        print("• Multi-domain support (retail, finance, healthcare)")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())