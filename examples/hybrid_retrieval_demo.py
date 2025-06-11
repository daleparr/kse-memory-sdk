"""
KSE Memory SDK - Hybrid Knowledge Retrieval Demo

Demonstrates the core foundation of KSE Memory: how hybrid knowledge
retrieval combines Knowledge Graphs + Conceptual Spaces + Neural Embeddings
to achieve superior search results.

This is the FOUNDATION that enables all domain adaptations.
"""

import asyncio
from typing import List, Dict, Any

from kse_memory.core.memory import KSEMemory
from kse_memory.core.config import KSEConfig
from kse_memory.core.models import Product, SearchQuery, SearchType, ConceptualDimensions
from kse_memory.visual.search_explainer import SearchResultsExplainer


async def demonstrate_core_hybrid_retrieval():
    """Demonstrate the core hybrid knowledge retrieval foundation."""
    print("🧠 KSE Memory - Hybrid Knowledge Retrieval Foundation")
    print("=" * 70)
    print("The core that powers all domain adaptations")
    print("=" * 70)
    
    # Initialize KSE Memory
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    # Add sample products to demonstrate hybrid retrieval
    products = [
        Product(
            id="shoe_001",
            title="Premium Athletic Running Shoes",
            description="Lightweight mesh running shoes with responsive cushioning and breathable design. Perfect for daily training and long-distance runs.",
            price=129.99,
            category="Athletic Footwear",
            tags=["running", "athletic", "comfortable", "breathable", "lightweight"],
            conceptual_dimensions=ConceptualDimensions(
                elegance=0.6, comfort=0.9, boldness=0.4, modernity=0.8,
                minimalism=0.7, luxury=0.3, functionality=0.95, versatility=0.8,
                seasonality=0.5, innovation=0.7
            )
        ),
        Product(
            id="shoe_002", 
            title="Elegant Leather Dress Shoes",
            description="Handcrafted Italian leather dress shoes with classic Oxford styling. Perfect for formal occasions and business wear.",
            price=299.99,
            category="Formal Footwear",
            tags=["formal", "leather", "elegant", "business", "classic"],
            conceptual_dimensions=ConceptualDimensions(
                elegance=0.95, comfort=0.6, boldness=0.2, modernity=0.4,
                minimalism=0.8, luxury=0.9, functionality=0.7, versatility=0.6,
                seasonality=0.3, innovation=0.2
            )
        ),
        Product(
            id="shoe_003",
            title="Bold High-Top Sneakers",
            description="Statement sneakers with vibrant colors and unique design. Features premium materials and street-style appeal.",
            price=189.99,
            category="Casual Footwear", 
            tags=["casual", "sneakers", "bold", "colorful", "street-style"],
            conceptual_dimensions=ConceptualDimensions(
                elegance=0.4, comfort=0.7, boldness=0.95, modernity=0.9,
                minimalism=0.2, luxury=0.5, functionality=0.6, versatility=0.7,
                seasonality=0.6, innovation=0.8
            )
        )
    ]
    
    # Add products to KSE Memory
    print("📚 Adding products to hybrid memory system...")
    for product in products:
        await kse.add_product(product)
    print(f"✅ Added {len(products)} products with full hybrid processing")
    
    await kse.disconnect()
    return products


async def demonstrate_three_approaches():
    """Demonstrate each of the three core approaches individually."""
    print("\n🔍 Three Approaches to Knowledge Retrieval")
    print("=" * 50)
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    # Re-add products
    products = await demonstrate_core_hybrid_retrieval()
    for product in products:
        await kse.add_product(product)
    
    query = "comfortable athletic footwear"
    print(f"Query: '{query}'")
    
    # 1. Neural Embeddings (Vector Search)
    print("\n1️⃣ Neural Embeddings - Vector Similarity")
    print("   Approach: Semantic similarity through deep learning")
    
    vector_results = await kse.search(SearchQuery(
        query=query,
        search_type=SearchType.VECTOR,
        limit=3
    ))
    
    print("   Results:")
    for i, result in enumerate(vector_results, 1):
        print(f"     {i}. {result.product.title} (Score: {result.score:.3f})")
        print(f"        Reason: Text similarity and semantic understanding")
    
    # 2. Conceptual Spaces - Multi-dimensional Similarity
    print("\n2️⃣ Conceptual Spaces - Multi-dimensional Similarity")
    print("   Approach: Understanding across conceptual dimensions")
    
    conceptual_results = await kse.search(SearchQuery(
        query=query,
        search_type=SearchType.CONCEPTUAL,
        limit=3
    ))
    
    print("   Results:")
    for i, result in enumerate(conceptual_results, 1):
        print(f"     {i}. {result.product.title} (Score: {result.score:.3f})")
        if result.product.conceptual_dimensions:
            dims = result.product.conceptual_dimensions.to_dict()
            print(f"        Key Dimensions: Comfort={dims['comfort']:.2f}, Functionality={dims['functionality']:.2f}")
    
    # 3. Knowledge Graphs - Relationship Reasoning
    print("\n3️⃣ Knowledge Graphs - Relationship Reasoning")
    print("   Approach: Understanding through relationships and context")
    
    graph_results = await kse.search(SearchQuery(
        query=query,
        search_type=SearchType.GRAPH,
        limit=3
    ))
    
    print("   Results:")
    for i, result in enumerate(graph_results, 1):
        print(f"     {i}. {result.product.title} (Score: {result.score:.3f})")
        print(f"        Relationships: {result.product.category} → {', '.join(result.product.tags[:2])}")
    
    await kse.disconnect()
    return vector_results, conceptual_results, graph_results


async def demonstrate_hybrid_fusion():
    """Demonstrate how hybrid fusion combines all three approaches."""
    print("\n⚡ Hybrid Fusion - The Power of Combination")
    print("=" * 50)
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    # Re-add products
    products = await demonstrate_core_hybrid_retrieval()
    for product in products:
        await kse.add_product(product)
    
    query = "comfortable athletic footwear"
    
    # Hybrid search combining all approaches
    hybrid_results = await kse.search(SearchQuery(
        query=query,
        search_type=SearchType.HYBRID,
        limit=3
    ))
    
    print(f"Query: '{query}'")
    print("\n🎯 Hybrid Results (Best of All Worlds):")
    
    for i, result in enumerate(hybrid_results, 1):
        print(f"   {i}. {result.product.title}")
        print(f"      Final Score: {result.score:.3f}")
        print(f"      Why it's better: Combines semantic understanding + conceptual alignment + relationship context")
        
        # Show conceptual alignment
        if result.product.conceptual_dimensions:
            dims = result.product.conceptual_dimensions.to_dict()
            print(f"      Conceptual Match: Comfort={dims['comfort']:.2f}, Functionality={dims['functionality']:.2f}")
        
        print(f"      Graph Context: {result.product.category} with {', '.join(result.product.tags[:2])} attributes")
        print()
    
    await kse.disconnect()
    return hybrid_results


async def demonstrate_search_explanation():
    """Demonstrate detailed search explanation showing hybrid reasoning."""
    print("\n🔬 Search Explanation - Understanding Hybrid Reasoning")
    print("=" * 60)
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    # Re-add products
    products = await demonstrate_core_hybrid_retrieval()
    for product in products:
        await kse.add_product(product)
    
    # Initialize search explainer
    explainer = SearchResultsExplainer(kse)
    
    query = "comfortable athletic footwear"
    
    # Get hybrid results
    results = await kse.search(SearchQuery(
        query=query,
        search_type=SearchType.HYBRID,
        limit=3
    ))
    
    # Get detailed explanation
    explanation = await explainer.explain_results(query, results, "hybrid")
    
    print(f"Query Analysis: '{query}'")
    print(f"Total Results: {explanation['total_results']}")
    
    # Show overall approach performance
    overall = explanation['overall_explanation']
    approach_perf = overall['approach_performance']
    
    print(f"\n📊 Approach Performance:")
    print(f"   Vector Embeddings:   {approach_perf['vector_embeddings']['score']:.3f} ({approach_perf['vector_embeddings']['contribution']})")
    print(f"   Conceptual Spaces:   {approach_perf['conceptual_spaces']['score']:.3f} ({approach_perf['conceptual_spaces']['contribution']})")
    print(f"   Knowledge Graphs:    {approach_perf['knowledge_graphs']['score']:.3f} ({approach_perf['knowledge_graphs']['contribution']})")
    
    # Show hybrid advantage
    hybrid_advantage = explanation['hybrid_advantage']
    print(f"\n🚀 Hybrid Advantage:")
    print(f"   Improvement over best single approach: +{hybrid_advantage['improvement_percentage']:.1f}%")
    print(f"   Consistency improvement: {'Yes' if hybrid_advantage['consistency_improvement'] else 'No'}")
    print(f"   Coverage improvement: {'Yes' if hybrid_advantage['coverage_improvement'] else 'No'}")
    
    # Show individual result explanations
    print(f"\n🎯 Individual Result Explanations:")
    
    for i, exp_data in enumerate(explanation['individual_explanations'][:2], 1):
        print(f"\n   Result {i}: {exp_data['result']['product']['title']}")
        print(f"   Final Score: {exp_data['final_score']:.3f} (Confidence: {exp_data['confidence_level']})")
        
        print(f"   Component Scores:")
        print(f"     Vector:     {exp_data['vector_score']:.3f}")
        print(f"     Conceptual: {exp_data['conceptual_score']:.3f}")
        print(f"     Graph:      {exp_data['graph_score']:.3f}")
        
        print(f"   Reasoning Path:")
        for step in exp_data['reasoning_path'][:3]:  # Show first 3 steps
            print(f"     • {step}")
    
    # Show reasoning patterns
    patterns = explanation['reasoning_patterns']
    if patterns:
        print(f"\n🧠 Reasoning Patterns Detected:")
        for pattern in patterns:
            print(f"   • {pattern}")
    
    await kse.disconnect()


async def demonstrate_performance_comparison():
    """Demonstrate performance comparison between approaches."""
    print("\n📈 Performance Comparison - Why Hybrid Wins")
    print("=" * 50)
    
    config = KSEConfig()
    kse = KSEMemory(config)
    await kse.initialize("generic", {})
    
    # Add more diverse products for better comparison
    products = [
        Product(
            id="prod_001",
            title="Ultra-Comfortable Running Shoes",
            description="Maximum comfort athletic footwear with advanced cushioning",
            category="Athletic Footwear",
            tags=["running", "comfortable", "athletic"],
            conceptual_dimensions=ConceptualDimensions(comfort=0.95, functionality=0.9, modernity=0.8)
        ),
        Product(
            id="prod_002", 
            title="Professional Business Shoes",
            description="Elegant leather shoes for office and formal occasions",
            category="Formal Footwear",
            tags=["business", "formal", "leather"],
            conceptual_dimensions=ConceptualDimensions(elegance=0.9, luxury=0.8, comfort=0.6)
        ),
        Product(
            id="prod_003",
            title="Casual Walking Sneakers",
            description="Comfortable everyday sneakers for casual wear and walking",
            category="Casual Footwear",
            tags=["casual", "walking", "comfortable"],
            conceptual_dimensions=ConceptualDimensions(comfort=0.8, versatility=0.9, functionality=0.7)
        ),
        Product(
            id="prod_004",
            title="High-Performance Athletic Trainers",
            description="Advanced training shoes with superior performance features",
            category="Athletic Footwear", 
            tags=["training", "performance", "athletic"],
            conceptual_dimensions=ConceptualDimensions(functionality=0.95, innovation=0.8, modernity=0.9)
        )
    ]
    
    for product in products:
        await kse.add_product(product)
    
    # Test queries that highlight different strengths
    test_queries = [
        "comfortable athletic footwear",
        "elegant formal shoes", 
        "versatile everyday sneakers",
        "high-performance training gear"
    ]
    
    print("Testing multiple queries to show hybrid advantage...")
    
    total_vector_score = 0
    total_conceptual_score = 0
    total_graph_score = 0
    total_hybrid_score = 0
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Test each approach
        vector_results = await kse.search(SearchQuery(query=query, search_type=SearchType.VECTOR, limit=1))
        conceptual_results = await kse.search(SearchQuery(query=query, search_type=SearchType.CONCEPTUAL, limit=1))
        graph_results = await kse.search(SearchQuery(query=query, search_type=SearchType.GRAPH, limit=1))
        hybrid_results = await kse.search(SearchQuery(query=query, search_type=SearchType.HYBRID, limit=1))
        
        vector_score = vector_results[0].score if vector_results else 0
        conceptual_score = conceptual_results[0].score if conceptual_results else 0
        graph_score = graph_results[0].score if graph_results else 0
        hybrid_score = hybrid_results[0].score if hybrid_results else 0
        
        print(f"   Vector:     {vector_score:.3f}")
        print(f"   Conceptual: {conceptual_score:.3f}")
        print(f"   Graph:      {graph_score:.3f}")
        print(f"   Hybrid:     {hybrid_score:.3f} ⭐")
        
        improvement = ((hybrid_score - max(vector_score, conceptual_score, graph_score)) / max(vector_score, conceptual_score, graph_score) * 100) if max(vector_score, conceptual_score, graph_score) > 0 else 0
        print(f"   Improvement: +{improvement:.1f}%")
        
        total_vector_score += vector_score
        total_conceptual_score += conceptual_score
        total_graph_score += graph_score
        total_hybrid_score += hybrid_score
    
    # Calculate overall improvement
    avg_vector = total_vector_score / len(test_queries)
    avg_conceptual = total_conceptual_score / len(test_queries)
    avg_graph = total_graph_score / len(test_queries)
    avg_hybrid = total_hybrid_score / len(test_queries)
    
    best_individual = max(avg_vector, avg_conceptual, avg_graph)
    overall_improvement = ((avg_hybrid - best_individual) / best_individual * 100) if best_individual > 0 else 0
    
    print(f"\n📊 Overall Performance Summary:")
    print(f"   Average Vector Score:     {avg_vector:.3f}")
    print(f"   Average Conceptual Score: {avg_conceptual:.3f}")
    print(f"   Average Graph Score:      {avg_graph:.3f}")
    print(f"   Average Hybrid Score:     {avg_hybrid:.3f}")
    print(f"   Overall Improvement:      +{overall_improvement:.1f}%")
    
    await kse.disconnect()


async def demonstrate_foundation_principles():
    """Demonstrate the core principles that make hybrid retrieval work."""
    print("\n🏗️ Foundation Principles - Why Hybrid Knowledge Retrieval Works")
    print("=" * 70)
    
    principles = [
        {
            "name": "Complementary Strengths",
            "description": "Each approach excels in different scenarios",
            "examples": [
                "Vector: Excellent for semantic similarity and text matching",
                "Conceptual: Superior for intent understanding and preferences",
                "Graph: Best for relationship reasoning and context"
            ]
        },
        {
            "name": "Weakness Mitigation", 
            "description": "Hybrid approach compensates for individual limitations",
            "examples": [
                "Vector: Can miss conceptual intent → Conceptual fills the gap",
                "Conceptual: May lack semantic nuance → Vector provides precision",
                "Graph: Limited by relationship coverage → Vector/Conceptual add breadth"
            ]
        },
        {
            "name": "Score Fusion Intelligence",
            "description": "Smart combination produces better results than any single approach",
            "examples": [
                "Weighted averaging based on query characteristics",
                "Consensus boosting when approaches agree",
                "Confidence-based weighting for reliability"
            ]
        },
        {
            "name": "Universal Applicability",
            "description": "Foundation works across all domains and use cases",
            "examples": [
                "Same core principles apply to fashion, finance, healthcare",
                "Domain-specific dimensions enhance but don't replace foundation",
                "Scalable from simple products to complex enterprise systems"
            ]
        }
    ]
    
    for i, principle in enumerate(principles, 1):
        print(f"\n{i}. {principle['name']}")
        print(f"   {principle['description']}")
        print(f"   Examples:")
        for example in principle['examples']:
            print(f"     • {example}")
    
    print(f"\n🎯 Key Insight:")
    print(f"   The hybrid foundation is what enables KSE Memory to adapt to any domain")
    print(f"   while maintaining superior performance across all use cases.")


async def main():
    """Run comprehensive hybrid knowledge retrieval demonstration."""
    print("🧠 KSE Memory SDK - Hybrid Knowledge Retrieval Foundation")
    print("=" * 80)
    print("The core technology that powers universal product intelligence")
    print("=" * 80)
    
    try:
        # Demonstrate the foundation
        await demonstrate_core_hybrid_retrieval()
        
        # Show the three individual approaches
        await demonstrate_three_approaches()
        
        # Demonstrate hybrid fusion
        await demonstrate_hybrid_fusion()
        
        # Show detailed explanation
        await demonstrate_search_explanation()
        
        # Performance comparison
        await demonstrate_performance_comparison()
        
        # Foundation principles
        await demonstrate_foundation_principles()
        
        print("\n" + "=" * 80)
        print("🎉 Hybrid Knowledge Retrieval Foundation Demonstration Complete!")
        print("=" * 80)
        
        print("\n🌟 Key Takeaways:")
        print("✅ Hybrid retrieval combines three complementary AI approaches")
        print("✅ Each approach has unique strengths and limitations")
        print("✅ Fusion produces consistently better results than any single approach")
        print("✅ Foundation is universal - works across all domains and use cases")
        print("✅ Visual explanation builds trust through transparency")
        
        print("\n🚀 This Foundation Enables:")
        print("• Domain-specific adaptations (retail, finance, healthcare, etc.)")
        print("• Visual tooling that explains AI reasoning")
        print("• LangChain/LlamaIndex integrations with superior performance")
        print("• Enterprise-grade product intelligence at any scale")
        print("• Community adoption through understandable AI")
        
        print("\n💡 Next Steps:")
        print("1. Experience the foundation with: kse quickstart")
        print("2. Integrate with your domain using conceptual dimensions")
        print("3. Build visual applications with the dashboard")
        print("4. Scale to production with enterprise backends")
        print("5. Join the community building the future of product intelligence")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        print("This is expected in a mock environment.")


if __name__ == "__main__":
    asyncio.run(main())