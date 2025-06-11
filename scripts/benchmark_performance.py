"""
KSE Memory SDK - Performance Benchmark Script

Validates the 18%+ performance improvement claims.
"""

import asyncio
import time
import statistics
import json
from typing import List, Dict, Any
from datetime import datetime

from kse_memory.core.memory import KSEMemory
from kse_memory.core.config import KSEConfig
from kse_memory.core.models import Product, SearchQuery, SearchType, ConceptualDimensions
from kse_memory.quickstart.datasets import SampleDatasets


class PerformanceBenchmark:
    """
    Comprehensive performance benchmark for KSE Memory.
    
    Validates the 18%+ improvement claim and measures:
    - Search relevance scores
    - Query latency
    - Memory usage
    - Consistency across queries
    """
    
    def __init__(self):
        """Initialize benchmark."""
        self.results = {}
        self.datasets = SampleDatasets()
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        print("🚀 KSE Memory SDK - Performance Benchmark")
        print("=" * 60)
        print("Validating 18%+ improvement over vector-only search")
        print("=" * 60)
        
        # Initialize KSE Memory
        config = KSEConfig(
            debug=False,
            vector_store={"backend": "memory"},
            graph_store={"backend": "memory"},
            concept_store={"backend": "memory"}
        )
        
        kse = KSEMemory(config)
        await kse.initialize("generic", {})
        
        try:
            # Load test datasets
            print("\n📚 Loading test datasets...")
            products = await self._load_test_products()
            
            # Add products to memory
            print(f"📥 Adding {len(products)} products to memory...")
            start_time = time.time()
            for product in products:
                await kse.add_product(product)
            load_time = time.time() - start_time
            print(f"✅ Products loaded in {load_time:.2f} seconds")
            
            # Run benchmarks
            print("\n🔍 Running search benchmarks...")
            
            # 1. Relevance benchmark
            relevance_results = await self._benchmark_relevance(kse)
            
            # 2. Latency benchmark
            latency_results = await self._benchmark_latency(kse)
            
            # 3. Consistency benchmark
            consistency_results = await self._benchmark_consistency(kse)
            
            # 4. Scalability benchmark
            scalability_results = await self._benchmark_scalability(kse)
            
            # Compile results
            benchmark_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "dataset_size": len(products),
                "load_time_seconds": load_time,
                "relevance": relevance_results,
                "latency": latency_results,
                "consistency": consistency_results,
                "scalability": scalability_results,
                "summary": self._generate_summary(relevance_results, latency_results, consistency_results)
            }
            
            # Display results
            self._display_results(benchmark_results)
            
            return benchmark_results
            
        finally:
            await kse.disconnect()
    
    async def _load_test_products(self) -> List[Product]:
        """Load comprehensive test dataset."""
        products = []
        
        # Load products from multiple domains
        retail_products = self.datasets.get_retail_products()[:50]
        finance_products = self.datasets.get_finance_products()[:30]
        healthcare_products = self.datasets.get_healthcare_products()[:20]
        
        products.extend(retail_products)
        products.extend(finance_products)
        products.extend(healthcare_products)
        
        print(f"✅ Loaded {len(products)} products across {3} domains")
        return products
    
    async def _benchmark_relevance(self, kse: KSEMemory) -> Dict[str, Any]:
        """Benchmark search relevance across different approaches."""
        print("\n📊 Benchmarking search relevance...")
        
        # Test queries across domains
        test_queries = [
            "comfortable athletic footwear",
            "elegant formal attire",
            "minimalist design items",
            "high-performance technology",
            "sustainable materials",
            "luxury premium products",
            "innovative solutions",
            "cost-effective options",
            "professional equipment",
            "versatile everyday items"
        ]
        
        results = {
            "vector_scores": [],
            "conceptual_scores": [],
            "graph_scores": [],
            "hybrid_scores": [],
            "query_results": []
        }
        
        for query in test_queries:
            print(f"  Testing: '{query}'")
            
            query_results = {}
            
            # Vector search
            vector_results = await kse.search(SearchQuery(
                query=query, search_type=SearchType.VECTOR, limit=5
            ))
            vector_score = sum(r.score for r in vector_results) / len(vector_results) if vector_results else 0
            results["vector_scores"].append(vector_score)
            query_results["vector"] = {"score": vector_score, "count": len(vector_results)}
            
            # Conceptual search
            conceptual_results = await kse.search(SearchQuery(
                query=query, search_type=SearchType.CONCEPTUAL, limit=5
            ))
            conceptual_score = sum(r.score for r in conceptual_results) / len(conceptual_results) if conceptual_results else 0
            results["conceptual_scores"].append(conceptual_score)
            query_results["conceptual"] = {"score": conceptual_score, "count": len(conceptual_results)}
            
            # Graph search
            graph_results = await kse.search(SearchQuery(
                query=query, search_type=SearchType.GRAPH, limit=5
            ))
            graph_score = sum(r.score for r in graph_results) / len(graph_results) if graph_results else 0
            results["graph_scores"].append(graph_score)
            query_results["graph"] = {"score": graph_score, "count": len(graph_results)}
            
            # Hybrid search
            hybrid_results = await kse.search(SearchQuery(
                query=query, search_type=SearchType.HYBRID, limit=5
            ))
            hybrid_score = sum(r.score for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0
            results["hybrid_scores"].append(hybrid_score)
            query_results["hybrid"] = {"score": hybrid_score, "count": len(hybrid_results)}
            
            query_results["query"] = query
            results["query_results"].append(query_results)
        
        # Calculate averages and improvements
        avg_vector = statistics.mean(results["vector_scores"]) if results["vector_scores"] else 0
        avg_conceptual = statistics.mean(results["conceptual_scores"]) if results["conceptual_scores"] else 0
        avg_graph = statistics.mean(results["graph_scores"]) if results["graph_scores"] else 0
        avg_hybrid = statistics.mean(results["hybrid_scores"]) if results["hybrid_scores"] else 0
        
        best_individual = max(avg_vector, avg_conceptual, avg_graph)
        improvement = ((avg_hybrid - best_individual) / best_individual * 100) if best_individual > 0 else 0
        
        results["averages"] = {
            "vector": avg_vector,
            "conceptual": avg_conceptual,
            "graph": avg_graph,
            "hybrid": avg_hybrid
        }
        
        results["improvement_percentage"] = improvement
        results["improvement_target_met"] = improvement >= 18.0
        
        print(f"  ✅ Relevance benchmark complete")
        print(f"     Hybrid improvement: +{improvement:.1f}% (Target: +18%)")
        
        return results
    
    async def _benchmark_latency(self, kse: KSEMemory) -> Dict[str, Any]:
        """Benchmark query latency across approaches."""
        print("\n⚡ Benchmarking query latency...")
        
        test_query = "comfortable athletic products"
        iterations = 10
        
        results = {
            "vector_latencies": [],
            "conceptual_latencies": [],
            "graph_latencies": [],
            "hybrid_latencies": []
        }
        
        # Warm up
        await kse.search(SearchQuery(query=test_query, search_type=SearchType.HYBRID, limit=5))
        
        for i in range(iterations):
            # Vector latency
            start_time = time.perf_counter()
            await kse.search(SearchQuery(query=test_query, search_type=SearchType.VECTOR, limit=5))
            vector_latency = (time.perf_counter() - start_time) * 1000
            results["vector_latencies"].append(vector_latency)
            
            # Conceptual latency
            start_time = time.perf_counter()
            await kse.search(SearchQuery(query=test_query, search_type=SearchType.CONCEPTUAL, limit=5))
            conceptual_latency = (time.perf_counter() - start_time) * 1000
            results["conceptual_latencies"].append(conceptual_latency)
            
            # Graph latency
            start_time = time.perf_counter()
            await kse.search(SearchQuery(query=test_query, search_type=SearchType.GRAPH, limit=5))
            graph_latency = (time.perf_counter() - start_time) * 1000
            results["graph_latencies"].append(graph_latency)
            
            # Hybrid latency
            start_time = time.perf_counter()
            await kse.search(SearchQuery(query=test_query, search_type=SearchType.HYBRID, limit=5))
            hybrid_latency = (time.perf_counter() - start_time) * 1000
            results["hybrid_latencies"].append(hybrid_latency)
        
        # Calculate statistics
        results["averages"] = {
            "vector": statistics.mean(results["vector_latencies"]),
            "conceptual": statistics.mean(results["conceptual_latencies"]),
            "graph": statistics.mean(results["graph_latencies"]),
            "hybrid": statistics.mean(results["hybrid_latencies"])
        }
        
        results["target_met"] = results["averages"]["hybrid"] < 100  # Target: sub-100ms
        
        print(f"  ✅ Latency benchmark complete")
        print(f"     Hybrid latency: {results['averages']['hybrid']:.1f}ms (Target: <100ms)")
        
        return results
    
    async def _benchmark_consistency(self, kse: KSEMemory) -> Dict[str, Any]:
        """Benchmark consistency across different query types."""
        print("\n🎯 Benchmarking result consistency...")
        
        # Different query types
        query_types = [
            ("specific", "premium running shoes"),
            ("abstract", "comfortable elegant items"),
            ("category", "athletic footwear"),
            ("attribute", "sustainable materials"),
            ("brand", "luxury premium products")
        ]
        
        results = {
            "consistency_scores": [],
            "query_type_results": []
        }
        
        for query_type, query in query_types:
            # Run multiple iterations
            scores = []
            for _ in range(5):
                hybrid_results = await kse.search(SearchQuery(
                    query=query, search_type=SearchType.HYBRID, limit=5
                ))
                if hybrid_results:
                    avg_score = sum(r.score for r in hybrid_results) / len(hybrid_results)
                    scores.append(avg_score)
            
            if scores:
                consistency = 1 - (statistics.stdev(scores) / statistics.mean(scores)) if statistics.mean(scores) > 0 else 0
                results["consistency_scores"].append(consistency)
                results["query_type_results"].append({
                    "type": query_type,
                    "query": query,
                    "consistency": consistency,
                    "avg_score": statistics.mean(scores),
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
                })
        
        results["overall_consistency"] = statistics.mean(results["consistency_scores"]) if results["consistency_scores"] else 0
        results["target_met"] = results["overall_consistency"] > 0.8  # Target: >80% consistency
        
        print(f"  ✅ Consistency benchmark complete")
        print(f"     Overall consistency: {results['overall_consistency']:.1%} (Target: >80%)")
        
        return results
    
    async def _benchmark_scalability(self, kse: KSEMemory) -> Dict[str, Any]:
        """Benchmark scalability with different dataset sizes."""
        print("\n📈 Benchmarking scalability...")
        
        # Test with different result limits
        limits = [1, 5, 10, 20, 50]
        query = "comfortable products"
        
        results = {
            "scalability_data": [],
            "latency_growth": "linear"  # Simplified for demo
        }
        
        for limit in limits:
            start_time = time.perf_counter()
            search_results = await kse.search(SearchQuery(
                query=query, search_type=SearchType.HYBRID, limit=limit
            ))
            latency = (time.perf_counter() - start_time) * 1000
            
            results["scalability_data"].append({
                "limit": limit,
                "latency_ms": latency,
                "results_count": len(search_results),
                "avg_score": sum(r.score for r in search_results) / len(search_results) if search_results else 0
            })
        
        # Check if latency grows sub-linearly
        latencies = [d["latency_ms"] for d in results["scalability_data"]]
        if len(latencies) >= 2:
            growth_rate = latencies[-1] / latencies[0] if latencies[0] > 0 else 1
            results["growth_rate"] = growth_rate
            results["target_met"] = growth_rate < len(limits)  # Sub-linear growth
        
        print(f"  ✅ Scalability benchmark complete")
        
        return results
    
    def _generate_summary(self, relevance: Dict, latency: Dict, consistency: Dict) -> Dict[str, Any]:
        """Generate benchmark summary."""
        return {
            "improvement_percentage": relevance.get("improvement_percentage", 0),
            "improvement_target_met": relevance.get("improvement_target_met", False),
            "avg_latency_ms": latency.get("averages", {}).get("hybrid", 0),
            "latency_target_met": latency.get("target_met", False),
            "consistency_score": consistency.get("overall_consistency", 0),
            "consistency_target_met": consistency.get("target_met", False),
            "overall_grade": self._calculate_grade(relevance, latency, consistency)
        }
    
    def _calculate_grade(self, relevance: Dict, latency: Dict, consistency: Dict) -> str:
        """Calculate overall performance grade."""
        targets_met = 0
        total_targets = 3
        
        if relevance.get("improvement_target_met", False):
            targets_met += 1
        if latency.get("target_met", False):
            targets_met += 1
        if consistency.get("target_met", False):
            targets_met += 1
        
        if targets_met == total_targets:
            return "A+ (Excellent)"
        elif targets_met >= 2:
            return "A (Good)"
        elif targets_met >= 1:
            return "B (Acceptable)"
        else:
            return "C (Needs Improvement)"
    
    def _display_results(self, results: Dict[str, Any]):
        """Display benchmark results."""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        summary = results["summary"]
        
        # Performance improvement
        improvement = summary["improvement_percentage"]
        improvement_status = "✅ PASS" if summary["improvement_target_met"] else "❌ FAIL"
        print(f"🎯 Performance Improvement: +{improvement:.1f}% {improvement_status}")
        print(f"   Target: +18% or better")
        
        # Latency
        latency = summary["avg_latency_ms"]
        latency_status = "✅ PASS" if summary["latency_target_met"] else "❌ FAIL"
        print(f"⚡ Average Latency: {latency:.1f}ms {latency_status}")
        print(f"   Target: <100ms")
        
        # Consistency
        consistency = summary["consistency_score"]
        consistency_status = "✅ PASS" if summary["consistency_target_met"] else "❌ FAIL"
        print(f"🎯 Result Consistency: {consistency:.1%} {consistency_status}")
        print(f"   Target: >80%")
        
        # Overall grade
        grade = summary["overall_grade"]
        print(f"\n🏆 Overall Grade: {grade}")
        
        # Detailed breakdown
        print(f"\n📈 Detailed Performance Breakdown:")
        relevance = results["relevance"]["averages"]
        print(f"   Vector Search:     {relevance['vector']:.3f}")
        print(f"   Conceptual Search: {relevance['conceptual']:.3f}")
        print(f"   Graph Search:      {relevance['graph']:.3f}")
        print(f"   Hybrid Search:     {relevance['hybrid']:.3f} ⭐")
        
        latency_data = results["latency"]["averages"]
        print(f"\n⚡ Latency Breakdown:")
        print(f"   Vector Search:     {latency_data['vector']:.1f}ms")
        print(f"   Conceptual Search: {latency_data['conceptual']:.1f}ms")
        print(f"   Graph Search:      {latency_data['graph']:.1f}ms")
        print(f"   Hybrid Search:     {latency_data['hybrid']:.1f}ms ⭐")
        
        print(f"\n📊 Dataset Information:")
        print(f"   Total Products: {results['dataset_size']}")
        print(f"   Load Time: {results['load_time_seconds']:.2f}s")
        print(f"   Benchmark Time: {results['timestamp']}")


async def main():
    """Run performance benchmark."""
    benchmark = PerformanceBenchmark()
    
    try:
        results = await benchmark.run_comprehensive_benchmark()
        
        # Save results to file
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: benchmark_results.json")
        
        # Return success if all targets met
        summary = results["summary"]
        if (summary["improvement_target_met"] and 
            summary["latency_target_met"] and 
            summary["consistency_target_met"]):
            print("\n🎉 All performance targets MET! Ready for launch.")
            return True
        else:
            print("\n⚠️  Some performance targets not met. Review needed.")
            return False
            
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1)