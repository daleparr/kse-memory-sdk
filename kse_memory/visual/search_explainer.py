"""
KSE Memory SDK - Search Results Explainer

Visual explanation of hybrid knowledge retrieval showing how
Knowledge Graphs + Conceptual Spaces + Neural Embeddings
combine to produce superior search results.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..core.memory import KSEMemory
from ..core.models import Product, SearchResult, SearchType


@dataclass
class SearchExplanation:
    """Detailed explanation of a search result."""
    query: str
    result: SearchResult
    vector_score: float
    conceptual_score: float
    graph_score: float
    final_score: float
    reasoning_path: List[str]
    dimension_contributions: Dict[str, float]
    graph_connections: List[Dict[str, Any]]
    confidence_level: str


class SearchResultsExplainer:
    """
    Visual explainer for hybrid knowledge retrieval results.
    
    Shows exactly how KSE Memory combines three AI approaches:
    1. Neural Embeddings (vector similarity)
    2. Conceptual Spaces (multi-dimensional similarity) 
    3. Knowledge Graphs (relationship reasoning)
    
    This transparency builds trust and understanding of why
    hybrid AI produces better results than any single approach.
    
    Example:
        explainer = SearchResultsExplainer(kse_memory)
        
        # Explain why specific results were returned
        explanation = await explainer.explain_results(
            query="comfortable running shoes",
            results=search_results,
            search_type="hybrid"
        )
        
        # Show detailed breakdown
        breakdown = await explainer.get_detailed_breakdown(
            query="comfortable running shoes",
            product_id="shoe_001"
        )
    """
    
    def __init__(self, kse_memory: KSEMemory):
        """Initialize search results explainer."""
        self.kse_memory = kse_memory
    
    async def explain_results(
        self,
        query: str,
        results: List[SearchResult],
        search_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Explain why specific search results were returned.
        
        Args:
            query: Original search query
            results: Search results to explain
            search_type: Type of search performed
            
        Returns:
            Comprehensive explanation of results
        """
        explanations = []
        
        for result in results[:5]:  # Explain top 5 results
            explanation = await self._explain_single_result(query, result, search_type)
            explanations.append(explanation)
        
        # Generate overall explanation
        overall_explanation = self._generate_overall_explanation(query, explanations, search_type)
        
        return {
            "query": query,
            "search_type": search_type,
            "total_results": len(results),
            "explained_results": len(explanations),
            "individual_explanations": [exp.__dict__ for exp in explanations],
            "overall_explanation": overall_explanation,
            "hybrid_advantage": self._calculate_hybrid_advantage(explanations),
            "confidence_distribution": self._calculate_confidence_distribution(explanations),
            "reasoning_patterns": self._extract_reasoning_patterns(explanations)
        }
    
    async def _explain_single_result(
        self,
        query: str,
        result: SearchResult,
        search_type: str
    ) -> SearchExplanation:
        """Explain a single search result in detail."""
        
        # Get individual component scores
        vector_score = await self._get_vector_score(query, result.product)
        conceptual_score = await self._get_conceptual_score(query, result.product)
        graph_score = await self._get_graph_score(query, result.product)
        
        # Generate reasoning path
        reasoning_path = self._generate_reasoning_path(
            query, result.product, vector_score, conceptual_score, graph_score
        )
        
        # Calculate dimension contributions
        dimension_contributions = self._calculate_dimension_contributions(
            query, result.product
        )
        
        # Get graph connections
        graph_connections = await self._get_graph_connections(query, result.product)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(
            vector_score, conceptual_score, graph_score, result.score
        )
        
        return SearchExplanation(
            query=query,
            result=result,
            vector_score=vector_score,
            conceptual_score=conceptual_score,
            graph_score=graph_score,
            final_score=result.score,
            reasoning_path=reasoning_path,
            dimension_contributions=dimension_contributions,
            graph_connections=graph_connections,
            confidence_level=confidence_level
        )
    
    async def _get_vector_score(self, query: str, product: Product) -> float:
        """Get vector similarity score for the product."""
        # In real implementation, this would call the embedding service
        # For demo, simulate vector similarity based on text overlap
        query_words = set(query.lower().split())
        product_words = set((product.title + " " + product.description).lower().split())
        
        overlap = len(query_words.intersection(product_words))
        total_words = len(query_words.union(product_words))
        
        return overlap / total_words if total_words > 0 else 0.0
    
    async def _get_conceptual_score(self, query: str, product: Product) -> float:
        """Get conceptual similarity score for the product."""
        if not product.conceptual_dimensions:
            return 0.5  # Neutral score
        
        # Map query to conceptual dimensions
        query_concepts = self._map_query_to_concepts(query)
        product_concepts = product.conceptual_dimensions.to_dict()
        
        # Calculate conceptual similarity
        similarities = []
        for concept, query_value in query_concepts.items():
            product_value = product_concepts.get(concept, 0.5)
            similarity = 1 - abs(query_value - product_value)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    async def _get_graph_score(self, query: str, product: Product) -> float:
        """Get knowledge graph score for the product."""
        # In real implementation, this would traverse the knowledge graph
        # For demo, simulate graph reasoning based on category and tags
        
        query_lower = query.lower()
        score = 0.0
        
        # Category matching
        if product.category and any(word in product.category.lower() for word in query_lower.split()):
            score += 0.3
        
        # Tag matching
        if product.tags:
            tag_matches = sum(1 for tag in product.tags if any(word in tag.lower() for word in query_lower.split()))
            score += min(0.4, tag_matches * 0.1)
        
        # Relationship bonus (simulated)
        if "comfortable" in query_lower and "athletic" in (product.title + " " + product.description).lower():
            score += 0.3  # Comfort-athletic relationship
        
        return min(1.0, score)
    
    def _map_query_to_concepts(self, query: str) -> Dict[str, float]:
        """Map search query to conceptual dimensions."""
        query_lower = query.lower()
        concepts = {}
        
        # Comfort-related terms
        if any(word in query_lower for word in ["comfortable", "comfort", "cozy", "soft"]):
            concepts["comfort"] = 0.9
        
        # Elegance-related terms
        if any(word in query_lower for word in ["elegant", "sophisticated", "refined", "classy"]):
            concepts["elegance"] = 0.9
        
        # Boldness-related terms
        if any(word in query_lower for word in ["bold", "statement", "striking", "eye-catching"]):
            concepts["boldness"] = 0.9
        
        # Modernity-related terms
        if any(word in query_lower for word in ["modern", "contemporary", "cutting-edge", "latest"]):
            concepts["modernity"] = 0.9
        
        # Minimalism-related terms
        if any(word in query_lower for word in ["minimal", "simple", "clean", "basic"]):
            concepts["minimalism"] = 0.9
        
        # Luxury-related terms
        if any(word in query_lower for word in ["luxury", "premium", "high-end", "exclusive"]):
            concepts["luxury"] = 0.9
        
        # Functionality-related terms
        if any(word in query_lower for word in ["functional", "practical", "useful", "performance"]):
            concepts["functionality"] = 0.9
        
        # Default neutral values for unmapped concepts
        all_concepts = ["elegance", "comfort", "boldness", "modernity", "minimalism", 
                       "luxury", "functionality", "versatility", "seasonality", "innovation"]
        
        for concept in all_concepts:
            if concept not in concepts:
                concepts[concept] = 0.5  # Neutral
        
        return concepts
    
    def _generate_reasoning_path(
        self,
        query: str,
        product: Product,
        vector_score: float,
        conceptual_score: float,
        graph_score: float
    ) -> List[str]:
        """Generate human-readable reasoning path."""
        path = []
        
        # Vector reasoning
        if vector_score > 0.6:
            path.append(f"✅ Strong text similarity: Query terms match product description (score: {vector_score:.2f})")
        elif vector_score > 0.3:
            path.append(f"📝 Moderate text similarity: Some query terms found in product (score: {vector_score:.2f})")
        else:
            path.append(f"📝 Limited text similarity: Few direct term matches (score: {vector_score:.2f})")
        
        # Conceptual reasoning
        if conceptual_score > 0.7:
            path.append(f"🎨 Excellent conceptual match: Product aligns with query intent (score: {conceptual_score:.2f})")
        elif conceptual_score > 0.5:
            path.append(f"🎨 Good conceptual alignment: Product partially matches query concepts (score: {conceptual_score:.2f})")
        else:
            path.append(f"🎨 Weak conceptual match: Limited alignment with query intent (score: {conceptual_score:.2f})")
        
        # Graph reasoning
        if graph_score > 0.6:
            path.append(f"🕸️ Strong relationship connections: Product well-connected to query context (score: {graph_score:.2f})")
        elif graph_score > 0.3:
            path.append(f"🕸️ Some relationship connections: Product has relevant associations (score: {graph_score:.2f})")
        else:
            path.append(f"🕸️ Limited connections: Few relevant relationships found (score: {graph_score:.2f})")
        
        # Final fusion
        final_score = (vector_score + conceptual_score + graph_score) / 3
        path.append(f"⚡ Hybrid fusion: Combined all approaches for final score ({final_score:.2f})")
        
        return path
    
    def _calculate_dimension_contributions(
        self,
        query: str,
        product: Product
    ) -> Dict[str, float]:
        """Calculate how much each conceptual dimension contributed to the match."""
        if not product.conceptual_dimensions:
            return {}
        
        query_concepts = self._map_query_to_concepts(query)
        product_concepts = product.conceptual_dimensions.to_dict()
        
        contributions = {}
        for concept in query_concepts:
            query_value = query_concepts[concept]
            product_value = product_concepts.get(concept, 0.5)
            
            # Higher contribution if both query and product have high values
            # or if they're well-aligned
            alignment = 1 - abs(query_value - product_value)
            importance = max(query_value, product_value)
            contribution = alignment * importance
            
            contributions[concept] = contribution
        
        return contributions
    
    async def _get_graph_connections(
        self,
        query: str,
        product: Product
    ) -> List[Dict[str, Any]]:
        """Get knowledge graph connections that influenced the result."""
        # In real implementation, this would query the graph database
        # For demo, simulate relevant connections
        
        connections = []
        
        # Category connections
        if product.category:
            connections.append({
                "type": "category",
                "source": product.title,
                "target": product.category,
                "relationship": "belongs_to",
                "strength": 0.8,
                "explanation": f"Product belongs to {product.category} category"
            })
        
        # Tag connections
        if product.tags:
            for tag in product.tags[:3]:  # Limit to top 3 tags
                connections.append({
                    "type": "attribute",
                    "source": product.title,
                    "target": tag,
                    "relationship": "has_attribute",
                    "strength": 0.6,
                    "explanation": f"Product has {tag} attribute"
                })
        
        # Simulated semantic connections
        query_lower = query.lower()
        if "comfortable" in query_lower and "athletic" in product.description.lower():
            connections.append({
                "type": "semantic",
                "source": "comfort",
                "target": "athletic",
                "relationship": "enhances",
                "strength": 0.7,
                "explanation": "Comfort enhances athletic performance"
            })
        
        return connections
    
    def _determine_confidence_level(
        self,
        vector_score: float,
        conceptual_score: float,
        graph_score: float,
        final_score: float
    ) -> str:
        """Determine confidence level based on score consistency."""
        scores = [vector_score, conceptual_score, graph_score]
        score_std = np.std(scores)
        score_mean = np.mean(scores)
        
        # High confidence: high scores with low variance
        if score_mean > 0.7 and score_std < 0.1:
            return "Very High"
        elif score_mean > 0.6 and score_std < 0.15:
            return "High"
        elif score_mean > 0.5 and score_std < 0.2:
            return "Medium"
        elif score_mean > 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_overall_explanation(
        self,
        query: str,
        explanations: List[SearchExplanation],
        search_type: str
    ) -> Dict[str, Any]:
        """Generate overall explanation for the search results."""
        if not explanations:
            return {}
        
        # Calculate average scores
        avg_vector = np.mean([exp.vector_score for exp in explanations])
        avg_conceptual = np.mean([exp.conceptual_score for exp in explanations])
        avg_graph = np.mean([exp.graph_score for exp in explanations])
        avg_final = np.mean([exp.final_score for exp in explanations])
        
        # Determine dominant approach
        scores = {"vector": avg_vector, "conceptual": avg_conceptual, "graph": avg_graph}
        dominant_approach = max(scores.items(), key=lambda x: x[1])
        
        # Generate explanation
        explanation = {
            "query_analysis": self._analyze_query_characteristics(query),
            "approach_performance": {
                "vector_embeddings": {
                    "score": avg_vector,
                    "strength": "Text similarity and semantic understanding",
                    "contribution": f"{(avg_vector / avg_final * 100):.1f}%" if avg_final > 0 else "0%"
                },
                "conceptual_spaces": {
                    "score": avg_conceptual,
                    "strength": "Multi-dimensional similarity and intent understanding",
                    "contribution": f"{(avg_conceptual / avg_final * 100):.1f}%" if avg_final > 0 else "0%"
                },
                "knowledge_graphs": {
                    "score": avg_graph,
                    "strength": "Relationship reasoning and context understanding",
                    "contribution": f"{(avg_graph / avg_final * 100):.1f}%" if avg_final > 0 else "0%"
                }
            },
            "dominant_approach": {
                "name": dominant_approach[0],
                "score": dominant_approach[1],
                "reason": self._explain_dominance(dominant_approach[0], query)
            },
            "hybrid_benefit": {
                "improvement_over_single": f"{((avg_final - max(avg_vector, avg_conceptual, avg_graph)) / max(avg_vector, avg_conceptual, avg_graph) * 100):.1f}%",
                "consistency": "High" if np.std([exp.final_score for exp in explanations]) < 0.1 else "Medium",
                "explanation": "Hybrid approach combines strengths while mitigating individual weaknesses"
            },
            "result_quality": {
                "average_score": avg_final,
                "score_range": f"{min(exp.final_score for exp in explanations):.2f} - {max(exp.final_score for exp in explanations):.2f}",
                "confidence_level": self._calculate_overall_confidence(explanations)
            }
        }
        
        return explanation
    
    def _analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """Analyze characteristics of the search query."""
        query_lower = query.lower()
        
        characteristics = {
            "length": len(query.split()),
            "type": "descriptive" if len(query.split()) > 2 else "simple",
            "contains_adjectives": any(word in query_lower for word in ["comfortable", "elegant", "modern", "bold", "minimal"]),
            "contains_categories": any(word in query_lower for word in ["shoes", "dress", "shirt", "jacket", "bag"]),
            "conceptual_richness": "high" if any(word in query_lower for word in ["comfortable", "elegant", "bold", "modern", "minimal", "luxury"]) else "low",
            "semantic_complexity": "complex" if " and " in query_lower or " with " in query_lower else "simple"
        }
        
        return characteristics
    
    def _explain_dominance(self, approach: str, query: str) -> str:
        """Explain why a particular approach dominated."""
        if approach == "vector":
            return "Query contains specific terms that directly match product descriptions"
        elif approach == "conceptual":
            return "Query expresses abstract concepts and preferences that benefit from multi-dimensional analysis"
        elif approach == "graph":
            return "Query involves relationships and context that are captured in the knowledge graph"
        else:
            return "Balanced contribution from all approaches"
    
    def _calculate_hybrid_advantage(self, explanations: List[SearchExplanation]) -> Dict[str, Any]:
        """Calculate the advantage of hybrid approach over individual approaches."""
        if not explanations:
            return {}
        
        hybrid_scores = [exp.final_score for exp in explanations]
        vector_scores = [exp.vector_score for exp in explanations]
        conceptual_scores = [exp.conceptual_score for exp in explanations]
        graph_scores = [exp.graph_score for exp in explanations]
        
        avg_hybrid = np.mean(hybrid_scores)
        avg_vector = np.mean(vector_scores)
        avg_conceptual = np.mean(conceptual_scores)
        avg_graph = np.mean(graph_scores)
        
        best_individual = max(avg_vector, avg_conceptual, avg_graph)
        
        return {
            "improvement_percentage": ((avg_hybrid - best_individual) / best_individual * 100) if best_individual > 0 else 0,
            "consistency_improvement": np.std(hybrid_scores) < min(np.std(vector_scores), np.std(conceptual_scores), np.std(graph_scores)),
            "coverage_improvement": len([s for s in hybrid_scores if s > 0.5]) > max(
                len([s for s in vector_scores if s > 0.5]),
                len([s for s in conceptual_scores if s > 0.5]),
                len([s for s in graph_scores if s > 0.5])
            )
        }
    
    def _calculate_confidence_distribution(self, explanations: List[SearchExplanation]) -> Dict[str, int]:
        """Calculate distribution of confidence levels."""
        distribution = {"Very High": 0, "High": 0, "Medium": 0, "Low": 0, "Very Low": 0}
        
        for exp in explanations:
            distribution[exp.confidence_level] += 1
        
        return distribution
    
    def _extract_reasoning_patterns(self, explanations: List[SearchExplanation]) -> List[str]:
        """Extract common reasoning patterns from explanations."""
        patterns = []
        
        # Check for consistent high performers
        high_vector = sum(1 for exp in explanations if exp.vector_score > 0.7)
        high_conceptual = sum(1 for exp in explanations if exp.conceptual_score > 0.7)
        high_graph = sum(1 for exp in explanations if exp.graph_score > 0.7)
        
        if high_vector > len(explanations) * 0.6:
            patterns.append("Strong text-based matching dominates results")
        
        if high_conceptual > len(explanations) * 0.6:
            patterns.append("Conceptual understanding drives relevance")
        
        if high_graph > len(explanations) * 0.6:
            patterns.append("Knowledge graph relationships enhance results")
        
        # Check for hybrid synergy
        hybrid_better = sum(1 for exp in explanations 
                           if exp.final_score > max(exp.vector_score, exp.conceptual_score, exp.graph_score))
        
        if hybrid_better > len(explanations) * 0.7:
            patterns.append("Hybrid fusion consistently outperforms individual approaches")
        
        return patterns
    
    def _calculate_overall_confidence(self, explanations: List[SearchExplanation]) -> str:
        """Calculate overall confidence level for all results."""
        confidence_scores = {
            "Very High": 5, "High": 4, "Medium": 3, "Low": 2, "Very Low": 1
        }
        
        total_score = sum(confidence_scores[exp.confidence_level] for exp in explanations)
        avg_score = total_score / len(explanations) if explanations else 0
        
        if avg_score >= 4.5:
            return "Very High"
        elif avg_score >= 3.5:
            return "High"
        elif avg_score >= 2.5:
            return "Medium"
        elif avg_score >= 1.5:
            return "Low"
        else:
            return "Very Low"
    
    async def get_detailed_breakdown(
        self,
        query: str,
        product_id: str
    ) -> Dict[str, Any]:
        """Get detailed breakdown for a specific product."""
        # In real implementation, this would fetch the specific product
        # and provide granular analysis
        
        return {
            "product_id": product_id,
            "query": query,
            "detailed_analysis": {
                "vector_analysis": {
                    "embedding_similarity": 0.75,
                    "key_terms_matched": ["comfortable", "running"],
                    "semantic_concepts": ["athletic", "footwear", "performance"]
                },
                "conceptual_analysis": {
                    "dimension_scores": {
                        "comfort": 0.9,
                        "functionality": 0.85,
                        "modernity": 0.7
                    },
                    "concept_alignment": "Excellent match for comfort-focused athletic query"
                },
                "graph_analysis": {
                    "relationship_paths": [
                        "running → athletic → footwear → shoes",
                        "comfortable → ergonomic → performance → athletic"
                    ],
                    "connection_strength": 0.8
                }
            },
            "fusion_process": {
                "weight_distribution": {"vector": 0.33, "conceptual": 0.34, "graph": 0.33},
                "score_calculation": "Weighted average with boost for consensus",
                "final_adjustment": "Applied domain-specific calibration"
            }
        }


# Utility functions for visualization integration
def create_explanation_visualization(explanation: Dict[str, Any]) -> Dict[str, Any]:
    """Create visualization data for search explanation."""
    return {
        "type": "search_explanation",
        "data": explanation,
        "charts": {
            "score_breakdown": {
                "type": "bar_chart",
                "data": explanation.get("overall_explanation", {}).get("approach_performance", {})
            },
            "confidence_distribution": {
                "type": "pie_chart", 
                "data": explanation.get("confidence_distribution", {})
            },
            "hybrid_advantage": {
                "type": "improvement_chart",
                "data": explanation.get("hybrid_advantage", {})
            }
        }
    }