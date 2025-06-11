"""
KSE Memory SDK - Knowledge Graph Visualizer

Interactive visualization of knowledge graph relationships
showing how products connect through semantic networks.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import asyncio

from ..core.memory import KSEMemory
from ..core.models import Product


class KnowledgeGraphVisualizer:
    """
    Interactive knowledge graph visualization for KSE Memory.
    
    Shows how products connect through:
    - Category hierarchies
    - Attribute relationships
    - Semantic associations
    - Cross-domain connections
    
    Example:
        visualizer = KnowledgeGraphVisualizer(kse_memory)
        graph_data = await visualizer.get_graph_data(
            focus_products=["shoe_001", "shoe_002"],
            relationship_types=["category", "attribute", "semantic"]
        )
    """
    
    def __init__(self, kse_memory: KSEMemory):
        """Initialize knowledge graph visualizer."""
        self.kse_memory = kse_memory
    
    async def get_graph_data(
        self,
        focus_products: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        max_nodes: int = 100,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get knowledge graph data for visualization.
        
        Args:
            focus_products: Specific products to center the graph around
            relationship_types: Types of relationships to include
            max_nodes: Maximum number of nodes to include
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            Graph data structure for visualization
        """
        # Generate sample graph data for demonstration
        nodes, edges = await self._generate_sample_graph(
            focus_products, relationship_types, max_nodes, max_depth
        )
        
        # Calculate graph metrics
        metrics = self._calculate_graph_metrics(nodes, edges)
        
        # Generate layout configuration
        layout_config = self._get_layout_config(len(nodes), len(edges))
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": metrics,
            "layout": layout_config,
            "legend": self._get_legend(),
            "interactions": self._get_interaction_config(),
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "focus_products": focus_products or [],
                "relationship_types": relationship_types or ["all"],
                "generated_at": "2024-01-01T00:00:00Z"
            }
        }
    
    async def _generate_sample_graph(
        self,
        focus_products: Optional[List[str]],
        relationship_types: Optional[List[str]],
        max_nodes: int,
        max_depth: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate sample knowledge graph data."""
        
        # Sample nodes (products and concepts)
        nodes = [
            # Product nodes
            {
                "id": "shoe_001",
                "label": "Premium Running Shoes",
                "type": "product",
                "category": "Athletic Footwear",
                "size": 30,
                "color": "#3B82F6",
                "properties": {
                    "price": 129.99,
                    "brand": "SportTech",
                    "rating": 4.5
                }
            },
            {
                "id": "shoe_002", 
                "label": "Elegant Dress Shoes",
                "type": "product",
                "category": "Formal Footwear",
                "size": 25,
                "color": "#8B5CF6",
                "properties": {
                    "price": 299.99,
                    "brand": "LuxuryWalk",
                    "rating": 4.8
                }
            },
            {
                "id": "shoe_003",
                "label": "Casual Sneakers",
                "type": "product", 
                "category": "Casual Footwear",
                "size": 20,
                "color": "#10B981",
                "properties": {
                    "price": 89.99,
                    "brand": "ComfortStep",
                    "rating": 4.2
                }
            },
            
            # Category nodes
            {
                "id": "footwear",
                "label": "Footwear",
                "type": "category",
                "size": 40,
                "color": "#F59E0B",
                "properties": {
                    "level": "root",
                    "product_count": 150
                }
            },
            {
                "id": "athletic_footwear",
                "label": "Athletic Footwear",
                "type": "category",
                "size": 35,
                "color": "#EF4444",
                "properties": {
                    "level": "subcategory",
                    "product_count": 45
                }
            },
            {
                "id": "formal_footwear",
                "label": "Formal Footwear", 
                "type": "category",
                "size": 30,
                "color": "#6B7280",
                "properties": {
                    "level": "subcategory",
                    "product_count": 32
                }
            },
            
            # Attribute nodes
            {
                "id": "comfortable",
                "label": "Comfortable",
                "type": "attribute",
                "size": 25,
                "color": "#10B981",
                "properties": {
                    "type": "comfort",
                    "importance": 0.9
                }
            },
            {
                "id": "elegant",
                "label": "Elegant",
                "type": "attribute",
                "size": 20,
                "color": "#8B5CF6",
                "properties": {
                    "type": "style",
                    "importance": 0.8
                }
            },
            {
                "id": "athletic",
                "label": "Athletic",
                "type": "attribute",
                "size": 22,
                "color": "#EF4444",
                "properties": {
                    "type": "function",
                    "importance": 0.85
                }
            },
            
            # Brand nodes
            {
                "id": "sporttech",
                "label": "SportTech",
                "type": "brand",
                "size": 18,
                "color": "#06B6D4",
                "properties": {
                    "founded": 1995,
                    "specialty": "athletic"
                }
            },
            {
                "id": "luxurywalk",
                "label": "LuxuryWalk",
                "type": "brand",
                "size": 16,
                "color": "#EC4899",
                "properties": {
                    "founded": 1987,
                    "specialty": "formal"
                }
            }
        ]
        
        # Sample edges (relationships)
        edges = [
            # Category relationships
            {
                "id": "edge_001",
                "source": "shoe_001",
                "target": "athletic_footwear",
                "type": "belongs_to",
                "label": "belongs to",
                "weight": 1.0,
                "color": "#94A3B8",
                "properties": {
                    "confidence": 0.95,
                    "source_type": "classification"
                }
            },
            {
                "id": "edge_002",
                "source": "shoe_002",
                "target": "formal_footwear",
                "type": "belongs_to",
                "label": "belongs to",
                "weight": 1.0,
                "color": "#94A3B8",
                "properties": {
                    "confidence": 0.98,
                    "source_type": "classification"
                }
            },
            {
                "id": "edge_003",
                "source": "athletic_footwear",
                "target": "footwear",
                "type": "subcategory_of",
                "label": "subcategory of",
                "weight": 0.8,
                "color": "#64748B",
                "properties": {
                    "confidence": 1.0,
                    "source_type": "taxonomy"
                }
            },
            {
                "id": "edge_004",
                "source": "formal_footwear",
                "target": "footwear",
                "type": "subcategory_of",
                "label": "subcategory of",
                "weight": 0.8,
                "color": "#64748B",
                "properties": {
                    "confidence": 1.0,
                    "source_type": "taxonomy"
                }
            },
            
            # Attribute relationships
            {
                "id": "edge_005",
                "source": "shoe_001",
                "target": "comfortable",
                "type": "has_attribute",
                "label": "has attribute",
                "weight": 0.9,
                "color": "#10B981",
                "properties": {
                    "confidence": 0.9,
                    "source_type": "conceptual"
                }
            },
            {
                "id": "edge_006",
                "source": "shoe_001",
                "target": "athletic",
                "type": "has_attribute",
                "label": "has attribute",
                "weight": 0.95,
                "color": "#EF4444",
                "properties": {
                    "confidence": 0.95,
                    "source_type": "conceptual"
                }
            },
            {
                "id": "edge_007",
                "source": "shoe_002",
                "target": "elegant",
                "type": "has_attribute",
                "label": "has attribute",
                "weight": 0.95,
                "color": "#8B5CF6",
                "properties": {
                    "confidence": 0.95,
                    "source_type": "conceptual"
                }
            },
            
            # Brand relationships
            {
                "id": "edge_008",
                "source": "shoe_001",
                "target": "sporttech",
                "type": "manufactured_by",
                "label": "manufactured by",
                "weight": 1.0,
                "color": "#06B6D4",
                "properties": {
                    "confidence": 1.0,
                    "source_type": "metadata"
                }
            },
            {
                "id": "edge_009",
                "source": "shoe_002",
                "target": "luxurywalk",
                "type": "manufactured_by",
                "label": "manufactured by",
                "weight": 1.0,
                "color": "#EC4899",
                "properties": {
                    "confidence": 1.0,
                    "source_type": "metadata"
                }
            },
            
            # Semantic relationships
            {
                "id": "edge_010",
                "source": "comfortable",
                "target": "athletic",
                "type": "enhances",
                "label": "enhances",
                "weight": 0.7,
                "color": "#F59E0B",
                "properties": {
                    "confidence": 0.7,
                    "source_type": "semantic"
                }
            },
            {
                "id": "edge_011",
                "source": "shoe_001",
                "target": "shoe_003",
                "type": "similar_to",
                "label": "similar to",
                "weight": 0.6,
                "color": "#A78BFA",
                "properties": {
                    "confidence": 0.6,
                    "source_type": "similarity"
                }
            }
        ]
        
        return nodes, edges
    
    def _calculate_graph_metrics(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate graph topology metrics."""
        
        # Node type distribution
        node_types = {}
        for node in nodes:
            node_type = node.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Edge type distribution
        edge_types = {}
        for edge in edges:
            edge_type = edge.get("type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Calculate connectivity metrics
        node_degrees = {}
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1
        
        avg_degree = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0
        max_degree = max(node_degrees.values()) if node_degrees else 0
        
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "connectivity": {
                "average_degree": avg_degree,
                "max_degree": max_degree,
                "density": len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
            },
            "components": {
                "connected_components": 1,  # Simplified for demo
                "largest_component_size": len(nodes)
            }
        }
    
    def _get_layout_config(self, node_count: int, edge_count: int) -> Dict[str, Any]:
        """Get optimal layout configuration based on graph size."""
        
        # Choose layout algorithm based on graph size
        if node_count < 20:
            layout_name = "circle"
        elif node_count < 50:
            layout_name = "cose"
        else:
            layout_name = "cola"
        
        return {
            "name": layout_name,
            "animate": True,
            "animationDuration": 1000,
            "fit": True,
            "padding": 50,
            "nodeRepulsion": 8000,
            "idealEdgeLength": 100,
            "edgeElasticity": 200,
            "nestingFactor": 1.2,
            "gravity": 1,
            "numIter": 1000,
            "initialTemp": 1000,
            "coolingFactor": 0.99,
            "minTemp": 1.0
        }
    
    def _get_legend(self) -> Dict[str, Any]:
        """Get legend configuration for graph elements."""
        return {
            "node_types": {
                "product": {
                    "color": "#3B82F6",
                    "shape": "ellipse",
                    "description": "Product items"
                },
                "category": {
                    "color": "#F59E0B",
                    "shape": "rectangle",
                    "description": "Product categories"
                },
                "attribute": {
                    "color": "#10B981",
                    "shape": "diamond",
                    "description": "Product attributes"
                },
                "brand": {
                    "color": "#06B6D4",
                    "shape": "triangle",
                    "description": "Brand entities"
                }
            },
            "edge_types": {
                "belongs_to": {
                    "color": "#94A3B8",
                    "style": "solid",
                    "description": "Category membership"
                },
                "has_attribute": {
                    "color": "#10B981",
                    "style": "dashed",
                    "description": "Attribute relationships"
                },
                "manufactured_by": {
                    "color": "#06B6D4",
                    "style": "dotted",
                    "description": "Brand relationships"
                },
                "similar_to": {
                    "color": "#A78BFA",
                    "style": "solid",
                    "description": "Similarity connections"
                },
                "enhances": {
                    "color": "#F59E0B",
                    "style": "dashed",
                    "description": "Semantic relationships"
                }
            }
        }
    
    def _get_interaction_config(self) -> Dict[str, Any]:
        """Get interaction configuration for the graph."""
        return {
            "hover": {
                "enabled": True,
                "highlight_neighbors": True,
                "show_tooltip": True,
                "fade_others": True
            },
            "click": {
                "enabled": True,
                "select_node": True,
                "expand_neighbors": True,
                "show_details": True
            },
            "zoom": {
                "enabled": True,
                "min_zoom": 0.1,
                "max_zoom": 3.0,
                "zoom_speed": 0.1
            },
            "pan": {
                "enabled": True,
                "pan_speed": 1.0
            },
            "selection": {
                "enabled": True,
                "multiple": True,
                "box_selection": True
            },
            "search": {
                "enabled": True,
                "highlight_results": True,
                "center_on_result": True
            }
        }
    
    async def get_search_path(
        self,
        query: str,
        result_product_id: str
    ) -> Dict[str, Any]:
        """
        Get the reasoning path through the knowledge graph for a search result.
        
        Args:
            query: Original search query
            result_product_id: Product ID that was returned as a result
            
        Returns:
            Path data showing how the search traversed the graph
        """
        # Simulate search path through knowledge graph
        path_nodes = [
            {
                "id": "query_node",
                "label": f"Query: {query}",
                "type": "query",
                "step": 0,
                "reasoning": "Starting point for graph traversal"
            },
            {
                "id": "comfortable",
                "label": "Comfortable",
                "type": "attribute",
                "step": 1,
                "reasoning": "Query contains 'comfortable' - matched to comfort attribute"
            },
            {
                "id": "athletic",
                "label": "Athletic",
                "type": "attribute", 
                "step": 2,
                "reasoning": "Comfort enhances athletic performance - semantic relationship"
            },
            {
                "id": "athletic_footwear",
                "label": "Athletic Footwear",
                "type": "category",
                "step": 3,
                "reasoning": "Athletic attribute leads to athletic footwear category"
            },
            {
                "id": result_product_id,
                "label": "Premium Running Shoes",
                "type": "product",
                "step": 4,
                "reasoning": "Product belongs to athletic footwear category"
            }
        ]
        
        path_edges = [
            {
                "source": "query_node",
                "target": "comfortable",
                "type": "matches",
                "reasoning": "Query term extraction"
            },
            {
                "source": "comfortable",
                "target": "athletic",
                "type": "enhances",
                "reasoning": "Semantic relationship in knowledge graph"
            },
            {
                "source": "athletic",
                "target": "athletic_footwear",
                "type": "categorizes",
                "reasoning": "Attribute to category mapping"
            },
            {
                "source": "athletic_footwear",
                "target": result_product_id,
                "type": "contains",
                "reasoning": "Category membership"
            }
        ]
        
        return {
            "query": query,
            "result_product_id": result_product_id,
            "path_nodes": path_nodes,
            "path_edges": path_edges,
            "path_length": len(path_nodes) - 1,
            "reasoning_summary": "Query traversed through comfort → athletic → category → product",
            "confidence": 0.85,
            "alternative_paths": [
                {
                    "path": ["query", "athletic", "athletic_footwear", result_product_id],
                    "confidence": 0.75,
                    "reasoning": "Direct athletic attribute matching"
                }
            ]
        }
    
    async def get_product_neighborhood(
        self,
        product_id: str,
        radius: int = 2
    ) -> Dict[str, Any]:
        """
        Get the neighborhood around a specific product in the knowledge graph.
        
        Args:
            product_id: Product to center the neighborhood around
            radius: Number of hops to include
            
        Returns:
            Subgraph showing product's immediate context
        """
        # Generate neighborhood data
        center_product = {
            "id": product_id,
            "label": "Premium Running Shoes",
            "type": "product",
            "distance": 0,
            "importance": 1.0
        }
        
        # 1-hop neighbors
        neighbors_1 = [
            {
                "id": "athletic_footwear",
                "label": "Athletic Footwear",
                "type": "category",
                "distance": 1,
                "importance": 0.9,
                "relationship": "belongs_to"
            },
            {
                "id": "comfortable",
                "label": "Comfortable",
                "type": "attribute",
                "distance": 1,
                "importance": 0.8,
                "relationship": "has_attribute"
            },
            {
                "id": "sporttech",
                "label": "SportTech",
                "type": "brand",
                "distance": 1,
                "importance": 0.7,
                "relationship": "manufactured_by"
            }
        ]
        
        # 2-hop neighbors
        neighbors_2 = [
            {
                "id": "footwear",
                "label": "Footwear",
                "type": "category",
                "distance": 2,
                "importance": 0.6,
                "relationship": "subcategory_of"
            },
            {
                "id": "athletic",
                "label": "Athletic",
                "type": "attribute",
                "distance": 2,
                "importance": 0.7,
                "relationship": "enhances"
            }
        ]
        
        all_nodes = [center_product] + neighbors_1 + neighbors_2
        
        return {
            "center_product": center_product,
            "neighborhood": {
                "nodes": all_nodes,
                "radius": radius,
                "total_nodes": len(all_nodes),
                "node_types": {
                    "product": 1,
                    "category": 2,
                    "attribute": 2,
                    "brand": 1
                }
            },
            "insights": {
                "strongest_connections": ["athletic_footwear", "comfortable"],
                "connection_diversity": "High - spans categories, attributes, and brands",
                "centrality_score": 0.85
            }
        }


# Utility functions for graph analysis
def calculate_node_centrality(nodes: List[Dict], edges: List[Dict]) -> Dict[str, float]:
    """Calculate centrality scores for nodes."""
    centrality = {}
    
    # Simple degree centrality
    for node in nodes:
        node_id = node["id"]
        degree = sum(1 for edge in edges if edge["source"] == node_id or edge["target"] == node_id)
        centrality[node_id] = degree / (len(nodes) - 1) if len(nodes) > 1 else 0
    
    return centrality


def find_shortest_path(edges: List[Dict], start: str, end: str) -> List[str]:
    """Find shortest path between two nodes."""
    # Simple BFS implementation
    from collections import deque
    
    # Build adjacency list
    graph = {}
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source not in graph:
            graph[source] = []
        if target not in graph:
            graph[target] = []
        graph[source].append(target)
        graph[target].append(source)  # Undirected
    
    if start not in graph or end not in graph:
        return []
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        node, path = queue.popleft()
        
        if node == end:
            return path
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path found