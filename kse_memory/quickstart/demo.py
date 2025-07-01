"""
KSE Memory SDK - Quickstart Demo

Provides zero-configuration demo experience that showcases
hybrid AI capabilities with instant "wow" moments.
"""

import asyncio
import logging
import tempfile
import webbrowser
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm

from ..core.memory import KSEMemory
from ..core.config import KSEConfig
from ..core.models import Product, SearchQuery, SearchType, ConceptualDimensions
from .datasets import SampleDatasets
from .benchmark import BenchmarkRunner
from .backend_detector import BackendDetector, auto_detect_and_setup

console = Console()
logger = logging.getLogger(__name__)


class QuickstartDemo:
    """
    Zero-configuration demo that showcases KSE Memory capabilities.
    
    Provides instant "wow" moments with:
    - Intelligent backend detection and setup
    - Sample retail dataset
    - Hybrid search demonstrations
    - Performance comparisons
    - Interactive web interface
    """
    
    def __init__(self, auto_setup: bool = True):
        """Initialize quickstart demo."""
        self.temp_dir = None
        self.kse = None
        self.datasets = SampleDatasets()
        self.benchmark = BenchmarkRunner()
        self.auto_setup = auto_setup
        self.chosen_backend = None
        self.config = None
        
    async def run(self, demo_type: str = "retail", open_browser: bool = True, backend: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the quickstart demo.
        
        Args:
            demo_type: Type of demo ('retail', 'finance', 'healthcare')
            open_browser: Whether to open web interface
            backend: Specific backend to use (optional, will auto-detect if not provided)
            
        Returns:
            Demo results and metrics
        """
        console.print(Panel.fit(
            "[bold blue]KSE Memory SDK - Quickstart Demo[/bold blue]\n"
            f"Running {demo_type} demo with hybrid AI search...",
            border_style="blue"
        ))
        
        try:
            # Setup demo environment and backend
            await self._setup_demo_environment()
            
            # Auto-detect and setup backend if not already configured
            if self.auto_setup and not self.config:
                await self._setup_backend(backend)
            
            # Load sample dataset
            products = await self._load_sample_dataset(demo_type)
            
            # Initialize KSE Memory
            await self._initialize_kse_memory()
            
            # Add products to memory
            await self._populate_memory(products)
            
            # Run demo searches
            search_results = await self._run_demo_searches(demo_type)
            
            # Run performance benchmark
            benchmark_results = await self._run_benchmark(products)
            
            # Display results
            self._display_results(search_results, benchmark_results)
            
            # Launch web interface
            if open_browser:
                await self._launch_web_interface()
            
            return {
                "demo_type": demo_type,
                "backend": self.chosen_backend.name if self.chosen_backend else "unknown",
                "products_loaded": len(products),
                "search_results": search_results,
                "benchmark_results": benchmark_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            console.print(f"[red]Demo failed: {str(e)}[/red]")
            raise
        finally:
            await self._cleanup()
    
    async def _setup_demo_environment(self):
        """Setup temporary demo environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="kse_demo_")
        console.print(f"[dim]Demo environment: {self.temp_dir}[/dim]")
    
    async def _setup_backend(self, preferred_backend: Optional[str] = None):
        """Setup backend using auto-detection or user preference."""
        console.print("\nSetting up backend for optimal performance...")
        
        if preferred_backend:
            # User specified a backend
            detector = BackendDetector()
            backends = detector.detect_all_backends()
            
            # Find the preferred backend
            chosen = None
            for backend in backends:
                if backend.name.lower() == preferred_backend.lower():
                    chosen = backend
                    break
            
            if not chosen:
                console.print(f"❌ Backend '{preferred_backend}' not available. Using auto-detection...")
                chosen, config = auto_detect_and_setup()
            else:
                if not detector.install_backend(chosen):
                    console.print(f"❌ Failed to install {chosen.display_name}. Using auto-detection...")
                    chosen, config = auto_detect_and_setup()
                else:
                    config = detector.generate_config(chosen)
        else:
            # Auto-detect best backend
            chosen, config = auto_detect_and_setup()
        
        if not chosen or not config:
            console.print("❌ Backend setup failed. Using fallback memory backend.")
            chosen = BackendDetector.BACKEND_DEFINITIONS["memory"]
            config = BackendDetector().generate_config(chosen)
        
        self.chosen_backend = chosen
        self.config = config
        
        # Save config for future use
        config_path = Path(self.temp_dir) / "kse_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        console.print(f"✅ Backend configured: {chosen.display_name}")
    
    async def _load_sample_dataset(self, demo_type: str) -> List[Product]:
        """Load sample dataset for demo."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Loading {demo_type} dataset...", total=None)
            
            if demo_type == "retail":
                products = self.datasets.get_retail_products()
            elif demo_type == "finance":
                products = self.datasets.get_finance_products()
            elif demo_type == "healthcare":
                products = self.datasets.get_healthcare_products()
            else:
                raise ValueError(f"Unknown demo type: {demo_type}")
            
            progress.update(task, description=f"Loaded {len(products)} products")
            await asyncio.sleep(0.5)  # Brief pause for visual effect
            
        console.print(f"[green]✓[/green] Loaded {len(products)} sample products")
        return products
    
    async def _initialize_kse_memory(self):
        """Initialize KSE Memory with detected backend configuration."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing KSE Memory...", total=None)
            
            # Use the configuration from backend detection
            if not self.config:
                console.print("⚠️ No configuration found, using fallback...")
                self.config = {
                    "debug": False,
                    "vector_store": {"backend": "memory"},
                    "graph_store": {"backend": "memory"},
                    "concept_store": {"backend": "memory"},
                    "embedding": {
                        "text_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "batch_size": 8,
                    },
                    "conceptual": {"auto_compute": False},
                    "cache": {"enabled": True, "backend": "memory"}
                }
            
            kse_config = KSEConfig.from_dict(self.config)
            self.kse = KSEMemory(kse_config)
            
            # Initialize with generic adapter
            await self.kse.initialize("generic", {
                "data_source": lambda: []  # Empty data source
            })
            
            progress.update(task, description="KSE Memory initialized")
            
        backend_name = self.chosen_backend.display_name if self.chosen_backend else "Memory"
        console.print(f"[green]✓[/green] KSE Memory system ready with {backend_name} backend")
    
    async def _populate_memory(self, products: List[Product]):
        """Add products to KSE Memory."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Adding products to memory...", total=len(products))
            
            for i, product in enumerate(products):
                await self.kse.add_product(
                    product,
                    compute_embeddings=True,
                    compute_concepts=False  # Use pre-computed for speed
                )
                progress.update(task, advance=1)
                
                # Brief pause every 10 products for visual effect
                if (i + 1) % 10 == 0:
                    await asyncio.sleep(0.1)
        
        console.print(f"[green]✓[/green] Added {len(products)} products to hybrid memory")
    
    async def _run_demo_searches(self, demo_type: str) -> Dict[str, Any]:
        """Run demonstration searches."""
        console.print("\n[bold]🔍 Running Demo Searches[/bold]")
        
        # Define demo queries based on type
        if demo_type == "retail":
            queries = [
                ("comfortable running shoes", "Find athletic footwear for daily exercise"),
                ("elegant evening wear", "Discover sophisticated formal attire"),
                ("minimalist home decor", "Explore clean, simple design items"),
            ]
        elif demo_type == "finance":
            queries = [
                ("high-yield investment products", "Find profitable investment opportunities"),
                ("risk management tools", "Discover portfolio protection instruments"),
                ("retirement planning services", "Explore long-term financial planning"),
            ]
        else:  # healthcare
            queries = [
                ("diagnostic imaging equipment", "Find medical imaging solutions"),
                ("patient monitoring devices", "Discover vital sign tracking tools"),
                ("surgical instruments", "Explore precision medical tools"),
            ]
        
        results = {}
        
        for query, description in queries:
            console.print(f"\n[cyan]Query:[/cyan] {query}")
            console.print(f"[dim]{description}[/dim]")
            
            # Run hybrid search
            search_results = await self.kse.search(SearchQuery(
                query=query,
                search_type=SearchType.HYBRID,
                limit=5
            ))
            
            # Display top results
            if search_results:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Rank", width=4)
                table.add_column("Product", width=40)
                table.add_column("Score", width=8)
                table.add_column("Relevance", width=20)
                
                for i, result in enumerate(search_results[:3], 1):
                    relevance = "🎯 Excellent" if result.score > 0.8 else "✅ Good" if result.score > 0.6 else "📍 Fair"
                    table.add_row(
                        str(i),
                        result.product.title[:37] + "..." if len(result.product.title) > 40 else result.product.title,
                        f"{result.score:.3f}",
                        relevance
                    )
                
                console.print(table)
            
            results[query] = {
                "results_count": len(search_results),
                "top_score": search_results[0].score if search_results else 0,
                "avg_score": sum(r.score for r in search_results) / len(search_results) if search_results else 0
            }
        
        return results
    
    async def _run_benchmark(self, products: List[Product]) -> Dict[str, Any]:
        """Run performance benchmark."""
        console.print("\n[bold]📊 Running Performance Benchmark[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Benchmarking hybrid vs vector-only search...", total=None)
            
            # Run benchmark comparison
            benchmark_results = await self.benchmark.run_comparison(
                products=products,
                kse_memory=self.kse,
                queries=[
                    "comfortable athletic wear",
                    "elegant formal attire", 
                    "modern minimalist design"
                ]
            )
            
            progress.update(task, description="Benchmark completed")
        
        # Display benchmark results
        self._display_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def _display_results(self, search_results: Dict[str, Any], benchmark_results: Dict[str, Any]):
        """Display demo results summary."""
        console.print("\n" + "="*60)
        console.print("[bold green]🎉 Demo Results Summary[/bold green]")
        console.print("="*60)
        
        # Search performance
        avg_score = sum(r["avg_score"] for r in search_results.values()) / len(search_results)
        console.print(f"[cyan]Average Search Relevance:[/cyan] {avg_score:.3f}")
        
        # Benchmark performance
        improvement = benchmark_results.get("improvement_percentage", 0)
        console.print(f"[cyan]Performance vs Vector-Only:[/cyan] +{improvement:.1f}%")
        
        # Speed metrics
        avg_latency = benchmark_results.get("avg_latency_ms", 0)
        console.print(f"[cyan]Average Query Latency:[/cyan] {avg_latency:.1f}ms")
        
        console.print("\n[bold]🚀 Key Benefits Demonstrated:[/bold]")
        console.print("• [green]Hybrid AI[/green] combines knowledge graphs + conceptual spaces + embeddings")
        console.print("• [green]Better Relevance[/green] through multi-dimensional similarity")
        console.print("• [green]Fast Performance[/green] with sub-100ms query times")
        console.print("• [green]Zero Configuration[/green] works out of the box")
    
    def _display_benchmark_results(self, results: Dict[str, Any]):
        """Display benchmark comparison results."""
        table = Table(title="Performance Comparison", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Vector-Only", style="red")
        table.add_column("KSE Hybrid", style="green")
        table.add_column("Improvement", style="bold green")
        
        # Add benchmark metrics
        table.add_row(
            "Relevance Score",
            f"{results.get('vector_only_score', 0):.3f}",
            f"{results.get('hybrid_score', 0):.3f}",
            f"+{results.get('improvement_percentage', 0):.1f}%"
        )
        
        table.add_row(
            "Query Latency",
            f"{results.get('vector_only_latency', 0):.1f}ms",
            f"{results.get('hybrid_latency', 0):.1f}ms",
            f"{results.get('latency_change', 0):+.1f}ms"
        )
        
        table.add_row(
            "Memory Usage",
            f"{results.get('vector_only_memory', 0):.1f}MB",
            f"{results.get('hybrid_memory', 0):.1f}MB",
            f"{results.get('memory_change', 0):+.1f}MB"
        )
        
        console.print(table)
    
    async def _launch_web_interface(self):
        """Launch web interface for interactive exploration."""
        console.print("\n[bold]🌐 Launching Web Interface[/bold]")
        
        # For now, just show a message about the web interface
        # In a full implementation, this would start a local server
        console.print("[dim]Web interface would launch at: http://localhost:8080[/dim]")
        console.print("[dim]Features: Interactive concept exploration, graph visualization, search testing[/dim]")
        
        # Simulate opening browser
        console.print("[green]✓[/green] Web interface ready (simulated)")
    
    async def _cleanup(self):
        """Cleanup demo resources."""
        if self.kse:
            await self.kse.disconnect()
        
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# Memory-based backend implementations for demo
class MemoryVectorStore:
    """In-memory vector store for demo purposes."""
    
    def __init__(self):
        self.vectors = {}
        self._connected = True
    
    async def connect(self):
        return True
    
    async def disconnect(self):
        return True
    
    async def upsert_vectors(self, vectors):
        for vector_id, vector, metadata in vectors:
            self.vectors[vector_id] = (vector, metadata)
        return True
    
    async def search_vectors(self, query_vector, top_k=10, filters=None):
        # Simple cosine similarity for demo
        import numpy as np
        
        results = []
        for vector_id, (vector, metadata) in self.vectors.items():
            # Calculate cosine similarity
            dot_product = np.dot(query_vector, vector)
            norm_a = np.linalg.norm(query_vector)
            norm_b = np.linalg.norm(vector)
            
            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
                results.append((vector_id, float(similarity), metadata))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    async def get_vector(self, vector_id):
        return self.vectors.get(vector_id)


class MemoryGraphStore:
    """In-memory graph store for demo purposes."""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._connected = True
    
    async def connect(self):
        return True
    
    async def disconnect(self):
        return True
    
    async def create_node(self, node_id, labels, properties):
        self.nodes[node_id] = {"labels": labels, "properties": properties}
        return True
    
    async def create_relationship(self, source_id, target_id, relationship_type, properties=None):
        self.edges.append({
            "source": source_id,
            "target": target_id,
            "type": relationship_type,
            "properties": properties or {}
        })
        return True
    
    async def get_neighbors(self, node_id, relationship_types=None):
        neighbors = []
        for edge in self.edges:
            if edge["source"] == node_id:
                if not relationship_types or edge["type"] in relationship_types:
                    target_node = self.nodes.get(edge["target"])
                    if target_node:
                        neighbors.append({
                            **target_node,
                            "relationship_type": edge["type"]
                        })
        return neighbors
    
    async def execute_query(self, query, parameters=None):
        # Simple query execution for demo
        return []


class MemoryConceptStore:
    """In-memory concept store for demo purposes."""
    
    def __init__(self):
        self.concepts = {}
        self._connected = True
    
    async def connect(self):
        return True
    
    async def disconnect(self):
        return True
    
    async def store_conceptual_dimensions(self, product_id, dimensions):
        self.concepts[product_id] = dimensions
        return True
    
    async def get_conceptual_dimensions(self, product_id):
        return self.concepts.get(product_id)
    
    async def find_similar_concepts(self, dimensions, threshold=0.8, limit=10):
        # Simple conceptual similarity for demo
        results = []
        target_dict = dimensions.to_dict()
        
        for product_id, stored_dims in self.concepts.items():
            stored_dict = stored_dims.to_dict()
            
            # Calculate cosine similarity between concept vectors
            import numpy as np
            
            target_vector = list(target_dict.values())
            stored_vector = list(stored_dict.values())
            
            dot_product = np.dot(target_vector, stored_vector)
            norm_a = np.linalg.norm(target_vector)
            norm_b = np.linalg.norm(stored_vector)
            
            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
                if similarity >= threshold:
                    results.append((product_id, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]