"""
KSE Memory SDK - Command Line Interface

Enhanced CLI with quickstart command for zero-config demos.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from .core.memory import KSEMemory
from .core.config import KSEConfig
from .core.models import Product, SearchQuery, SearchType
from .quickstart.demo import QuickstartDemo
from .quickstart.datasets import SampleDatasets

console = Console()


@click.group()
@click.version_option()
def cli():
    """
    🧠 KSE Memory SDK - Hybrid AI for Intelligent Search
    
    Transform your product catalogs with Knowledge Graphs,
    Conceptual Spaces, and Neural Embeddings.
    """
    pass


@cli.command()
@click.option(
    "--demo-type",
    type=click.Choice(["retail", "finance", "healthcare"]),
    default="retail",
    help="Type of demo to run"
)
@click.option(
    "--backend",
    type=click.Choice(["chromadb", "weaviate", "qdrant", "memory", "auto"]),
    default="auto",
    help="Backend to use (auto-detects if not specified)"
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Skip opening web interface"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Save results to JSON file"
)
def quickstart(demo_type: str, backend: str, no_browser: bool, output: Optional[str]):
    """
    🚀 Run zero-configuration quickstart demo
    
    Experience hybrid AI search with instant "wow" moments.
    Automatically detects and sets up the best available backend.
    
    Examples:
        kse quickstart                          # Auto-detect best backend
        kse quickstart --backend chromadb       # Use ChromaDB (local, free)
        kse quickstart --backend weaviate       # Use Weaviate (cloud, free tier)
        kse quickstart --demo-type finance      # Run finance demo
        kse quickstart --no-browser             # Skip web interface
    """
    console.print(Panel.fit(
        "[bold blue]🚀 KSE Memory SDK Quickstart[/bold blue]\n"
        "Smart backend detection + zero-configuration demo",
        border_style="blue"
    ))
    
    async def run_demo():
        demo = QuickstartDemo()
        try:
            # Pass backend preference to demo
            backend_choice = None if backend == "auto" else backend
            
            results = await demo.run(
                demo_type=demo_type,
                open_browser=not no_browser,
                backend=backend_choice
            )
            
            if output:
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                console.print(f"[green]✓[/green] Results saved to {output}")
            
            console.print("\n[bold green]🎉 Quickstart demo completed![/bold green]")
            console.print("Ready to integrate KSE Memory into your application.")
            
        except Exception as e:
            console.print(f"[red]❌ Demo failed: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run_demo())


@cli.command()
@click.option(
    "--interactive",
    is_flag=True,
    help="Interactive backend selection with detailed comparison"
)
@click.option(
    "--output",
    type=click.Path(),
    default="kse_config.yaml",
    help="Output configuration file path"
)
def setup(interactive: bool, output: str):
    """
    🔧 Setup KSE Memory backend configuration
    
    Detect available backends and generate production-ready configuration.
    Perfect for first-time setup or switching backends.
    
    Examples:
        kse setup                    # Quick setup with auto-detection
        kse setup --interactive      # Detailed backend comparison
        kse setup --output my.yaml  # Save to custom file
    """
    from .quickstart.backend_detector import auto_detect_and_setup, BackendDetector
    import yaml
    
    console.print(Panel.fit(
        "[bold green]🔧 KSE Memory Backend Setup[/bold green]\n"
        "Configure your optimal backend for production use",
        border_style="green"
    ))
    
    try:
        if interactive:
            # Full interactive setup
            detector = BackendDetector()
            backends = detector.detect_all_backends()
            detector.display_backend_options(backends)
            chosen_backend = detector.get_user_choice(backends)
            
            if not chosen_backend:
                console.print("❌ Setup cancelled.")
                return
            
            if not detector.install_backend(chosen_backend):
                console.print("❌ Backend installation failed.")
                return
            
            config = detector.generate_config(chosen_backend)
        else:
            # Quick auto-detection
            chosen_backend, config = auto_detect_and_setup()
            
            if not chosen_backend or not config:
                console.print("❌ Backend setup failed.")
                return
        
        # Save configuration
        output_path = Path(output)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        console.print(Panel.fit(
            f"✅ [bold green]Configuration saved![/bold green]\n"
            f"Backend: {chosen_backend.display_name}\n"
            f"Config file: {output_path.absolute()}\n"
            f"Ready for: {chosen_backend.best_for}",
            border_style="green"
        ))
        
        console.print("\n🚀 Next steps:")
        console.print("  1. Review the configuration file")
        console.print("  2. Set any required environment variables")
        console.print("  3. Run: kse quickstart")
        
    except Exception as e:
        console.print(f"[red]❌ Setup failed: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--adapter",
    type=click.Choice(["shopify", "woocommerce", "generic"]),
    required=True,
    help="Platform adapter to use"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes"
)
def init(config: Optional[str], adapter: str, dry_run: bool):
    """
    🔧 Initialize KSE Memory for your platform
    
    Set up KSE Memory with your chosen platform adapter
    and configuration.
    
    Examples:
        kse init --adapter shopify
        kse init --adapter generic --config my-config.yaml
        kse init --adapter woocommerce --dry-run
    """
    console.print(f"[cyan]Initializing KSE Memory with {adapter} adapter...[/cyan]")
    
    if dry_run:
        console.print("[yellow]DRY RUN - No changes will be made[/yellow]")
    
    # Load configuration
    if config:
        console.print(f"[dim]Loading configuration from {config}[/dim]")
        kse_config = KSEConfig.from_file(config)
    else:
        console.print("[dim]Using default configuration[/dim]")
        kse_config = KSEConfig()
    
    # Display configuration summary
    _display_config_summary(kse_config, adapter)
    
    if not dry_run:
        if Confirm.ask("Proceed with initialization?"):
            async def init_kse():
                kse = KSEMemory(kse_config)
                await kse.initialize(adapter, {})
                console.print("[green]✓[/green] KSE Memory initialized successfully")
                await kse.disconnect()
            
            asyncio.run(init_kse())
        else:
            console.print("[yellow]Initialization cancelled[/yellow]")


@cli.command()
@click.option(
    "--query",
    required=True,
    help="Search query"
)
@click.option(
    "--type",
    "search_type",
    type=click.Choice(["vector", "conceptual", "graph", "hybrid"]),
    default="hybrid",
    help="Type of search to perform"
)
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of results"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
def search(query: str, search_type: str, limit: int, config: Optional[str]):
    """
    🔍 Search products using KSE Memory
    
    Perform searches using different search types to compare
    results and understand hybrid AI capabilities.
    
    Examples:
        kse search --query "comfortable running shoes"
        kse search --query "elegant dress" --type conceptual
        kse search --query "tech gadgets" --type hybrid --limit 5
    """
    console.print(f"[cyan]Searching for: {query}[/cyan]")
    console.print(f"[dim]Search type: {search_type}, Limit: {limit}[/dim]")
    
    async def perform_search():
        # Load configuration
        if config:
            kse_config = KSEConfig.from_file(config)
        else:
            kse_config = KSEConfig()
        
        kse = KSEMemory(kse_config)
        
        try:
            # Initialize with generic adapter for search
            await kse.initialize("generic", {})
            
            # Perform search
            search_query = SearchQuery(
                query=query,
                search_type=SearchType(search_type.upper()),
                limit=limit
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Searching...", total=None)
                results = await kse.search(search_query)
                progress.update(task, description=f"Found {len(results)} results")
            
            # Display results
            _display_search_results(results, query, search_type)
            
        except Exception as e:
            console.print(f"[red]❌ Search failed: {str(e)}[/red]")
            sys.exit(1)
        finally:
            await kse.disconnect()
    
    asyncio.run(perform_search())


@cli.command()
@click.option(
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to products JSON file"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Batch size for processing"
)
def ingest(input: str, config: Optional[str], batch_size: int):
    """
    📥 Ingest products into KSE Memory
    
    Load products from a JSON file and add them to KSE Memory
    with full hybrid AI processing.
    
    Examples:
        kse ingest --input products.json
        kse ingest --input catalog.json --batch-size 50
    """
    console.print(f"[cyan]Ingesting products from {input}...[/cyan]")
    
    async def ingest_products():
        # Load configuration
        if config:
            kse_config = KSEConfig.from_file(config)
        else:
            kse_config = KSEConfig()
        
        kse = KSEMemory(kse_config)
        
        try:
            # Initialize with generic adapter
            await kse.initialize("generic", {})
            
            # Load products from file
            with open(input, 'r') as f:
                products_data = json.load(f)
            
            products = [Product.from_dict(p) for p in products_data]
            console.print(f"[dim]Loaded {len(products)} products[/dim]")
            
            # Ingest products in batches
            with Progress(console=console) as progress:
                task = progress.add_task("Ingesting products...", total=len(products))
                
                for i in range(0, len(products), batch_size):
                    batch = products[i:i + batch_size]
                    
                    for product in batch:
                        await kse.add_product(product)
                        progress.advance(task)
            
            console.print(f"[green]✓[/green] Successfully ingested {len(products)} products")
            
        except Exception as e:
            console.print(f"[red]❌ Ingestion failed: {str(e)}[/red]")
            sys.exit(1)
        finally:
            await kse.disconnect()
    
    asyncio.run(ingest_products())


@cli.command()
@click.option(
    "--queries",
    type=click.Path(exists=True),
    help="Path to queries JSON file"
)
@click.option(
    "--iterations",
    type=int,
    default=3,
    help="Number of iterations per query"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Save benchmark results to file"
)
def benchmark(queries: Optional[str], iterations: int, output: Optional[str]):
    """
    📊 Run performance benchmarks
    
    Compare hybrid AI search performance against vector-only
    baselines across multiple metrics.
    
    Examples:
        kse benchmark                           # Use default queries
        kse benchmark --queries my-queries.json # Custom queries
        kse benchmark --iterations 5 --output results.json
    """
    console.print("[cyan]Running performance benchmarks...[/cyan]")
    
    async def run_benchmark():
        from .quickstart.benchmark import BenchmarkRunner
        
        # Load test queries
        if queries:
            with open(queries, 'r') as f:
                test_queries = json.load(f)
        else:
            # Use default queries
            datasets = SampleDatasets()
            test_queries = [q["query"] for q in datasets.get_sample_queries("retail")[:3]]
        
        # Initialize KSE Memory for benchmarking
        kse_config = KSEConfig()
        kse = KSEMemory(kse_config)
        
        try:
            await kse.initialize("generic", {})
            
            # Load sample products for benchmarking
            datasets = SampleDatasets()
            products = datasets.get_retail_products()
            
            # Add products to memory
            with Progress(console=console) as progress:
                task = progress.add_task("Loading products...", total=len(products))
                for product in products:
                    await kse.add_product(product)
                    progress.advance(task)
            
            # Run benchmark
            runner = BenchmarkRunner()
            results = await runner.run_comparison(
                products=products,
                kse_memory=kse,
                queries=test_queries,
                iterations=iterations
            )
            
            # Display results
            _display_benchmark_results(results)
            
            if output:
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                console.print(f"[green]✓[/green] Results saved to {output}")
            
        except Exception as e:
            console.print(f"[red]❌ Benchmark failed: {str(e)}[/red]")
            sys.exit(1)
        finally:
            await kse.disconnect()
    
    asyncio.run(run_benchmark())


@cli.command()
def status():
    """
    📋 Show KSE Memory system status
    
    Display current configuration, connected backends,
    and system health information.
    """
    console.print("[cyan]KSE Memory System Status[/cyan]")
    
    # Display version and basic info
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Version", "1.0.0")
    table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    table.add_row("Platform", sys.platform)
    
    console.print(table)
    
    # Check for configuration file
    config_paths = [
        Path.cwd() / "kse-config.yaml",
        Path.cwd() / "kse-config.json",
        Path.home() / ".kse" / "config.yaml"
    ]
    
    config_found = False
    for path in config_paths:
        if path.exists():
            console.print(f"[green]✓[/green] Configuration found: {path}")
            config_found = True
            break
    
    if not config_found:
        console.print("[yellow]⚠[/yellow] No configuration file found")
        console.print("[dim]Run 'kse init' to create one[/dim]")


def _display_config_summary(config: KSEConfig, adapter: str):
    """Display configuration summary."""
    console.print("\n[bold]Configuration Summary[/bold]")
    
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Component", style="cyan")
    table.add_column("Backend", style="white")
    table.add_column("Status", style="green")
    
    table.add_row("Platform Adapter", adapter, "✓ Selected")
    table.add_row("Vector Store", config.vector_store.get("backend", "default"), "✓ Configured")
    table.add_row("Graph Store", config.graph_store.get("backend", "default"), "✓ Configured")
    table.add_row("Concept Store", config.concept_store.get("backend", "default"), "✓ Configured")
    table.add_row("Embedding Model", config.embedding.get("text_model", "default"), "✓ Ready")
    
    console.print(table)


def _display_search_results(results, query: str, search_type: str):
    """Display search results in a formatted table."""
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    console.print(f"\n[bold]Search Results for '{query}' ({search_type})[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", width=4)
    table.add_column("Product", width=50)
    table.add_column("Score", width=8)
    table.add_column("Price", width=10)
    table.add_column("Category", width=20)
    
    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            result.product.title[:47] + "..." if len(result.product.title) > 50 else result.product.title,
            f"{result.score:.3f}",
            f"${result.product.price:.2f}" if result.product.price else "N/A",
            result.product.category or "Unknown"
        )
    
    console.print(table)
    
    # Display summary
    avg_score = sum(r.score for r in results) / len(results)
    console.print(f"\n[dim]Found {len(results)} results with average relevance score: {avg_score:.3f}[/dim]")


def _display_benchmark_results(results: Dict[str, Any]):
    """Display benchmark results."""
    console.print("\n[bold]📊 Benchmark Results[/bold]")
    
    # Summary table
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Vector-Only", style="red")
    table.add_column("KSE Hybrid", style="green")
    table.add_column("Improvement", style="bold green")
    
    table.add_row(
        "Relevance Score",
        f"{results.get('vector_only_score', 0):.3f}",
        f"{results.get('hybrid_score', 0):.3f}",
        f"+{results.get('improvement_percentage', 0):.1f}%"
    )
    
    table.add_row(
        "Avg Latency",
        f"{results.get('vector_only_latency', 0):.1f}ms",
        f"{results.get('hybrid_latency', 0):.1f}ms",
        f"{results.get('latency_change', 0):+.1f}ms"
    )
    
    console.print(table)
    
    # Individual query results
    if "individual_results" in results:
        console.print("\n[bold]Individual Query Results[/bold]")
        
        query_table = Table(show_header=True, header_style="bold magenta")
        query_table.add_column("Query", width=30)
        query_table.add_column("Vector Score", width=12)
        query_table.add_column("Hybrid Score", width=12)
        query_table.add_column("Improvement", width=12)
        
        for result in results["individual_results"]:
            query_table.add_row(
                result["query"][:27] + "..." if len(result["query"]) > 30 else result["query"],
                f"{result['vector_score']:.3f}",
                f"{result['hybrid_score']:.3f}",
                f"+{result['improvement']:.1f}%"
            )
        
        console.print(query_table)


if __name__ == "__main__":
    cli()