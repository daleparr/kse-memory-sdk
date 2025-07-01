"""
Backend Detection and Auto-Configuration for KSE Memory SDK.

Provides intelligent backend detection, user choice, and automatic setup
for the best possible quickstart experience.
"""

import os
import sys
import subprocess
import importlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class BackendInfo:
    """Information about a detected backend."""
    name: str
    display_name: str
    type: str  # "local", "cloud", "memory"
    cost: str  # "free", "free_tier", "paid"
    installed: bool = False
    available: bool = False
    configured: bool = False
    score: int = 0
    install_command: str = ""
    pros: List[str] = None
    cons: List[str] = None
    best_for: str = ""
    setup_complexity: str = "easy"  # "easy", "medium", "advanced"

    def __post_init__(self):
        if self.pros is None:
            self.pros = []
        if self.cons is None:
            self.cons = []


class BackendDetector:
    """Intelligent backend detection and recommendation system."""
    
    BACKEND_DEFINITIONS = {
        "chromadb": BackendInfo(
            name="chromadb",
            display_name="ChromaDB",
            type="local",
            cost="free",
            install_command="pip install chromadb",
            pros=["Completely free", "Local data control", "Persistent storage", "No API limits"],
            cons=["Local only", "Single machine", "Manual scaling"],
            best_for="Development, learning, small projects, data privacy",
            setup_complexity="easy"
        ),
        "weaviate": BackendInfo(
            name="weaviate",
            display_name="Weaviate",
            type="cloud",
            cost="free_tier",
            install_command="pip install weaviate-client",
            pros=["Cloud managed", "Scalable", "Free tier available", "GraphQL API"],
            cons=["Usage limits on free tier", "Internet required", "Vendor dependency"],
            best_for="Prototypes, demos, small production workloads",
            setup_complexity="medium"
        ),
        "qdrant": BackendInfo(
            name="qdrant",
            display_name="Qdrant",
            type="cloud",
            cost="free_tier",
            install_command="pip install qdrant-client",
            pros=["High performance", "Cloud managed", "Free tier", "Rust-based speed"],
            cons=["Usage limits on free tier", "Internet required", "Newer ecosystem"],
            best_for="Performance testing, production pilots, ML workloads",
            setup_complexity="medium"
        ),
        "memory": BackendInfo(
            name="memory",
            display_name="In-Memory",
            type="memory",
            cost="free",
            install_command="Built-in",
            pros=["Instant setup", "No dependencies", "Perfect for testing"],
            cons=["No persistence", "Lost on restart", "Development only"],
            best_for="Quick testing, CI/CD, temporary demos",
            setup_complexity="easy"
        )
    }
    
    def __init__(self):
        """Initialize the backend detector."""
        self.detected_backends: List[BackendInfo] = []
        self.recommended_backend: Optional[BackendInfo] = None
    
    def detect_all_backends(self) -> List[BackendInfo]:
        """Detect all available backends and rank them."""
        console.print("Scanning your environment for available backends...")
        
        backends = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Detecting backends...", total=len(self.BACKEND_DEFINITIONS))
            
            for name, backend_def in self.BACKEND_DEFINITIONS.items():
                backend = self._detect_backend(name, backend_def)
                backends.append(backend)
                progress.advance(task)
        
        # Sort by score (highest first)
        backends.sort(key=lambda x: x.score, reverse=True)
        
        self.detected_backends = backends
        self.recommended_backend = backends[0] if backends else None
        
        return backends
    
    def _detect_backend(self, name: str, backend_def: BackendInfo) -> BackendInfo:
        """Detect a specific backend and calculate its score."""
        backend = BackendInfo(**backend_def.__dict__)
        
        # Check if package is installed
        backend.installed = self._is_package_installed(self._get_package_name(name))
        
        # Check if backend is available (can be used)
        backend.available = backend.installed or name == "memory"
        
        # Check if backend is configured (has necessary env vars/config)
        backend.configured = self._is_backend_configured(name)
        
        # Calculate score
        backend.score = self._calculate_backend_score(backend)
        
        return backend
    
    def _is_package_installed(self, package_name: str) -> bool:
        """Check if a Python package is installed."""
        if package_name == "built-in":
            return True
        
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    def _get_package_name(self, backend_name: str) -> str:
        """Get the Python package name for a backend."""
        package_map = {
            "chromadb": "chromadb",
            "weaviate": "weaviate",
            "qdrant": "qdrant_client",
            "memory": "built-in"
        }
        return package_map.get(backend_name, backend_name)
    
    def _is_backend_configured(self, backend_name: str) -> bool:
        """Check if a backend has necessary configuration."""
        config_checks = {
            "chromadb": lambda: True,  # No config needed
            "weaviate": lambda: bool(os.getenv("WEAVIATE_URL") or os.getenv("WEAVIATE_API_KEY")),
            "qdrant": lambda: bool(os.getenv("QDRANT_URL") or os.getenv("QDRANT_API_KEY")),
            "memory": lambda: True  # No config needed
        }
        
        check_func = config_checks.get(backend_name, lambda: False)
        return check_func()
    
    def _calculate_backend_score(self, backend: BackendInfo) -> int:
        """Calculate a score for backend recommendation."""
        score = 0
        
        # Base scores by type
        type_scores = {
            "local": 100,    # Prefer local for ease of use
            "cloud": 80,     # Cloud is good but has dependencies
            "memory": 60     # Memory is last resort
        }
        score += type_scores.get(backend.type, 0)
        
        # Bonus for being installed
        if backend.installed:
            score += 50
        
        # Bonus for being configured
        if backend.configured:
            score += 30
        
        # Bonus for being free
        if backend.cost == "free":
            score += 20
        elif backend.cost == "free_tier":
            score += 10
        
        # Bonus for easy setup
        if backend.setup_complexity == "easy":
            score += 15
        elif backend.setup_complexity == "medium":
            score += 5
        
        return score
    
    def display_backend_options(self, backends: List[BackendInfo]) -> None:
        """Display available backend options in a nice table."""
        table = Table(title="Available KSE Memory Backends")
        
        table.add_column("Backend", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Cost", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Best For", style="blue")
        
        for backend in backends:
            # Status indicator
            if backend.installed and backend.configured:
                status = "Ready"
            elif backend.installed:
                status = "Needs Config"
            elif backend.available:
                status = "Available"
            else:
                status = "Not Available"
            
            table.add_row(
                backend.display_name,
                backend.type.title(),
                backend.cost.replace("_", " ").title(),
                status,
                backend.best_for
            )
        
        console.print(table)
    
    def get_user_choice(self, backends: List[BackendInfo]) -> BackendInfo:
        """Get user's backend choice through interactive prompt."""
        
        # Show recommendation
        if self.recommended_backend:
            console.print(Panel.fit(
                f"[bold green]Recommended:[/bold green] {self.recommended_backend.display_name}\n"
                f"[dim]{self.recommended_backend.best_for}[/dim]",
                border_style="green"
            ))
        
        # Create choice options
        choices = []
        choice_map = {}
        
        for i, backend in enumerate(backends, 1):
            status = "Ready" if backend.installed else "Available" if backend.available else "Not Available"
            choice_text = f"{backend.display_name} ({backend.type}, {backend.cost.replace('_', ' ')})"
            choices.append(f"[{i}] {status} - {choice_text}")
            choice_map[str(i)] = backend
        
        choices.append(f"[{len(backends) + 1}] Show detailed comparison")
        choices.append(f"[{len(backends) + 2}] I'll configure manually")
        
        console.print("\nWhich backend would you like to use?")
        for choice in choices:
            console.print(f"  {choice}")
        
        while True:
            choice = Prompt.ask(
                "\nEnter your choice",
                choices=[str(i) for i in range(1, len(choices) + 1)],
                default="1"
            )
            
            if choice in choice_map:
                return choice_map[choice]
            elif choice == str(len(backends) + 1):
                self._show_detailed_comparison(backends)
                continue
            elif choice == str(len(backends) + 2):
                console.print("📝 Manual configuration selected. Please edit your config file.")
                return None
    
    def _show_detailed_comparison(self, backends: List[BackendInfo]) -> None:
        """Show detailed comparison of backends."""
        for backend in backends:
            console.print(f"\n[bold cyan]{backend.display_name}[/bold cyan]")
            console.print(f"Type: {backend.type} | Cost: {backend.cost} | Setup: {backend.setup_complexity}")
            
            if backend.pros:
                console.print("[green]Pros:[/green]")
                for pro in backend.pros:
                    console.print(f"  + {pro}")
            
            if backend.cons:
                console.print("[red]Cons:[/red]")
                for con in backend.cons:
                    console.print(f"  - {con}")
            
            console.print(f"[blue]Best for:[/blue] {backend.best_for}")
            
            if not backend.installed and backend.install_command != "Built-in":
                console.print(f"[yellow]Install:[/yellow] {backend.install_command}")
    
    def install_backend(self, backend: BackendInfo) -> bool:
        """Install a backend if needed."""
        if backend.installed or backend.install_command == "Built-in":
            return True
        
        if not Confirm.ask(f"Install {backend.display_name}? ({backend.install_command})"):
            return False
        
        console.print(f"Installing {backend.display_name}...")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Installing {backend.display_name}...", total=None)
                
                # Run pip install
                result = subprocess.run(
                    backend.install_command.split(),
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                progress.update(task, description=f"✅ {backend.display_name} installed successfully!")
            
            backend.installed = True
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"❌ Failed to install {backend.display_name}: {e}")
            return False
    
    def generate_config(self, backend: BackendInfo, data_dir: str = "./kse_data") -> Dict[str, Any]:
        """Generate configuration for the chosen backend."""
        
        configs = {
            "chromadb": {
                "vector_store": {
                    "backend": "chromadb",
                    "persist_directory": f"{data_dir}/vectors",
                    "collection_name": "kse_products"
                },
                "graph_store": {
                    "backend": "memory"  # Start simple, can upgrade later
                },
                "concept_store": {
                    "backend": "memory"  # Start simple, can upgrade later
                },
                "embedding": {
                    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 32
                },
                "conceptual": {
                    "auto_compute": False  # No API keys needed
                },
                "cache": {
                    "enabled": True,
                    "backend": "memory"
                }
            },
            "weaviate": {
                "vector_store": {
                    "backend": "weaviate",
                    "url": "${WEAVIATE_URL:-https://your-cluster.weaviate.network}",
                    "api_key": "${WEAVIATE_API_KEY}",
                    "class_name": "KSEProducts"
                },
                "graph_store": {
                    "backend": "memory"
                },
                "concept_store": {
                    "backend": "memory"
                },
                "embedding": {
                    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 32
                },
                "conceptual": {
                    "auto_compute": False
                },
                "cache": {
                    "enabled": True,
                    "backend": "memory"
                }
            },
            "qdrant": {
                "vector_store": {
                    "backend": "qdrant",
                    "url": "${QDRANT_URL:-http://localhost:6333}",
                    "api_key": "${QDRANT_API_KEY}",
                    "collection_name": "kse_products"
                },
                "graph_store": {
                    "backend": "memory"
                },
                "concept_store": {
                    "backend": "memory"
                },
                "embedding": {
                    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 32
                },
                "conceptual": {
                    "auto_compute": False
                },
                "cache": {
                    "enabled": True,
                    "backend": "memory"
                }
            },
            "memory": {
                "vector_store": {
                    "backend": "memory"
                },
                "graph_store": {
                    "backend": "memory"
                },
                "concept_store": {
                    "backend": "memory"
                },
                "embedding": {
                    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 16  # Smaller for memory backend
                },
                "conceptual": {
                    "auto_compute": False
                },
                "cache": {
                    "enabled": False  # No caching for memory backend
                }
            }
        }
        
        base_config = configs.get(backend.name, configs["memory"])
        
        # Add common settings
        base_config.update({
            "app_name": "KSE Memory",
            "version": "1.1.0",
            "debug": False,
            "log_level": "INFO"
        })
        
        return base_config


def auto_detect_and_setup() -> Tuple[BackendInfo, Dict[str, Any]]:
    """Main function to auto-detect backends and set up configuration."""
    
    console.print(Panel.fit(
        "[bold blue]KSE Memory SDK - Smart Backend Setup[/bold blue]\n"
        "Finding the best backend for your environment...",
        border_style="blue"
    ))
    
    detector = BackendDetector()
    
    # Detect all backends
    backends = detector.detect_all_backends()
    
    if not backends:
        console.print("❌ No backends detected. Please install at least one backend.")
        return None, None
    
    # Show options
    detector.display_backend_options(backends)
    
    # Get user choice
    chosen_backend = detector.get_user_choice(backends)
    
    if not chosen_backend:
        return None, None
    
    # Install if needed
    if not detector.install_backend(chosen_backend):
        console.print("❌ Backend installation failed.")
        return None, None
    
    # Generate configuration
    config = detector.generate_config(chosen_backend)
    
    console.print(Panel.fit(
        f"✅ [bold green]Setup Complete![/bold green]\n"
        f"Backend: {chosen_backend.display_name}\n"
        f"Type: {chosen_backend.type}\n"
        f"Cost: {chosen_backend.cost.replace('_', ' ').title()}",
        border_style="green"
    ))
    
    return chosen_backend, config


if __name__ == "__main__":
    # Test the detector
    backend, config = auto_detect_and_setup()
    if backend and config:
        console.print("\n📋 Generated Configuration:")
        console.print(config)