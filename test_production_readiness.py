#!/usr/bin/env python3
"""
KSE Memory SDK - Production Readiness Test

This script tests the multi-backend auto-detection system and verifies
that the SDK is ready for AWS community sharing.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the SDK to the path
sys.path.insert(0, str(Path(__file__).parent / "kse-memory-sdk"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def test_backend_detection():
    """Test the backend detection system."""
    console.print("\nTesting Backend Detection System...")
    
    try:
        from kse_memory.quickstart.backend_detector import BackendDetector
        
        detector = BackendDetector()
        backends = detector.detect_all_backends()
        
        console.print(f"SUCCESS: Detected {len(backends)} backends")
        
        # Create results table
        table = Table(title="Backend Detection Results")
        table.add_column("Backend", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Type", style="magenta")
        
        for backend in backends:
            status = "Ready" if backend.installed else "Available" if backend.available else "Not Available"
            table.add_row(
                backend.display_name,
                status,
                str(backend.score),
                backend.type
            )
        
        console.print(table)
        return True
        
    except Exception as e:
        console.print(f"ERROR: Backend detection failed: {e}")
        return False


async def test_config_generation():
    """Test configuration generation."""
    console.print("\n⚙️ Testing Configuration Generation...")
    
    try:
        from kse_memory.quickstart.backend_detector import BackendDetector
        from kse_memory.core.config import KSEConfig
        
        detector = BackendDetector()
        
        # Test each backend type
        for backend_name in ["chromadb", "weaviate", "qdrant", "memory"]:
            backend_def = detector.BACKEND_DEFINITIONS[backend_name]
            config_dict = detector.generate_config(backend_def)
            
            # Verify config can be loaded
            config = KSEConfig.from_dict(config_dict)
            
            console.print(f"✅ {backend_def.display_name} config generated successfully")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Config generation failed: {e}")
        return False


async def test_quickstart_demo():
    """Test the quickstart demo with memory backend."""
    console.print("\n🚀 Testing Quickstart Demo...")
    
    try:
        from kse_memory.quickstart.demo import QuickstartDemo
        
        # Create demo with auto-setup disabled for testing
        demo = QuickstartDemo(auto_setup=False)
        
        # Manually set memory backend for testing
        from kse_memory.quickstart.backend_detector import BackendDetector
        detector = BackendDetector()
        memory_backend = detector.BACKEND_DEFINITIONS["memory"]
        demo.chosen_backend = memory_backend
        demo.config = detector.generate_config(memory_backend)
        
        # Run a quick demo
        results = await demo.run(
            demo_type="retail",
            open_browser=False,
            backend="memory"
        )
        
        console.print(f"✅ Demo completed successfully")
        console.print(f"   Backend: {results.get('backend', 'unknown')}")
        console.print(f"   Products: {results.get('products_loaded', 0)}")
        console.print(f"   Searches: {len(results.get('search_results', {}))}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Quickstart demo failed: {e}")
        return False


async def test_cli_commands():
    """Test CLI command structure."""
    console.print("\n💻 Testing CLI Commands...")
    
    try:
        from kse_memory.cli import cli
        
        # Check if commands are properly registered
        commands = list(cli.commands.keys())
        expected_commands = ["quickstart", "setup"]
        
        for cmd in expected_commands:
            if cmd in commands:
                console.print(f"✅ CLI command '{cmd}' registered")
            else:
                console.print(f"❌ CLI command '{cmd}' missing")
                return False
        
        return True
        
    except Exception as e:
        console.print(f"❌ CLI test failed: {e}")
        return False


async def test_environment_files():
    """Test environment file templates."""
    console.print("\n📁 Testing Environment Files...")
    
    try:
        # Check if .env.example exists
        env_example = Path("kse-memory-sdk/.env.example")
        if env_example.exists():
            console.print("✅ .env.example file exists")
            
            # Check if it contains key sections
            content = env_example.read_text()
            required_sections = [
                "PINECONE_API_KEY",
                "WEAVIATE_URL",
                "QDRANT_URL",
                "NEO4J_URI",
                "OPENAI_API_KEY"
            ]
            
            for section in required_sections:
                if section in content:
                    console.print(f"✅ {section} template found")
                else:
                    console.print(f"❌ {section} template missing")
                    return False
        else:
            console.print("❌ .env.example file missing")
            return False
        
        # Check quickstart guide
        quickstart_guide = Path("kse-memory-sdk/QUICKSTART_GUIDE.md")
        if quickstart_guide.exists():
            console.print("✅ QUICKSTART_GUIDE.md exists")
        else:
            console.print("❌ QUICKSTART_GUIDE.md missing")
            return False
        
        return True
        
    except Exception as e:
        console.print(f"❌ Environment files test failed: {e}")
        return False


async def test_docker_setup():
    """Test Docker configuration."""
    console.print("\n🐳 Testing Docker Setup...")
    
    try:
        # Check Docker files
        docker_files = [
            "kse-memory-sdk/docker-compose.community.yml",
            "kse-memory-sdk/Dockerfile.community"
        ]
        
        for docker_file in docker_files:
            if Path(docker_file).exists():
                console.print(f"✅ {Path(docker_file).name} exists")
            else:
                console.print(f"❌ {Path(docker_file).name} missing")
                return False
        
        return True
        
    except Exception as e:
        console.print(f"❌ Docker setup test failed: {e}")
        return False


async def main():
    """Run all production readiness tests."""
    console.print(Panel.fit(
        "[bold blue]KSE Memory SDK - Production Readiness Test[/bold blue]\n"
        "Verifying AWS community sharing readiness...",
        border_style="blue"
    ))
    
    tests = [
        ("Backend Detection", test_backend_detection),
        ("Configuration Generation", test_config_generation),
        ("Quickstart Demo", test_quickstart_demo),
        ("CLI Commands", test_cli_commands),
        ("Environment Files", test_environment_files),
        ("Docker Setup", test_docker_setup),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        console.print(f"\n{'='*60}")
        console.print(f"Running: {test_name}")
        console.print(f"{'='*60}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    console.print(f"\n{'='*60}")
    console.print("📊 PRODUCTION READINESS SUMMARY")
    console.print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    summary_table = Table(title="Test Results")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Status", style="green")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        summary_table.add_row(test_name, status)
    
    console.print(summary_table)
    
    if passed == total:
        console.print(Panel.fit(
            f"[bold green]🎉 ALL TESTS PASSED! ({passed}/{total})[/bold green]\n"
            "KSE Memory SDK is ready for AWS community sharing!",
            border_style="green"
        ))
        
        console.print("\n🚀 Ready for AWS Community:")
        console.print("  ✅ Multi-backend auto-detection")
        console.print("  ✅ Zero-config quickstart")
        console.print("  ✅ Free backend options")
        console.print("  ✅ Production-ready configuration")
        console.print("  ✅ Docker deployment")
        console.print("  ✅ Comprehensive documentation")
        
        return True
    else:
        console.print(Panel.fit(
            f"[bold red]❌ TESTS FAILED ({passed}/{total})[/bold red]\n"
            "Please fix the failing tests before AWS sharing.",
            border_style="red"
        ))
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)