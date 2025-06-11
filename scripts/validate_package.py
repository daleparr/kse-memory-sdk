"""
KSE Memory SDK - Package Validation Script

Validates the package is ready for PyPI publication.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a required file exists."""
    if os.path.exists(filepath):
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[FAIL] {description}: {filepath} - NOT FOUND")
        return False


def check_pyproject_toml():
    """Validate pyproject.toml configuration."""
    print("\n📦 Checking pyproject.toml...")
    
    if not check_file_exists("pyproject.toml", "Build configuration"):
        return False
    
    try:
        import tomli
        with open("pyproject.toml", "rb") as f:
            config = tomli.load(f)
        
        # Check required fields
        project = config.get("project", {})
        required_fields = ["name", "version", "description", "authors"]
        
        for field in required_fields:
            if field in project:
                print(f"[OK] Project {field}: {project[field]}")
            else:
                print(f"[FAIL] Missing required field: {field}")
                return False
        
        # Check dependencies
        deps = project.get("dependencies", [])
        print(f"[OK] Dependencies: {len(deps)} packages")
        
        # Check optional dependencies
        optional_deps = project.get("optional-dependencies", {})
        print(f"[OK] Optional dependencies: {len(optional_deps)} groups")
        
        return True
        
    except ImportError:
        print("[WARN]  tomli not available, skipping detailed validation")
        return True
    except Exception as e:
        print(f"[FAIL] Error reading pyproject.toml: {e}")
        return False


def check_core_files():
    """Check that all core files exist."""
    print("\n📁 Checking core files...")
    
    required_files = [
        ("README.md", "Main documentation"),
        ("LICENSE", "License file"),
        ("CHANGELOG.md", "Change log"),
        ("CONTRIBUTING.md", "Contributing guidelines"),
        ("kse_memory/__init__.py", "Main package init"),
        ("kse_memory/core/__init__.py", "Core module init"),
        ("kse_memory/core/memory.py", "Core memory class"),
        ("kse_memory/core/models.py", "Core models"),
        ("kse_memory/core/config.py", "Configuration"),
        ("kse_memory/cli.py", "CLI interface"),
    ]
    
    all_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def check_package_structure():
    """Check package structure is correct."""
    print("\n🏗️  Checking package structure...")
    
    required_dirs = [
        "kse_memory",
        "kse_memory/core",
        "kse_memory/services", 
        "kse_memory/backends",
        "kse_memory/adapters",
        "kse_memory/integrations",
        "kse_memory/visual",
        "kse_memory/quickstart",
        "tests",
        "examples",
        "docs"
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"[OK] Directory: {directory}")
        else:
            print(f"[FAIL] Missing directory: {directory}")
            all_exist = False
    
    return all_exist


def check_imports():
    """Check that main imports work."""
    print("\n Checking imports...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Test core imports
        from kse_memory import KSEMemory, KSEConfig
        print("[OK] Core imports: KSEMemory, KSEConfig")
        
        from kse_memory.core.models import Product, SearchQuery, SearchType
        print("[OK] Model imports: Product, SearchQuery, SearchType")
        
        from kse_memory.cli import cli
        print("[OK] CLI import: cli")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False


def check_cli_command():
    """Check that CLI command is available."""
    print("\n⚡ Checking CLI command...")
    
    try:
        # Check if kse command is available
        result = subprocess.run(
            [sys.executable, "-m", "kse_memory.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("[OK] CLI command works")
            return True
        else:
            print(f"[FAIL] CLI command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[FAIL] CLI command timed out")
        return False
    except Exception as e:
        print(f"[FAIL] CLI command error: {e}")
        return False


def check_tests():
    """Check that tests can be discovered."""
    print("\n🧪 Checking tests...")
    
    if not os.path.isdir("tests"):
        print("[FAIL] Tests directory not found")
        return False
    
    test_files = list(Path("tests").glob("test_*.py"))
    if len(test_files) == 0:
        print("[FAIL] No test files found")
        return False
    
    print(f"[OK] Found {len(test_files)} test files")
    
    # Check if pytest is available
    try:
        import pytest
        print("[OK] pytest is available")
        return True
    except ImportError:
        print("[WARN]  pytest not available, tests may not run")
        return True


def check_examples():
    """Check that examples exist and are valid."""
    print("\n📚 Checking examples...")
    
    if not os.path.isdir("examples"):
        print("[FAIL] Examples directory not found")
        return False
    
    example_files = list(Path("examples").glob("*.py"))
    if len(example_files) == 0:
        print("[FAIL] No example files found")
        return False
    
    print(f"[OK] Found {len(example_files)} example files")
    
    # Check key examples exist
    key_examples = [
        "examples/quickstart_demo.py",
        "examples/hybrid_retrieval_demo.py",
        "examples/multi_domain_visualization.py",
        "examples/langchain_integration.py"
    ]
    
    all_exist = True
    for example in key_examples:
        if os.path.exists(example):
            print(f"[OK] Key example: {example}")
        else:
            print(f"[FAIL] Missing key example: {example}")
            all_exist = False
    
    return all_exist


def check_documentation():
    """Check documentation completeness."""
    print("\n📖 Checking documentation...")
    
    if not os.path.isdir("docs"):
        print("[FAIL] Docs directory not found")
        return False
    
    doc_files = list(Path("docs").glob("*.md"))
    print(f"[OK] Found {len(doc_files)} documentation files")
    
    # Check key documentation exists
    key_docs = [
        "docs/API_REFERENCE.md",
        "docs/DOMAIN_ADAPTATIONS.md", 
        "docs/VISUAL_TOOLING_ROADMAP.md",
        "docs/GTM_LAUNCH_PLAN.md"
    ]
    
    all_exist = True
    for doc in key_docs:
        if os.path.exists(doc):
            print(f"[OK] Key documentation: {doc}")
        else:
            print(f"[FAIL] Missing key documentation: {doc}")
            all_exist = False
    
    return all_exist


def run_validation():
    """Run complete package validation."""
    print("KSE Memory SDK - Package Validation")
    print("=" * 50)
    
    checks = [
        ("Core Files", check_core_files),
        ("Package Structure", check_package_structure),
        ("PyProject Configuration", check_pyproject_toml),
        ("Import Validation", check_imports),
        ("CLI Command", check_cli_command),
        ("Test Discovery", check_tests),
        ("Examples", check_examples),
        ("Documentation", check_documentation),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"[FAIL] {name} check failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n[SUCCESS] Package validation PASSED! Ready for PyPI publication.")
        return True
    else:
        print(f"\n[WARN]  Package validation FAILED! {total - passed} issues need to be resolved.")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)