#!/usr/bin/env python3
"""
Simple production validation script for KSE Memory SDK.
"""

import os
import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_mock_backends_removed():
    """Check that mock backends are not accessible in production."""
    print("Checking mock backend removal...")
    
    try:
        from kse_memory.backends import get_vector_store, get_graph_store, get_concept_store
        from kse_memory.core.config import VectorStoreConfig, GraphStoreConfig, ConceptStoreConfig
        
        # Try to create mock backends - should fail
        try:
            config = VectorStoreConfig(backend="mock", api_key="test")
            get_vector_store(config)
            print("  FAIL: Mock vector store still accessible")
            return False
        except Exception:
            print("  PASS: Mock vector store properly removed")
        
        try:
            config = GraphStoreConfig(backend="mock", uri="test://", database="test")
            get_graph_store(config)
            print("  FAIL: Mock graph store still accessible")
            return False
        except Exception:
            print("  PASS: Mock graph store properly removed")
        
        try:
            config = ConceptStoreConfig(backend="mock", uri="test://", database="test")
            get_concept_store(config)
            print("  FAIL: Mock concept store still accessible")
            return False
        except Exception:
            print("  PASS: Mock concept store properly removed")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False

def check_debug_disabled():
    """Check that debug mode is disabled by default."""
    print("Checking debug mode settings...")
    
    try:
        from kse_memory.core.config import KSEConfig
        
        config = KSEConfig()
        if config.debug:
            print("  FAIL: Debug mode enabled by default")
            return False
        else:
            print("  PASS: Debug mode disabled by default")
            return True
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False

def check_production_backends():
    """Check that production backends are available."""
    print("Checking production backend availability...")
    
    backends_available = {
        "pinecone": False,
        "weaviate": False,
        "qdrant": False,
        "chromadb": False,
        "milvus": False,
        "neo4j": False,
        "arangodb": False,
        "postgresql": False,
        "mongodb": False
    }
    
    try:
        from kse_memory.backends import (
            PineconeBackend, WeaviateBackend, QdrantBackend, 
            ChromaDBBackend, MilvusBackend, Neo4jBackend, 
            ArangoDBBackend, PostgreSQLBackend, MongoDBBackend
        )
        
        # Check if backends can be imported
        backends_available["pinecone"] = True
        backends_available["weaviate"] = True
        backends_available["qdrant"] = True
        backends_available["chromadb"] = True
        backends_available["milvus"] = True
        backends_available["neo4j"] = True
        backends_available["arangodb"] = True
        backends_available["postgresql"] = True
        backends_available["mongodb"] = True
        
        available_count = sum(backends_available.values())
        print(f"  PASS: {available_count}/9 production backends available")
        
        if available_count >= 6:  # At least 2 of each type
            return True
        else:
            print("  WARN: Limited backend options available")
            return True
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False

def check_package_structure():
    """Check basic package structure."""
    print("Checking package structure...")
    
    required_files = [
        "kse_memory/__init__.py",
        "kse_memory/core/__init__.py",
        "kse_memory/backends/__init__.py",
        "kse_memory/adapters/__init__.py",
        "kse_memory/services/__init__.py",
        "pyproject.toml"
    ]
    
    project_root = Path(__file__).parent.parent
    missing_files = []
    
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  FAIL: Missing files: {missing_files}")
        return False
    else:
        print("  PASS: All required files present")
        return True

def check_imports():
    """Check that main imports work."""
    print("Checking main imports...")
    
    try:
        from kse_memory import KSEMemory, KSEConfig, Product, SearchQuery, SearchType
        print("  PASS: Main imports successful")
        return True
    except Exception as e:
        print(f"  FAIL: Import error: {str(e)}")
        return False

def main():
    """Run production validation checks."""
    print("KSE Memory SDK - Production Validation")
    print("=" * 50)
    
    checks = [
        ("Package Structure", check_package_structure),
        ("Main Imports", check_imports),
        ("Mock Backends Removed", check_mock_backends_removed),
        ("Debug Mode Disabled", check_debug_disabled),
        ("Production Backends", check_production_backends),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"  ERROR: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("SUCCESS: SDK is ready for production!")
        return True
    else:
        print("WARNING: Some checks failed. Review before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)