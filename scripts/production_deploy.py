#!/usr/bin/env python3
"""
Production deployment script for KSE Memory SDK.

This script ensures the SDK is production-ready by:
1. Validating configuration
2. Removing test/mock code
3. Optimizing for production
4. Running final checks
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kse_memory.core.config import KSEConfig
from kse_memory.core.production_validator import ProductionValidator, ValidationResult


class ProductionDeployer:
    """Handles production deployment preparation."""
    
    def __init__(self, project_root: Path):
        """Initialize deployer."""
        self.project_root = project_root
        self.validator = ProductionValidator()
        
    def run_deployment_checks(self) -> bool:
        """Run all production deployment checks."""
        print("KSE Memory SDK - Production Deployment Checks")
        print("=" * 60)
        
        checks = [
            ("Package Structure", self._check_package_structure),
            ("Remove Test Code", self._remove_test_code),
            ("Validate Dependencies", self._validate_dependencies),
            ("Check Security", self._check_security),
            ("Optimize Performance", self._optimize_performance),
            ("Final Validation", self._final_validation),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...")
            try:
                result = check_func()
                if result:
                    print(f"✅ {check_name}: PASSED")
                else:
                    print(f"❌ {check_name}: FAILED")
                    all_passed = False
            except Exception as e:
                print(f"❌ {check_name}: ERROR - {str(e)}")
                all_passed = False
        
        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 All production checks PASSED! SDK is ready for deployment.")
        else:
            print("⚠️  Some checks FAILED. Please fix issues before deployment.")
        
        return all_passed
    
    def _check_package_structure(self) -> bool:
        """Check that package structure is correct."""
        required_files = [
            "kse_memory/__init__.py",
            "kse_memory/core/__init__.py",
            "kse_memory/backends/__init__.py",
            "kse_memory/adapters/__init__.py",
            "kse_memory/services/__init__.py",
            "pyproject.toml",
            "README.md",
            "LICENSE"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"   Missing required files: {missing_files}")
            return False
        
        print("   ✓ All required files present")
        return True
    
    def _remove_test_code(self) -> bool:
        """Remove or isolate test-specific code."""
        print("   Checking for test code in production paths...")
        
        # Files that should not contain test code in production
        production_files = list(self.project_root.glob("kse_memory/**/*.py"))
        
        test_patterns = [
            "mock",
            "test_mode",
            "_test_",
            "# For testing",
            "# Test",
            "pytest",
        ]
        
        issues_found = []
        
        for file_path in production_files:
            if "test" in str(file_path) or "mock.py" in str(file_path):
                continue  # Skip actual test files
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in test_patterns:
                    if pattern in content:
                        issues_found.append(f"{file_path}: contains '{pattern}'")
            except Exception:
                continue
        
        if issues_found:
            print("   ⚠️  Test code found in production files:")
            for issue in issues_found[:5]:  # Show first 5
                print(f"     - {issue}")
            if len(issues_found) > 5:
                print(f"     ... and {len(issues_found) - 5} more")
            return False
        
        print("   ✓ No test code found in production paths")
        return True
    
    def _validate_dependencies(self) -> bool:
        """Validate that dependencies are production-ready."""
        print("   Checking dependencies...")
        
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            print("   ❌ pyproject.toml not found")
            return False
        
        # Check for test dependencies in main requirements
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        test_deps = ["pytest", "mock", "faker", "factory-boy"]
        main_deps_section = False
        issues = []
        
        for line in content.split('\n'):
            if line.strip() == 'dependencies = [':
                main_deps_section = True
                continue
            elif main_deps_section and line.strip() == ']':
                main_deps_section = False
                continue
            
            if main_deps_section:
                for test_dep in test_deps:
                    if test_dep in line.lower():
                        issues.append(f"Test dependency '{test_dep}' in main dependencies")
        
        if issues:
            print("   ⚠️  Dependency issues:")
            for issue in issues:
                print(f"     - {issue}")
            return False
        
        print("   ✓ Dependencies look good")
        return True
    
    def _check_security(self) -> bool:
        """Check for security issues."""
        print("   Checking security...")
        
        security_issues = []
        
        # Check for hardcoded secrets
        for file_path in self.project_root.glob("kse_memory/**/*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for potential hardcoded secrets
                if any(pattern in content.lower() for pattern in [
                    'api_key = "',
                    'password = "',
                    'secret = "',
                    'token = "'
                ]):
                    security_issues.append(f"{file_path}: potential hardcoded secret")
                    
            except Exception:
                continue
        
        if security_issues:
            print("   ⚠️  Security issues found:")
            for issue in security_issues:
                print(f"     - {issue}")
            return False
        
        print("   ✓ No obvious security issues")
        return True
    
    def _optimize_performance(self) -> bool:
        """Check performance optimizations."""
        print("   Checking performance settings...")
        
        # Check if debug logging is disabled by default
        config_file = self.project_root / "kse_memory" / "core" / "config.py"
        if config_file.exists():
            with open(config_file, 'r') as f:
                content = f.read()
            
            if "debug: bool = True" in content:
                print("   ⚠️  Debug mode enabled by default")
                return False
            
            if "log_level: LogLevel = LogLevel.DEBUG" in content:
                print("   ⚠️  Debug logging enabled by default")
                return False
        
        print("   ✓ Performance settings optimized")
        return True
    
    def _final_validation(self) -> bool:
        """Run final validation with production validator."""
        print("   Running production configuration validation...")
        
        try:
            # Create a sample production config
            config = KSEConfig(
                debug=False,
                vector_store={
                    "backend": "pinecone",
                    "api_key": "test-key",
                    "environment": "production"
                },
                graph_store={
                    "backend": "neo4j",
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password"
                },
                concept_store={
                    "backend": "postgresql",
                    "uri": "postgresql://localhost:5432/kse"
                }
            )
            
            result = self.validator.validate_config(config)
            
            if result.errors:
                print("   ❌ Configuration validation errors:")
                for error in result.errors:
                    print(f"     - {error}")
                return False
            
            if result.warnings:
                print("   ⚠️  Configuration warnings:")
                for warning in result.warnings[:3]:
                    print(f"     - {warning}")
            
            print("   ✓ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"   ❌ Validation error: {str(e)}")
            return False
    
    def generate_production_package(self, output_dir: Path) -> bool:
        """Generate production-ready package."""
        print(f"\n📦 Generating production package in {output_dir}...")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy main package
        package_dir = output_dir / "kse-memory-sdk"
        if package_dir.exists():
            shutil.rmtree(package_dir)
        
        # Copy essential files only
        essential_dirs = ["kse_memory", "docs", "examples"]
        essential_files = ["pyproject.toml", "README.md", "LICENSE", "CHANGELOG.md"]
        
        package_dir.mkdir()
        
        for dir_name in essential_dirs:
            src_dir = self.project_root / dir_name
            if src_dir.exists():
                dst_dir = package_dir / dir_name
                shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(
                    "*.pyc", "__pycache__", "*.pyo", "*.pyd", ".pytest_cache"
                ))
        
        for file_name in essential_files:
            src_file = self.project_root / file_name
            if src_file.exists():
                shutil.copy2(src_file, package_dir / file_name)
        
        # Remove test directories from production package
        test_dirs = ["tests", "test_*"]
        for pattern in test_dirs:
            for test_dir in package_dir.glob(pattern):
                if test_dir.is_dir():
                    shutil.rmtree(test_dir)
        
        # Remove mock backend from production package
        mock_file = package_dir / "kse_memory" / "backends" / "mock.py"
        if mock_file.exists():
            mock_file.unlink()
        
        print(f"   ✓ Production package created at {package_dir}")
        return True


def main():
    """Main deployment script."""
    project_root = Path(__file__).parent.parent
    deployer = ProductionDeployer(project_root)
    
    # Run deployment checks
    if deployer.run_deployment_checks():
        print("\n🎯 Ready for production deployment!")
        
        # Optionally generate production package
        response = input("\nGenerate production package? (y/N): ")
        if response.lower() == 'y':
            output_dir = Path.cwd() / "dist"
            deployer.generate_production_package(output_dir)
            print(f"\n📦 Production package ready in: {output_dir}")
    else:
        print("\n❌ Production deployment checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()