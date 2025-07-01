#!/usr/bin/env python3
"""
Comprehensive stress test for the universal Entity model.
Tests performance, edge cases, and robustness across all supported domains.
"""

import sys
import time
import warnings
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import json

# Add the kse_memory package to the path
sys.path.insert(0, str(Path(__file__).parent))

from kse_memory.core.models import (
    Entity, 
    ConceptualSpace, 
    Product,  # Deprecated
    ConceptualDimensions,  # Deprecated
    SearchResult
)

class StressTestResults:
    """Track stress test results and metrics."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
        self.performance_metrics = {}
        self.warnings_captured = []
    
    def add_test_result(self, test_name: str, passed: bool, error: str = None, duration: float = 0):
        """Add a test result."""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
            if error:
                self.errors.append(f"{test_name}: {error}")
        
        self.performance_metrics[test_name] = duration
    
    def add_warning(self, warning_msg: str):
        """Add a captured warning."""
        self.warnings_captured.append(warning_msg)
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("STRESS TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        print(f"Warnings Captured: {len(self.warnings_captured)}")
        
        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        print(f"\nPERFORMANCE METRICS:")
        for test_name, duration in self.performance_metrics.items():
            print(f"  {test_name}: {duration:.4f}s")
        
        total_time = sum(self.performance_metrics.values())
        print(f"\nTotal Execution Time: {total_time:.4f}s")

def stress_test_entity_creation(results: StressTestResults):
    """Stress test entity creation across all domains."""
    print("\n=== STRESS TEST: Entity Creation ===")
    
    domains = [
        ("healthcare", "patient", Entity.create_healthcare_entity),
        ("finance", "asset", Entity.create_finance_entity),
        ("real_estate", "property", Entity.create_real_estate_entity),
        ("enterprise", "document", Entity.create_enterprise_entity),
        ("research", "research_paper", Entity.create_research_entity),
        ("retail", "product", Entity.create_retail_entity),
    ]
    
    # Test creating many entities quickly
    for domain_name, entity_type, factory_method in domains:
        start_time = time.time()
        try:
            entities = []
            for i in range(100):  # Create 100 entities per domain
                entity = factory_method(
                    title=f"{domain_name.title()} Entity {i}",
                    description=f"Stress test {entity_type} #{i}",
                    entity_type=entity_type
                )
                entities.append(entity)
            
            # Verify all entities were created correctly
            assert len(entities) == 100
            assert all(e.entity_type == entity_type for e in entities)
            assert all(e.conceptual_space is not None for e in entities)
            
            duration = time.time() - start_time
            results.add_test_result(f"create_100_{domain_name}_entities", True, duration=duration)
            print(f"  [PASS] Created 100 {domain_name} entities in {duration:.4f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_test_result(f"create_100_{domain_name}_entities", False, str(e), duration)
            print(f"  [FAIL] Failed to create {domain_name} entities: {e}")

def stress_test_domain_metadata(results: StressTestResults):
    """Stress test domain metadata operations."""
    print("\n=== STRESS TEST: Domain Metadata ===")
    
    start_time = time.time()
    try:
        entity = Entity.create_healthcare_entity(
            title="Metadata Stress Test",
            description="Testing metadata performance"
        )
        
        # Set large amounts of metadata
        for i in range(50):
            entity.set_domain_metadata(f"domain_{i}", 
                                     field1=f"value_{i}",
                                     field2=i * 100,
                                     field3=[f"item_{j}" for j in range(10)],
                                     field4={"nested": {"data": f"nested_value_{i}"}})
        
        # Verify all metadata was stored
        for i in range(50):
            metadata = entity.get_domain_metadata(f"domain_{i}")
            assert metadata["field1"] == f"value_{i}"
            assert metadata["field2"] == i * 100
            assert len(metadata["field3"]) == 10
            assert metadata["field4"]["nested"]["data"] == f"nested_value_{i}"
        
        duration = time.time() - start_time
        results.add_test_result("domain_metadata_stress", True, duration=duration)
        print(f"  [PASS] Handled 50 domains with complex metadata in {duration:.4f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_test_result("domain_metadata_stress", False, str(e), duration)
        print(f"  [FAIL] Domain metadata stress test failed: {e}")

def stress_test_serialization(results: StressTestResults):
    """Stress test serialization/deserialization."""
    print("\n=== STRESS TEST: Serialization ===")
    
    start_time = time.time()
    try:
        # Create complex entities with lots of data
        entities = []
        for i in range(20):
            entity = Entity.create_finance_entity(
                title=f"Complex Entity {i}",
                description="A" * 1000,  # Large description
                entity_type="complex_asset"
            )
            
            # Add lots of metadata
            entity.set_domain_metadata("finance",
                                     ticker=f"TICK{i}",
                                     price=100.50 + i,
                                     historical_data=[j * 1.1 for j in range(100)],
                                     complex_data={"level1": {"level2": {"level3": f"deep_value_{i}"}}})
            
            # Add tags and media
            entity.tags = [f"tag_{j}" for j in range(20)]
            entity.media = [f"media_file_{j}.jpg" for j in range(10)]
            entity.variations = [{"variant": j, "data": f"variant_data_{j}"} for j in range(5)]
            
            entities.append(entity)
        
        # Serialize all entities
        serialized_data = []
        for entity in entities:
            serialized_data.append(entity.to_dict())
        
        # Deserialize all entities
        deserialized_entities = []
        for data in serialized_data:
            deserialized_entities.append(Entity.from_dict(data))
        
        # Verify integrity
        for original, deserialized in zip(entities, deserialized_entities):
            assert original.title == deserialized.title
            assert original.description == deserialized.description
            assert original.entity_type == deserialized.entity_type
            assert len(original.tags) == len(deserialized.tags)
            assert len(original.media) == len(deserialized.media)
            assert len(original.variations) == len(deserialized.variations)
            
            # Check domain metadata
            orig_finance = original.get_domain_metadata("finance")
            deser_finance = deserialized.get_domain_metadata("finance")
            assert orig_finance["ticker"] == deser_finance["ticker"]
            assert orig_finance["price"] == deser_finance["price"]
            assert len(orig_finance["historical_data"]) == len(deser_finance["historical_data"])
        
        duration = time.time() - start_time
        results.add_test_result("serialization_stress", True, duration=duration)
        print(f"  [PASS] Serialized/deserialized 20 complex entities in {duration:.4f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_test_result("serialization_stress", False, str(e), duration)
        print(f"  [FAIL] Serialization stress test failed: {e}")

def stress_test_conceptual_spaces(results: StressTestResults):
    """Stress test conceptual space operations."""
    print("\n=== STRESS TEST: Conceptual Spaces ===")
    
    start_time = time.time()
    try:
        # Create many conceptual spaces
        spaces = []
        domains = ["healthcare", "finance", "real_estate", "enterprise", "research", "retail"]
        
        for _ in range(100):  # Create 100 of each domain
            for domain in domains:
                space = ConceptualSpace.create_for_domain(domain)
                spaces.append(space)
        
        # Verify all spaces were created correctly
        assert len(spaces) == 600  # 100 * 6 domains
        
        # Test space operations
        for space in spaces[:50]:  # Test subset for performance
            # Test dimension access
            dims = list(space.dimensions.keys())
            assert len(dims) > 0
            
            # Test dimension values
            for dim_name in dims:
                assert dim_name in space.dimensions
                assert isinstance(space.dimensions[dim_name], (int, float))
        
        duration = time.time() - start_time
        results.add_test_result("conceptual_spaces_stress", True, duration=duration)
        print(f"  [PASS] Created and tested 600 conceptual spaces in {duration:.4f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_test_result("conceptual_spaces_stress", False, str(e), duration)
        print(f"  [FAIL] Conceptual spaces stress test failed: {e}")

def stress_test_backward_compatibility(results: StressTestResults):
    """Stress test backward compatibility features."""
    print("\n=== STRESS TEST: Backward Compatibility ===")
    
    start_time = time.time()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create many deprecated Product instances
            products = []
            for i in range(50):
                product = Product(
                    title=f"Legacy Product {i}",
                    description=f"Backward compatibility test {i}"
                )
                product.price = 99.99 + i
                product.currency = "USD"
                products.append(product)
            
            # Test deprecated ConceptualDimensions
            old_dims = []
            for i in range(20):
                old_dim = ConceptualDimensions()
                new_space = old_dim.to_conceptual_space()
                old_dims.append((old_dim, new_space))
            
            # Test SearchResult backward compatibility
            search_results = []
            for i in range(30):
                entity = Entity.create_retail_entity(
                    title=f"Search Entity {i}",
                    description=f"Search test {i}"
                )
                result = SearchResult(entity=entity, score=0.9 + i * 0.001)
                
                # Test deprecated product property
                legacy_product = result.product
                assert legacy_product.title == entity.title
                search_results.append(result)
            
            # Verify warnings were generated
            assert len(w) > 0  # Should have deprecation warnings
            
            # Store warning count
            results.add_warning(f"Generated {len(w)} deprecation warnings as expected")
        
        duration = time.time() - start_time
        results.add_test_result("backward_compatibility_stress", True, duration=duration)
        print(f"  [PASS] Tested backward compatibility with {len(w)} warnings in {duration:.4f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_test_result("backward_compatibility_stress", False, str(e), duration)
        print(f"  [FAIL] Backward compatibility stress test failed: {e}")

def stress_test_concurrent_operations(results: StressTestResults):
    """Stress test concurrent entity operations."""
    print("\n=== STRESS TEST: Concurrent Operations ===")
    
    def create_entity_batch(batch_id: int, domain: str) -> List[Entity]:
        """Create a batch of entities in a thread."""
        entities = []
        factory_methods = {
            "healthcare": Entity.create_healthcare_entity,
            "finance": Entity.create_finance_entity,
            "real_estate": Entity.create_real_estate_entity,
            "enterprise": Entity.create_enterprise_entity,
            "research": Entity.create_research_entity,
            "retail": Entity.create_retail_entity,
        }
        
        factory = factory_methods[domain]
        for i in range(20):
            entity = factory(
                title=f"Concurrent {domain} Entity {batch_id}-{i}",
                description=f"Concurrent test entity from batch {batch_id}"
            )
            entity.set_domain_metadata(domain, batch_id=batch_id, entity_num=i)
            entities.append(entity)
        
        return entities
    
    start_time = time.time()
    try:
        # Run concurrent entity creation
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            domains = ["healthcare", "finance", "real_estate", "enterprise", "research", "retail"]
            
            # Submit 5 batches per domain (30 total batches)
            for batch_id in range(5):
                for domain in domains:
                    future = executor.submit(create_entity_batch, batch_id, domain)
                    futures.append((future, domain, batch_id))
            
            # Collect results
            all_entities = []
            for future, domain, batch_id in futures:
                try:
                    entities = future.result(timeout=10)
                    all_entities.extend(entities)
                except Exception as e:
                    raise Exception(f"Batch {batch_id} for {domain} failed: {e}")
        
        # Verify results
        assert len(all_entities) == 600  # 5 batches * 6 domains * 20 entities
        
        # Verify entity integrity
        for entity in all_entities[:50]:  # Sample check
            assert entity.title is not None
            assert entity.description is not None
            assert entity.conceptual_space is not None
        
        duration = time.time() - start_time
        results.add_test_result("concurrent_operations_stress", True, duration=duration)
        print(f"  [PASS] Created 600 entities concurrently in {duration:.4f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_test_result("concurrent_operations_stress", False, str(e), duration)
        print(f"  [FAIL] Concurrent operations stress test failed: {e}")

def stress_test_edge_cases(results: StressTestResults):
    """Test edge cases and error conditions."""
    print("\n=== STRESS TEST: Edge Cases ===")
    
    start_time = time.time()
    try:
        # Test empty/minimal entities
        minimal_entity = Entity(title="", description="")
        assert minimal_entity.id is not None  # Should auto-generate
        
        # Test very long strings
        long_title = "A" * 10000
        long_desc = "B" * 50000
        large_entity = Entity.create_healthcare_entity(
            title=long_title,
            description=long_desc
        )
        assert len(large_entity.title) == 10000
        assert len(large_entity.description) == 50000
        
        # Test special characters
        special_entity = Entity.create_finance_entity(
            title="Special chars: 你好 🚀 ñáéíóú",
            description="Unicode test: αβγδε ∑∏∆ ♠♣♥♦"
        )
        assert "你好" in special_entity.title
        assert "αβγδε" in special_entity.description
        
        # Test None values in metadata
        entity_with_nones = Entity.create_enterprise_entity(
            title="None Test",
            description="Testing None values"
        )
        entity_with_nones.set_domain_metadata("enterprise",
                                            none_field=None,
                                            empty_list=[],
                                            empty_dict={})
        metadata = entity_with_nones.get_domain_metadata("enterprise")
        assert metadata["none_field"] is None
        assert metadata["empty_list"] == []
        assert metadata["empty_dict"] == {}
        
        # Test serialization of edge cases
        edge_cases = [minimal_entity, large_entity, special_entity, entity_with_nones]
        for entity in edge_cases:
            serialized = entity.to_dict()
            deserialized = Entity.from_dict(serialized)
            assert deserialized.title == entity.title
            assert deserialized.description == entity.description
        
        duration = time.time() - start_time
        results.add_test_result("edge_cases_stress", True, duration=duration)
        print(f"  [PASS] Handled edge cases successfully in {duration:.4f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_test_result("edge_cases_stress", False, str(e), duration)
        print(f"  [FAIL] Edge cases stress test failed: {e}")

def main():
    """Run comprehensive stress tests."""
    print("KSE-SDK Universal Data Model - COMPREHENSIVE STRESS TEST")
    print("="*60)
    
    results = StressTestResults()
    
    try:
        # Run all stress tests
        stress_test_entity_creation(results)
        stress_test_domain_metadata(results)
        stress_test_serialization(results)
        stress_test_conceptual_spaces(results)
        stress_test_backward_compatibility(results)
        stress_test_concurrent_operations(results)
        stress_test_edge_cases(results)
        
        # Print comprehensive results
        results.print_summary()
        
        # Determine overall success
        if results.tests_failed == 0:
            print(f"\n[SUCCESS] ALL STRESS TESTS PASSED! [SUCCESS]")
            print(f"The universal data model is robust and ready for production.")
            return 0
        else:
            print(f"\n[WARNING]  {results.tests_failed} STRESS TESTS FAILED")
            print(f"Review errors above before proceeding to production.")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] CRITICAL ERROR during stress testing: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())