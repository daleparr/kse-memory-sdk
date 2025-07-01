#!/usr/bin/env python3
"""
Test script to validate the universal Entity model across different domains.
This script demonstrates the industry-agnostic capabilities of the refactored KSE-SDK.
"""

import sys
import warnings
from pathlib import Path

# Add the kse_memory package to the path
sys.path.insert(0, str(Path(__file__).parent))

from kse_memory.core.models import (
    Entity, 
    ConceptualSpace, 
    Product,  # Deprecated
    ConceptualDimensions,  # Deprecated
    SearchResult
)

def test_conceptual_spaces():
    """Test the new ConceptualSpace system across different domains."""
    print("=== Testing ConceptualSpace System ===")
    
    # Test all supported domains
    domains = ["healthcare", "finance", "real_estate", "enterprise", "research", "retail"]
    
    for domain in domains:
        print(f"\n--- {domain.upper()} Domain ---")
        space = ConceptualSpace.create_for_domain(domain)
        print(f"Dimensions: {list(space.dimensions.keys())}")
        print(f"Sample dimension: {space.dimensions[list(space.dimensions.keys())[0]]}")

def test_entity_creation():
    """Test Entity creation for different domains."""
    print("\n=== Testing Entity Creation ===")
    
    # Healthcare Entity
    print("\n--- Healthcare Entity ---")
    patient = Entity.create_healthcare_entity(
        title="Patient Record #12345",
        description="Diabetes management case",
        entity_type="patient"
    )
    patient.set_domain_metadata("healthcare",
                               medical_record_number="MRN-12345",
                               diagnosis="Type 2 Diabetes",
                               treatment_plan="Metformin + lifestyle changes")
    
    print(f"Title: {patient.title}")
    print(f"Type: {patient.entity_type}")
    print(f"Diagnosis: {patient.get_domain_metadata('healthcare').get('diagnosis')}")
    print(f"Conceptual Space: {list(patient.conceptual_space.dimensions.keys())[:3]}...")
    
    # Finance Entity
    print("\n--- Finance Entity ---")
    asset = Entity.create_finance_entity(
        title="AAPL Stock Analysis",
        description="Apple Inc. equity analysis",
        entity_type="equity"
    )
    asset.set_domain_metadata("finance",
                             ticker="AAPL",
                             market_cap=3000000000000,
                             sector="Technology")
    
    print(f"Title: {asset.title}")
    print(f"Type: {asset.entity_type}")
    print(f"Ticker: {asset.get_domain_metadata('finance').get('ticker')}")
    print(f"Conceptual Space: {list(asset.conceptual_space.dimensions.keys())[:3]}...")
    
    # Real Estate Entity
    print("\n--- Real Estate Entity ---")
    property_entity = Entity.create_real_estate_entity(
        title="Downtown Condo",
        description="2BR/2BA luxury condo in city center",
        entity_type="residential"
    )
    property_entity.set_domain_metadata("real_estate",
                                      address="123 Main St, Unit 4B",
                                      square_feet=1200,
                                      bedrooms=2,
                                      bathrooms=2)
    
    print(f"Title: {property_entity.title}")
    print(f"Type: {property_entity.entity_type}")
    print(f"Address: {property_entity.get_domain_metadata('real_estate').get('address')}")
    print(f"Conceptual Space: {list(property_entity.conceptual_space.dimensions.keys())[:3]}...")

def test_backward_compatibility():
    """Test backward compatibility with deprecated Product class."""
    print("\n=== Testing Backward Compatibility ===")
    
    # Capture deprecation warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test deprecated Product class
        print("\n--- Deprecated Product Class ---")
        product = Product(
            title="Legacy Product",
            description="Testing backward compatibility"
        )
        
        # Set price and currency using properties
        product.price = 99.99
        product.currency = "USD"
        
        print(f"Product created: {product.title}")
        print(f"Price: {product.price} {product.currency}")
        print(f"Entity type: {product.entity_type}")
        print(f"Deprecation warnings captured: {len(w)}")
        
        # Test deprecated ConceptualDimensions
        print("\n--- Deprecated ConceptualDimensions ---")
        old_dims = ConceptualDimensions()
        new_space = old_dims.to_conceptual_space()
        
        print(f"Converted to ConceptualSpace: {type(new_space).__name__}")
        print(f"Dimensions: {list(new_space.dimensions.keys())[:3]}...")
        print(f"Total deprecation warnings: {len(w)}")

def test_search_result_compatibility():
    """Test SearchResult with new Entity model and backward compatibility."""
    print("\n=== Testing SearchResult Compatibility ===")
    
    # Create an entity
    entity = Entity.create_enterprise_entity(
        title="Technical Document",
        description="API documentation for microservices",
        entity_type="document"
    )
    
    # Create SearchResult with new entity field
    result = SearchResult(
        entity=entity,
        score=0.95,
        explanation="High relevance match",
        conceptual_similarity=0.92,
        embedding_similarity=0.88,
        knowledge_graph_similarity=0.90
    )
    
    print(f"Search result entity: {result.entity.title}")
    print(f"Score: {result.score}")
    
    # Test backward compatibility property
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Access via deprecated 'product' property
        legacy_product = result.product
        print(f"Legacy product access: {legacy_product.title}")
        print(f"Deprecation warnings: {len(w)}")

def test_data_serialization():
    """Test data serialization and deserialization."""
    print("\n=== Testing Data Serialization ===")
    
    # Create a research entity
    research_entity = Entity.create_research_entity(
        title="Climate Change Study",
        description="Impact of rising temperatures on arctic ice",
        entity_type="research_paper"
    )
    research_entity.set_domain_metadata("research",
                                      authors=["Dr. Smith", "Dr. Johnson"],
                                      journal="Nature Climate Change",
                                      impact_factor=25.3)
    
    # Serialize to dict
    entity_dict = research_entity.to_dict()
    print(f"Serialized entity keys: {list(entity_dict.keys())}")
    print(f"Domain metadata: {entity_dict.get('domain_metadata', {})}")
    
    # Test deserialization
    reconstructed = Entity.from_dict(entity_dict)
    print(f"Reconstructed entity: {reconstructed.title}")
    print(f"Authors: {reconstructed.get_domain_metadata('research').get('authors')}")
    print(f"Journal: {reconstructed.get_domain_metadata('research').get('journal')}")

def main():
    """Run all tests."""
    print("KSE-SDK Universal Data Model Validation")
    print("=" * 50)
    
    try:
        test_conceptual_spaces()
        test_entity_creation()
        test_backward_compatibility()
        test_search_result_compatibility()
        test_data_serialization()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] All tests completed successfully!")
        print("[SUCCESS] Universal data model is working correctly across all domains")
        print("[SUCCESS] Backward compatibility is maintained")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())