"""
KSE Memory SDK - Sample Datasets

Provides curated sample datasets for quickstart demos
across different domains (retail, finance, healthcare).
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import random

from ..core.models import Product, ConceptualDimensions


class SampleDatasets:
    """
    Provides sample datasets for quickstart demonstrations.
    
    Each dataset is carefully curated to showcase different
    aspects of hybrid AI search capabilities.
    """
    
    def __init__(self):
        """Initialize sample datasets."""
        self.random = random.Random(42)  # Deterministic for consistent demos
    
    def get_retail_products(self) -> List[Product]:
        """
        Get sample retail products dataset.
        
        Includes fashion, electronics, home goods with rich
        conceptual dimensions for demonstration.
        """
        products = []
        
        # Fashion items
        fashion_items = [
            {
                "title": "Premium Athletic Running Shoes",
                "description": "Lightweight mesh running shoes with responsive cushioning and breathable design. Perfect for daily training and long-distance runs.",
                "price": 129.99,
                "category": "Athletic Footwear",
                "tags": ["running", "athletic", "comfortable", "breathable", "lightweight"],
                "concepts": ConceptualDimensions(
                    elegance=0.6, comfort=0.9, boldness=0.4, modernity=0.8,
                    minimalism=0.7, luxury=0.3, functionality=0.95, versatility=0.8,
                    seasonality=0.5, innovation=0.7
                )
            },
            {
                "title": "Elegant Black Evening Dress",
                "description": "Sophisticated midi dress in premium silk with subtle shimmer. Features classic A-line silhouette perfect for formal occasions.",
                "price": 299.99,
                "category": "Formal Wear",
                "tags": ["elegant", "formal", "sophisticated", "silk", "evening"],
                "concepts": ConceptualDimensions(
                    elegance=0.95, comfort=0.6, boldness=0.3, modernity=0.7,
                    minimalism=0.8, luxury=0.9, functionality=0.4, versatility=0.6,
                    seasonality=0.3, innovation=0.2
                )
            },
            {
                "title": "Minimalist White Cotton T-Shirt",
                "description": "Essential crew neck t-shirt in organic cotton. Clean lines and perfect fit for everyday wear and layering.",
                "price": 29.99,
                "category": "Basics",
                "tags": ["minimalist", "cotton", "basic", "versatile", "sustainable"],
                "concepts": ConceptualDimensions(
                    elegance=0.5, comfort=0.8, boldness=0.2, modernity=0.6,
                    minimalism=0.95, luxury=0.2, functionality=0.9, versatility=0.95,
                    seasonality=0.5, innovation=0.3
                )
            },
            {
                "title": "Bold Geometric Print Jacket",
                "description": "Statement bomber jacket featuring vibrant geometric patterns. Oversized fit with premium technical fabric.",
                "price": 189.99,
                "category": "Outerwear",
                "tags": ["bold", "geometric", "statement", "oversized", "technical"],
                "concepts": ConceptualDimensions(
                    elegance=0.4, comfort=0.7, boldness=0.95, modernity=0.9,
                    minimalism=0.1, luxury=0.6, functionality=0.7, versatility=0.5,
                    seasonality=0.7, innovation=0.8
                )
            }
        ]
        
        # Electronics
        electronics_items = [
            {
                "title": "Wireless Noise-Canceling Headphones",
                "description": "Premium over-ear headphones with adaptive noise cancellation and 30-hour battery life. Studio-quality sound.",
                "price": 349.99,
                "category": "Audio",
                "tags": ["wireless", "noise-canceling", "premium", "audio", "battery"],
                "concepts": ConceptualDimensions(
                    elegance=0.8, comfort=0.9, boldness=0.3, modernity=0.9,
                    minimalism=0.7, luxury=0.8, functionality=0.95, versatility=0.8,
                    seasonality=0.5, innovation=0.9
                )
            },
            {
                "title": "Smart Fitness Tracker Watch",
                "description": "Advanced fitness tracking with heart rate monitoring, GPS, and 7-day battery. Water-resistant design.",
                "price": 199.99,
                "category": "Wearables",
                "tags": ["smart", "fitness", "tracking", "GPS", "waterproof"],
                "concepts": ConceptualDimensions(
                    elegance=0.6, comfort=0.8, boldness=0.4, modernity=0.95,
                    minimalism=0.8, luxury=0.5, functionality=0.95, versatility=0.9,
                    seasonality=0.5, innovation=0.9
                )
            }
        ]
        
        # Home goods
        home_items = [
            {
                "title": "Scandinavian Oak Dining Table",
                "description": "Handcrafted dining table in sustainable oak with clean lines. Seats 6 comfortably with timeless design.",
                "price": 899.99,
                "category": "Furniture",
                "tags": ["scandinavian", "oak", "handcrafted", "sustainable", "dining"],
                "concepts": ConceptualDimensions(
                    elegance=0.9, comfort=0.7, boldness=0.2, modernity=0.6,
                    minimalism=0.9, luxury=0.7, functionality=0.9, versatility=0.6,
                    seasonality=0.3, innovation=0.3
                )
            },
            {
                "title": "Modern LED Floor Lamp",
                "description": "Adjustable LED floor lamp with touch controls and wireless charging base. Minimalist design with warm lighting.",
                "price": 159.99,
                "category": "Lighting",
                "tags": ["modern", "LED", "adjustable", "wireless", "minimalist"],
                "concepts": ConceptualDimensions(
                    elegance=0.7, comfort=0.6, boldness=0.3, modernity=0.95,
                    minimalism=0.9, luxury=0.4, functionality=0.9, versatility=0.7,
                    seasonality=0.5, innovation=0.8
                )
            }
        ]
        
        # Convert to Product objects
        all_items = fashion_items + electronics_items + home_items
        
        for i, item in enumerate(all_items):
            product = Product(
                id=f"demo_retail_{i+1:03d}",
                title=item["title"],
                description=item["description"],
                price=item["price"],
                category=item["category"],
                tags=item["tags"],
                metadata={
                    "brand": self._generate_brand_name(),
                    "availability": "in_stock",
                    "rating": round(self.random.uniform(4.0, 5.0), 1),
                    "reviews_count": self.random.randint(50, 500)
                },
                conceptual_dimensions=item["concepts"]
            )
            products.append(product)
        
        return products
    
    def get_finance_products(self) -> List[Product]:
        """
        Get sample financial products dataset.
        
        Includes investment products, insurance, banking services
        with appropriate conceptual dimensions.
        """
        products = []
        
        finance_items = [
            {
                "title": "High-Yield Savings Account",
                "description": "Competitive savings account with 4.5% APY and no minimum balance. FDIC insured with mobile banking.",
                "price": 0.0,  # No cost product
                "category": "Banking",
                "tags": ["savings", "high-yield", "FDIC", "mobile", "no-minimum"],
                "concepts": ConceptualDimensions(
                    elegance=0.6, comfort=0.9, boldness=0.2, modernity=0.8,
                    minimalism=0.8, luxury=0.3, functionality=0.9, versatility=0.7,
                    seasonality=0.5, innovation=0.6
                )
            },
            {
                "title": "Growth-Focused ETF Portfolio",
                "description": "Diversified portfolio of growth ETFs with low expense ratios. Automated rebalancing and tax optimization.",
                "price": 25.0,  # Monthly fee
                "category": "Investments",
                "tags": ["ETF", "growth", "diversified", "automated", "tax-optimized"],
                "concepts": ConceptualDimensions(
                    elegance=0.7, comfort=0.6, boldness=0.8, modernity=0.9,
                    minimalism=0.7, luxury=0.5, functionality=0.9, versatility=0.8,
                    seasonality=0.5, innovation=0.8
                )
            },
            {
                "title": "Comprehensive Life Insurance",
                "description": "Term life insurance with flexible coverage options. Online application with instant approval for qualified applicants.",
                "price": 89.99,  # Monthly premium
                "category": "Insurance",
                "tags": ["life-insurance", "term", "flexible", "online", "instant-approval"],
                "concepts": ConceptualDimensions(
                    elegance=0.5, comfort=0.8, boldness=0.3, modernity=0.7,
                    minimalism=0.6, luxury=0.4, functionality=0.95, versatility=0.6,
                    seasonality=0.5, innovation=0.5
                )
            }
        ]
        
        for i, item in enumerate(finance_items):
            product = Product(
                id=f"demo_finance_{i+1:03d}",
                title=item["title"],
                description=item["description"],
                price=item["price"],
                category=item["category"],
                tags=item["tags"],
                metadata={
                    "provider": self._generate_finance_provider(),
                    "regulation": "SEC_registered",
                    "rating": f"A{self.random.choice(['+', '', '-'])}",
                    "minimum_investment": self.random.choice([0, 100, 1000, 5000])
                },
                conceptual_dimensions=item["concepts"]
            )
            products.append(product)
        
        return products
    
    def get_healthcare_products(self) -> List[Product]:
        """
        Get sample healthcare products dataset.
        
        Includes medical devices, pharmaceuticals, health services
        with appropriate conceptual dimensions.
        """
        products = []
        
        healthcare_items = [
            {
                "title": "Digital Blood Pressure Monitor",
                "description": "Clinically validated automatic blood pressure monitor with smartphone connectivity and cloud data storage.",
                "price": 79.99,
                "category": "Monitoring Devices",
                "tags": ["blood-pressure", "digital", "validated", "smartphone", "cloud"],
                "concepts": ConceptualDimensions(
                    elegance=0.6, comfort=0.8, boldness=0.3, modernity=0.9,
                    minimalism=0.7, luxury=0.4, functionality=0.95, versatility=0.7,
                    seasonality=0.5, innovation=0.8
                )
            },
            {
                "title": "Advanced MRI Imaging System",
                "description": "High-resolution 3T MRI system with AI-enhanced imaging protocols. Reduced scan times and improved patient comfort.",
                "price": 2500000.0,  # Enterprise equipment
                "category": "Diagnostic Imaging",
                "tags": ["MRI", "3T", "AI-enhanced", "high-resolution", "patient-comfort"],
                "concepts": ConceptualDimensions(
                    elegance=0.8, comfort=0.7, boldness=0.6, modernity=0.95,
                    minimalism=0.5, luxury=0.9, functionality=0.95, versatility=0.8,
                    seasonality=0.5, innovation=0.95
                )
            },
            {
                "title": "Precision Surgical Instruments Set",
                "description": "Titanium surgical instrument set with ergonomic design. Autoclave-safe with lifetime warranty.",
                "price": 1299.99,
                "category": "Surgical Tools",
                "tags": ["surgical", "titanium", "ergonomic", "autoclave-safe", "warranty"],
                "concepts": ConceptualDimensions(
                    elegance=0.9, comfort=0.8, boldness=0.4, modernity=0.8,
                    minimalism=0.8, luxury=0.8, functionality=0.95, versatility=0.9,
                    seasonality=0.5, innovation=0.7
                )
            }
        ]
        
        for i, item in enumerate(healthcare_items):
            product = Product(
                id=f"demo_healthcare_{i+1:03d}",
                title=item["title"],
                description=item["description"],
                price=item["price"],
                category=item["category"],
                tags=item["tags"],
                metadata={
                    "manufacturer": self._generate_healthcare_manufacturer(),
                    "fda_approved": True,
                    "certification": self.random.choice(["ISO_13485", "CE_marked", "FDA_510k"]),
                    "warranty_years": self.random.randint(1, 10)
                },
                conceptual_dimensions=item["concepts"]
            )
            products.append(product)
        
        return products
    
    def _generate_brand_name(self) -> str:
        """Generate realistic brand names for retail products."""
        prefixes = ["Nova", "Zen", "Pure", "Elite", "Prime", "Luxe", "Urban", "Modern"]
        suffixes = ["Co", "Labs", "Studio", "Design", "Craft", "Works", "House", "Brand"]
        
        return f"{self.random.choice(prefixes)} {self.random.choice(suffixes)}"
    
    def _generate_finance_provider(self) -> str:
        """Generate realistic financial service provider names."""
        names = [
            "Meridian Financial", "Apex Capital", "Sterling Investments",
            "Pinnacle Wealth", "Horizon Bank", "Summit Financial",
            "Vanguard Partners", "Cornerstone Capital"
        ]
        return self.random.choice(names)
    
    def _generate_healthcare_manufacturer(self) -> str:
        """Generate realistic healthcare manufacturer names."""
        names = [
            "MedTech Solutions", "Precision Medical", "Advanced Diagnostics",
            "HealthCare Innovations", "BioMed Systems", "Clinical Devices Inc",
            "Medical Instruments Corp", "Diagnostic Technologies"
        ]
        return self.random.choice(names)
    
    def get_sample_queries(self, domain: str) -> List[Dict[str, str]]:
        """
        Get sample search queries for demonstration.
        
        Args:
            domain: Domain type ('retail', 'finance', 'healthcare')
            
        Returns:
            List of query dictionaries with 'query' and 'description'
        """
        if domain == "retail":
            return [
                {
                    "query": "comfortable running shoes",
                    "description": "Find athletic footwear for daily exercise"
                },
                {
                    "query": "elegant evening wear",
                    "description": "Discover sophisticated formal attire"
                },
                {
                    "query": "minimalist home decor",
                    "description": "Explore clean, simple design items"
                },
                {
                    "query": "bold statement pieces",
                    "description": "Find eye-catching fashion items"
                },
                {
                    "query": "sustainable organic materials",
                    "description": "Discover eco-friendly products"
                }
            ]
        
        elif domain == "finance":
            return [
                {
                    "query": "high-yield investment products",
                    "description": "Find profitable investment opportunities"
                },
                {
                    "query": "risk management tools",
                    "description": "Discover portfolio protection instruments"
                },
                {
                    "query": "retirement planning services",
                    "description": "Explore long-term financial planning"
                },
                {
                    "query": "automated investment solutions",
                    "description": "Find hands-off investment options"
                },
                {
                    "query": "tax-optimized strategies",
                    "description": "Discover tax-efficient financial products"
                }
            ]
        
        else:  # healthcare
            return [
                {
                    "query": "diagnostic imaging equipment",
                    "description": "Find medical imaging solutions"
                },
                {
                    "query": "patient monitoring devices",
                    "description": "Discover vital sign tracking tools"
                },
                {
                    "query": "surgical instruments",
                    "description": "Explore precision medical tools"
                },
                {
                    "query": "AI-enhanced medical devices",
                    "description": "Find smart healthcare technology"
                },
                {
                    "query": "portable diagnostic tools",
                    "description": "Discover mobile medical equipment"
                }
            ]