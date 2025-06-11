# KSE Memory SDK - API Reference

This document provides a comprehensive reference for the KSE Memory SDK API.

## Table of Contents

- [Core Classes](#core-classes)
- [Configuration](#configuration)
- [Data Models](#data-models)
- [Adapters](#adapters)
- [Backends](#backends)
- [Services](#services)
- [Exceptions](#exceptions)
- [Examples](#examples)

## Core Classes

### KSEMemory

The main interface for the KSE Memory SDK.

```python
from kse_memory import KSEMemory, KSEConfig

# Initialize
config = KSEConfig()
kse = KSEMemory(config)

# Connect to data source
await kse.initialize("shopify", shopify_config)

# Add products
await kse.add_product(product)

# Search
results = await kse.search("comfortable shoes")

# Get recommendations
recommendations = await kse.get_recommendations(product_id)

# Health check
health = await kse.health_check()

# Cleanup
await kse.disconnect()
```

#### Methods

##### `__init__(config: Optional[KSEConfig] = None)`

Initialize KSE Memory system.

**Parameters:**
- `config`: Configuration object (optional, uses defaults if None)

##### `async initialize(adapter_type: str, adapter_config: Dict[str, Any]) -> bool`

Initialize the system with a specific adapter.

**Parameters:**
- `adapter_type`: Type of adapter ('shopify', 'woocommerce', 'generic')
- `adapter_config`: Configuration for the adapter

**Returns:** `True` if initialization successful

**Raises:** `KSEError` if initialization fails

##### `async add_product(product: Product, compute_embeddings: bool = True, compute_concepts: bool = True) -> bool`

Add a product to the memory system.

**Parameters:**
- `product`: Product to add
- `compute_embeddings`: Whether to compute embeddings
- `compute_concepts`: Whether to compute conceptual dimensions

**Returns:** `True` if product added successfully

##### `async search(query: Union[str, SearchQuery]) -> List[SearchResult]`

Search for products.

**Parameters:**
- `query`: Search query (string or SearchQuery object)

**Returns:** List of search results

##### `async get_recommendations(product_id: str, limit: int = 10) -> List[SearchResult]`

Get product recommendations.

**Parameters:**
- `product_id`: ID of the product to base recommendations on
- `limit`: Maximum number of recommendations

**Returns:** List of recommended products

##### `async health_check() -> Dict[str, Any]`

Perform a health check on all components.

**Returns:** Dictionary with health status

## Configuration

### KSEConfig

Main configuration class for the SDK.

```python
from kse_memory import KSEConfig

# Default configuration
config = KSEConfig()

# Custom configuration
config = KSEConfig(
    debug=True,
    vector_store=VectorStoreConfig(
        backend="pinecone",
        api_key="your-api-key"
    )
)

# From dictionary
config = KSEConfig.from_dict({
    "debug": True,
    "vector_store": {
        "backend": "weaviate",
        "host": "localhost"
    }
})

# From file
config = KSEConfig.from_file("config.yaml")
```

#### Properties

- `app_name: str` - Application name
- `version: str` - Version string
- `debug: bool` - Debug mode flag
- `log_level: LogLevel` - Logging level
- `vector_store: VectorStoreConfig` - Vector store configuration
- `graph_store: GraphStoreConfig` - Graph store configuration
- `concept_store: ConceptStoreConfig` - Concept store configuration
- `embedding: EmbeddingConfig` - Embedding configuration
- `conceptual: ConceptualConfig` - Conceptual configuration
- `search: SearchConfig` - Search configuration
- `cache: CacheConfig` - Cache configuration

### VectorStoreConfig

Configuration for vector storage backends.

```python
from kse_memory.core.config import VectorStoreConfig

config = VectorStoreConfig(
    backend="pinecone",
    api_key="your-pinecone-api-key",
    environment="us-west1-gcp",
    index_name="kse-products",
    dimension=384
)
```

#### Properties

- `backend: str` - Backend type ('pinecone', 'weaviate', 'qdrant')
- `api_key: str` - API key for the service
- `environment: str` - Environment/region
- `index_name: str` - Name of the index
- `dimension: int` - Vector dimension
- `metric: str` - Distance metric ('cosine', 'euclidean', 'dotproduct')

## Data Models

### Product

Core product representation.

```python
from kse_memory import Product, ConceptualDimensions

product = Product(
    id="prod_001",
    title="Nike Air Max 270",
    description="Comfortable running shoes...",
    price=150.00,
    currency="USD",
    category="Footwear",
    brand="Nike",
    tags=["running", "comfortable"],
    images=["https://example.com/image.jpg"],
    conceptual_dimensions=ConceptualDimensions(
        comfort=0.9,
        functionality=0.8
    )
)
```

#### Properties

- `id: str` - Unique product identifier
- `title: str` - Product title
- `description: str` - Product description
- `price: Optional[float]` - Product price
- `currency: Optional[str]` - Price currency
- `category: Optional[str]` - Product category
- `brand: Optional[str]` - Product brand
- `tags: List[str]` - Product tags
- `images: List[str]` - Image URLs
- `variants: List[Dict[str, Any]]` - Product variants
- `conceptual_dimensions: Optional[ConceptualDimensions]` - Conceptual dimensions
- `text_embedding: Optional[EmbeddingVector]` - Text embedding
- `image_embedding: Optional[EmbeddingVector]` - Image embedding

### ConceptualDimensions

Conceptual space coordinates for products.

```python
from kse_memory import ConceptualDimensions

dimensions = ConceptualDimensions(
    elegance=0.8,
    comfort=0.9,
    boldness=0.3,
    modernity=0.7,
    minimalism=0.6,
    luxury=0.8,
    functionality=0.9,
    versatility=0.7,
    seasonality=0.4,
    innovation=0.6
)
```

#### Properties

All properties are `float` values between 0.0 and 1.0:

- `elegance` - How refined and sophisticated
- `comfort` - How comfortable and pleasant
- `boldness` - How striking and attention-grabbing
- `modernity` - How contemporary and current
- `minimalism` - How simple and clean
- `luxury` - How premium and exclusive
- `functionality` - How practical and useful
- `versatility` - How adaptable for multiple uses
- `seasonality` - How tied to specific seasons
- `innovation` - How novel and advanced

### SearchQuery

Search query configuration.

```python
from kse_memory import SearchQuery, SearchType

# Basic search
query = SearchQuery(
    query="comfortable running shoes",
    search_type=SearchType.SEMANTIC,
    limit=10
)

# Hybrid search with weights
query = SearchQuery(
    query="elegant luxury items",
    search_type=SearchType.HYBRID,
    conceptual_weights={"elegance": 0.6, "luxury": 0.8},
    filters={"category": "Fashion", "price_min": 100}
)
```

#### Properties

- `query: str` - Search query text
- `search_type: SearchType` - Type of search
- `filters: Dict[str, Any]` - Search filters
- `limit: int` - Maximum results (default: 10)
- `offset: int` - Results offset (default: 0)
- `conceptual_weights: Optional[Dict[str, float]]` - Dimension weights

#### SearchType Enum

- `SEMANTIC` - Embedding-based search
- `CONCEPTUAL` - Conceptual dimension search
- `KNOWLEDGE_GRAPH` - Graph relationship search
- `HYBRID` - Combined approach

### SearchResult

Search result with scoring information.

```python
# Results from search
results = await kse.search("comfortable shoes")

for result in results:
    print(f"Product: {result.product.title}")
    print(f"Score: {result.score}")
    print(f"Explanation: {result.explanation}")
```

#### Properties

- `product: Product` - The product
- `score: float` - Overall relevance score
- `explanation: Optional[str]` - Explanation of the match
- `conceptual_similarity: Optional[float]` - Conceptual similarity score
- `embedding_similarity: Optional[float]` - Embedding similarity score
- `knowledge_graph_similarity: Optional[float]` - Graph similarity score

## Adapters

### ShopifyAdapter

Shopify platform integration.

```python
from kse_memory.adapters import ShopifyAdapter

adapter = ShopifyAdapter()

# Connect
await adapter.connect({
    "shop_url": "your-shop.myshopify.com",
    "access_token": "your-access-token"
})

# Get products
products = await adapter.get_products(limit=100)

# Handle webhooks
success = await adapter.webhook_handler("products/create", webhook_payload)
```

### WooCommerceAdapter

WooCommerce platform integration.

```python
from kse_memory.adapters import WooCommerceAdapter

adapter = WooCommerceAdapter()

# Connect
await adapter.connect({
    "store_url": "https://your-store.com",
    "consumer_key": "your-consumer-key",
    "consumer_secret": "your-consumer-secret"
})
```

### GenericAdapter

Generic data source integration.

```python
from kse_memory.adapters import GenericAdapter

# CSV data source
csv_source = GenericAdapter.create_csv_data_source("products.csv")

# JSON data source
json_source = GenericAdapter.create_json_data_source("products.json")

# API data source
api_source = GenericAdapter.create_api_data_source(
    "https://api.example.com",
    headers={"Authorization": "Bearer token"}
)

adapter = GenericAdapter()
await adapter.connect({
    "data_source": csv_source
})
```

## Backends

### PineconeBackend

Pinecone vector store backend.

```python
from kse_memory.backends import PineconeBackend
from kse_memory.core.config import VectorStoreConfig

config = VectorStoreConfig(
    backend="pinecone",
    api_key="your-api-key",
    environment="us-west1-gcp"
)

backend = PineconeBackend(config)
await backend.connect()
```

### Neo4jBackend

Neo4j graph store backend.

```python
from kse_memory.backends import Neo4jBackend
from kse_memory.core.config import GraphStoreConfig

config = GraphStoreConfig(
    backend="neo4j",
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

backend = Neo4jBackend(config)
await backend.connect()
```

### PostgreSQLBackend

PostgreSQL concept store backend.

```python
from kse_memory.backends import PostgreSQLBackend
from kse_memory.core.config import ConceptStoreConfig

config = ConceptStoreConfig(
    backend="postgresql",
    host="localhost",
    database="kse_concepts",
    username="postgres",
    password="password"
)

backend = PostgreSQLBackend(config)
await backend.connect()
```

## Services

### EmbeddingService

Generate text and image embeddings.

```python
from kse_memory.services import EmbeddingService
from kse_memory.core.config import EmbeddingConfig

config = EmbeddingConfig(
    text_model="sentence-transformers/all-MiniLM-L6-v2"
)

service = EmbeddingService(config)

# Generate text embedding
embedding = await service.generate_text_embedding("product description")

# Generate batch embeddings
embeddings = await service.generate_batch_text_embeddings(["text1", "text2"])
```

### ConceptualService

Compute conceptual dimensions.

```python
from kse_memory.services import ConceptualService
from kse_memory.core.config import ConceptualConfig

config = ConceptualConfig(
    auto_compute=True,
    llm_api_key="your-openai-key"
)

service = ConceptualService(config)

# Compute dimensions
dimensions = await service.compute_dimensions(product)

# Explain dimensions
explanation = await service.explain_dimensions(product, dimensions)
```

### SearchService

Hybrid search functionality.

```python
from kse_memory.services import SearchService

service = SearchService(
    config=search_config,
    vector_store=vector_store,
    graph_store=graph_store,
    concept_store=concept_store,
    embedding_service=embedding_service
)

# Perform search
results = await service.search(search_query)
```

## Exceptions

### KSEError

Base exception for all KSE Memory errors.

```python
from kse_memory.exceptions import KSEError

try:
    await kse.search("query")
except KSEError as e:
    print(f"Error: {e.message}")
    print(f"Code: {e.error_code}")
    print(f"Details: {e.details}")
```

### Specific Exceptions

- `ConfigurationError` - Configuration issues
- `AdapterError` - Adapter-related errors
- `BackendError` - Storage backend errors
- `SearchError` - Search operation errors
- `EmbeddingError` - Embedding generation errors
- `ConceptualError` - Conceptual computation errors

## Examples

### Basic Usage

```python
import asyncio
from kse_memory import KSEMemory, KSEConfig, Product

async def main():
    # Initialize
    config = KSEConfig()
    kse = KSEMemory(config)
    
    # Connect to data source
    await kse.initialize("generic", {
        "data_source": lambda: [
            Product(id="1", title="Test Product", description="A test")
        ]
    })
    
    # Search
    results = await kse.search("test")
    print(f"Found {len(results)} results")
    
    # Cleanup
    await kse.disconnect()

asyncio.run(main())
```

### Advanced Search

```python
from kse_memory import SearchQuery, SearchType

# Semantic search
results = await kse.search(SearchQuery(
    query="comfortable running shoes",
    search_type=SearchType.SEMANTIC,
    filters={"category": "Footwear"},
    limit=20
))

# Conceptual search
results = await kse.search(SearchQuery(
    query="elegant luxury items",
    search_type=SearchType.CONCEPTUAL,
    conceptual_weights={"elegance": 0.8, "luxury": 0.9}
))

# Hybrid search
results = await kse.search(SearchQuery(
    query="modern minimalist design",
    search_type=SearchType.HYBRID,
    filters={"price_min": 50, "price_max": 500}
))
```

### Custom Configuration

```python
config = KSEConfig(
    debug=True,
    vector_store=VectorStoreConfig(
        backend="weaviate",
        host="localhost",
        port=8080
    ),
    embedding=EmbeddingConfig(
        text_model="sentence-transformers/all-mpnet-base-v2",
        batch_size=16
    ),
    conceptual=ConceptualConfig(
        auto_compute=True,
        llm_model="gpt-4"
    )
)
```

For more examples, see the `examples/` directory in the repository.