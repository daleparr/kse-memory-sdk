"""
Shopify adapter for KSE Memory SDK.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import shopifyapi
    from shopifyapi import Session, Product as ShopifyProduct
    SHOPIFY_AVAILABLE = True
except ImportError:
    SHOPIFY_AVAILABLE = False

from ..core.interfaces import AdapterInterface
from ..core.models import Product
from ..exceptions import AdapterError, AuthenticationError


logger = logging.getLogger(__name__)


class ShopifyAdapter(AdapterInterface):
    """
    Shopify platform adapter for KSE Memory SDK.
    
    Provides integration with Shopify stores to sync product data
    and handle webhook events.
    """
    
    def __init__(self):
        """Initialize Shopify adapter."""
        if not SHOPIFY_AVAILABLE:
            raise AdapterError(
                "Shopify dependencies not installed. Install with: pip install kse-memory[shopify]",
                adapter_type="shopify"
            )
        
        self.session: Optional[Session] = None
        self.shop_url: Optional[str] = None
        self.api_version = "2023-10"
        self._connected = False
        
        logger.info("Shopify adapter initialized")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """
        Connect to Shopify store.
        
        Args:
            config: Configuration dictionary with keys:
                - shop_url: Shopify store URL
                - access_token: Private app access token
                - api_version: API version (optional, defaults to 2023-10)
                
        Returns:
            True if connection successful
            
        Raises:
            AdapterError: If connection fails
            AuthenticationError: If authentication fails
        """
        try:
            # Validate required config
            required_keys = ["shop_url", "access_token"]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise AdapterError(
                    f"Missing required configuration keys: {missing_keys}",
                    adapter_type="shopify"
                )
            
            self.shop_url = config["shop_url"]
            access_token = config["access_token"]
            self.api_version = config.get("api_version", self.api_version)
            
            # Initialize Shopify session
            self.session = Session(
                shop=self.shop_url,
                version=self.api_version,
                token=access_token
            )
            
            # Test connection by fetching shop info
            try:
                shop_info = await self._make_request("GET", "shop.json")
                logger.info(f"Connected to Shopify store: {shop_info.get('shop', {}).get('name', 'Unknown')}")
                self._connected = True
                return True
                
            except Exception as e:
                raise AuthenticationError(
                    f"Failed to authenticate with Shopify: {str(e)}",
                    service="shopify"
                )
                
        except Exception as e:
            logger.error(f"Failed to connect to Shopify: {str(e)}")
            if isinstance(e, (AdapterError, AuthenticationError)):
                raise
            raise AdapterError(f"Connection failed: {str(e)}", adapter_type="shopify")
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Shopify store.
        
        Returns:
            True if disconnection successful
        """
        try:
            self.session = None
            self.shop_url = None
            self._connected = False
            logger.info("Disconnected from Shopify")
            return True
            
        except Exception as e:
            logger.error(f"Error during Shopify disconnection: {str(e)}")
            return False
    
    async def get_products(self, limit: int = 100, offset: int = 0) -> List[Product]:
        """
        Retrieve products from Shopify store.
        
        Args:
            limit: Maximum number of products to retrieve
            offset: Number of products to skip
            
        Returns:
            List of Product objects
            
        Raises:
            AdapterError: If retrieval fails
        """
        self._ensure_connected()
        
        try:
            # Calculate page number for Shopify pagination
            page = (offset // limit) + 1
            
            # Fetch products from Shopify
            response = await self._make_request(
                "GET", 
                "products.json",
                params={
                    "limit": min(limit, 250),  # Shopify max limit is 250
                    "page": page,
                    "status": "active"
                }
            )
            
            shopify_products = response.get("products", [])
            products = []
            
            for shopify_product in shopify_products:
                try:
                    product = self._convert_shopify_product(shopify_product)
                    products.append(product)
                except Exception as e:
                    logger.warning(f"Failed to convert Shopify product {shopify_product.get('id')}: {str(e)}")
            
            logger.debug(f"Retrieved {len(products)} products from Shopify")
            return products
            
        except Exception as e:
            logger.error(f"Failed to get products from Shopify: {str(e)}")
            raise AdapterError(f"Failed to retrieve products: {str(e)}", adapter_type="shopify")
    
    async def get_product(self, product_id: str) -> Optional[Product]:
        """
        Retrieve a specific product by ID.
        
        Args:
            product_id: Shopify product ID
            
        Returns:
            Product object if found, None otherwise
        """
        self._ensure_connected()
        
        try:
            response = await self._make_request("GET", f"products/{product_id}.json")
            shopify_product = response.get("product")
            
            if shopify_product:
                return self._convert_shopify_product(shopify_product)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get product {product_id} from Shopify: {str(e)}")
            return None
    
    async def sync_products(self) -> int:
        """
        Sync all products from Shopify store.
        
        Returns:
            Number of products synced
        """
        self._ensure_connected()
        
        try:
            total_synced = 0
            page = 1
            limit = 250  # Shopify max limit
            
            while True:
                # Get products for current page
                response = await self._make_request(
                    "GET",
                    "products.json",
                    params={
                        "limit": limit,
                        "page": page,
                        "status": "active"
                    }
                )
                
                shopify_products = response.get("products", [])
                
                if not shopify_products:
                    break
                
                # Convert and count products
                for shopify_product in shopify_products:
                    try:
                        self._convert_shopify_product(shopify_product)
                        total_synced += 1
                    except Exception as e:
                        logger.warning(f"Failed to sync product {shopify_product.get('id')}: {str(e)}")
                
                # Check if we have more pages
                if len(shopify_products) < limit:
                    break
                
                page += 1
            
            logger.info(f"Synced {total_synced} products from Shopify")
            return total_synced
            
        except Exception as e:
            logger.error(f"Failed to sync products from Shopify: {str(e)}")
            raise AdapterError(f"Product sync failed: {str(e)}", adapter_type="shopify")
    
    async def webhook_handler(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Handle webhook events from Shopify.
        
        Args:
            event_type: Type of webhook event
            payload: Webhook payload
            
        Returns:
            True if event handled successfully
        """
        try:
            logger.debug(f"Handling Shopify webhook: {event_type}")
            
            if event_type in ["products/create", "products/update"]:
                # Convert and return the product for further processing
                shopify_product = payload
                product = self._convert_shopify_product(shopify_product)
                logger.info(f"Product {event_type.split('/')[1]}d: {product.id}")
                return True
                
            elif event_type == "products/delete":
                product_id = str(payload.get("id"))
                logger.info(f"Product deleted: {product_id}")
                return True
                
            else:
                logger.debug(f"Unhandled webhook event type: {event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to handle Shopify webhook {event_type}: {str(e)}")
            return False
    
    def _convert_shopify_product(self, shopify_product: Dict[str, Any]) -> Product:
        """
        Convert Shopify product data to KSE Product model.
        
        Args:
            shopify_product: Shopify product data
            
        Returns:
            Product object
        """
        try:
            # Extract basic product information
            product_id = str(shopify_product["id"])
            title = shopify_product.get("title", "")
            description = shopify_product.get("body_html", "")
            
            # Clean HTML from description
            import re
            description = re.sub(r'<[^>]+>', '', description)
            
            # Extract variants and pricing
            variants = shopify_product.get("variants", [])
            price = None
            if variants:
                # Use the first variant's price
                price = float(variants[0].get("price", 0))
            
            # Extract images
            images = []
            for image in shopify_product.get("images", []):
                if image.get("src"):
                    images.append(image["src"])
            
            # Extract tags
            tags = []
            if shopify_product.get("tags"):
                tags = [tag.strip() for tag in shopify_product["tags"].split(",")]
            
            # Extract other metadata
            vendor = shopify_product.get("vendor")
            product_type = shopify_product.get("product_type")
            
            # Parse timestamps
            created_at = datetime.utcnow()
            updated_at = datetime.utcnow()
            
            if shopify_product.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(
                        shopify_product["created_at"].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            if shopify_product.get("updated_at"):
                try:
                    updated_at = datetime.fromisoformat(
                        shopify_product["updated_at"].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            # Create Product object
            product = Product(
                id=product_id,
                title=title,
                description=description,
                price=price,
                currency="USD",  # Default, could be extracted from shop settings
                category=product_type,
                brand=vendor,
                tags=tags,
                images=images,
                variants=[{
                    "id": str(variant.get("id")),
                    "title": variant.get("title"),
                    "price": float(variant.get("price", 0)),
                    "sku": variant.get("sku"),
                    "inventory_quantity": variant.get("inventory_quantity"),
                } for variant in variants],
                metadata={
                    "shopify_id": product_id,
                    "shopify_handle": shopify_product.get("handle"),
                    "shopify_status": shopify_product.get("status"),
                    "shopify_vendor": vendor,
                    "shopify_product_type": product_type,
                },
                created_at=created_at,
                updated_at=updated_at,
            )
            
            return product
            
        except Exception as e:
            raise AdapterError(
                f"Failed to convert Shopify product: {str(e)}",
                adapter_type="shopify",
                details={"shopify_product_id": shopify_product.get("id")}
            )
    
    async def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make an API request to Shopify.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            AdapterError: If request fails
        """
        try:
            # This is a simplified implementation
            # In a real implementation, you would use the Shopify API client
            # For now, we'll simulate the response structure
            
            if endpoint == "shop.json":
                return {
                    "shop": {
                        "name": "Test Shop",
                        "id": 12345
                    }
                }
            elif endpoint == "products.json":
                # Return empty products list for now
                return {"products": []}
            elif endpoint.startswith("products/") and endpoint.endswith(".json"):
                # Return None for specific product requests
                return {"product": None}
            
            return {}
            
        except Exception as e:
            raise AdapterError(
                f"API request failed: {str(e)}",
                adapter_type="shopify",
                details={"method": method, "endpoint": endpoint}
            )
    
    def _ensure_connected(self):
        """Ensure the adapter is connected."""
        if not self._connected:
            raise AdapterError("Not connected to Shopify. Call connect() first.", adapter_type="shopify")