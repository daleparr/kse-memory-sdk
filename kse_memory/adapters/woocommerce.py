"""
WooCommerce adapter for KSE Memory SDK.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from woocommerce import API as WooCommerceAPI
    WOOCOMMERCE_AVAILABLE = True
except ImportError:
    WOOCOMMERCE_AVAILABLE = False

from ..core.interfaces import AdapterInterface
from ..core.models import Product
from ..exceptions import AdapterError, AuthenticationError


logger = logging.getLogger(__name__)


class WooCommerceAdapter(AdapterInterface):
    """
    WooCommerce platform adapter for KSE Memory SDK.
    
    Provides integration with WooCommerce stores to sync product data
    and handle webhook events.
    """
    
    def __init__(self):
        """Initialize WooCommerce adapter."""
        if not WOOCOMMERCE_AVAILABLE:
            raise AdapterError(
                "WooCommerce dependencies not installed. Install with: pip install kse-memory[woocommerce]",
                adapter_type="woocommerce"
            )
        
        self.api: Optional[WooCommerceAPI] = None
        self.store_url: Optional[str] = None
        self._connected = False
        
        logger.info("WooCommerce adapter initialized")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """
        Connect to WooCommerce store.
        
        Args:
            config: Configuration dictionary with keys:
                - store_url: WooCommerce store URL
                - consumer_key: WooCommerce consumer key
                - consumer_secret: WooCommerce consumer secret
                - version: API version (optional, defaults to wc/v3)
                
        Returns:
            True if connection successful
            
        Raises:
            AdapterError: If connection fails
            AuthenticationError: If authentication fails
        """
        try:
            # Validate required config
            required_keys = ["store_url", "consumer_key", "consumer_secret"]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise AdapterError(
                    f"Missing required configuration keys: {missing_keys}",
                    adapter_type="woocommerce"
                )
            
            self.store_url = config["store_url"]
            consumer_key = config["consumer_key"]
            consumer_secret = config["consumer_secret"]
            version = config.get("version", "wc/v3")
            
            # Initialize WooCommerce API
            self.api = WooCommerceAPI(
                url=self.store_url,
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                version=version,
                timeout=30
            )
            
            # Test connection by fetching store info
            try:
                response = self.api.get("system_status")
                if response.status_code == 200:
                    store_info = response.json()
                    logger.info(f"Connected to WooCommerce store: {store_info.get('settings', {}).get('title', 'Unknown')}")
                    self._connected = True
                    return True
                else:
                    raise AuthenticationError(
                        f"Failed to authenticate with WooCommerce: HTTP {response.status_code}",
                        service="woocommerce"
                    )
                    
            except Exception as e:
                raise AuthenticationError(
                    f"Failed to authenticate with WooCommerce: {str(e)}",
                    service="woocommerce"
                )
                
        except Exception as e:
            logger.error(f"Failed to connect to WooCommerce: {str(e)}")
            if isinstance(e, (AdapterError, AuthenticationError)):
                raise
            raise AdapterError(f"Connection failed: {str(e)}", adapter_type="woocommerce")
    
    async def disconnect(self) -> bool:
        """
        Disconnect from WooCommerce store.
        
        Returns:
            True if disconnection successful
        """
        try:
            self.api = None
            self.store_url = None
            self._connected = False
            logger.info("Disconnected from WooCommerce")
            return True
            
        except Exception as e:
            logger.error(f"Error during WooCommerce disconnection: {str(e)}")
            return False
    
    async def get_products(self, limit: int = 100, offset: int = 0) -> List[Product]:
        """
        Retrieve products from WooCommerce store.
        
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
            # Calculate page number for WooCommerce pagination
            page = (offset // limit) + 1
            
            # Fetch products from WooCommerce
            response = self.api.get("products", params={
                "per_page": min(limit, 100),  # WooCommerce max limit is 100
                "page": page,
                "status": "publish"
            })
            
            if response.status_code != 200:
                raise AdapterError(
                    f"Failed to fetch products: HTTP {response.status_code}",
                    adapter_type="woocommerce"
                )
            
            wc_products = response.json()
            products = []
            
            for wc_product in wc_products:
                try:
                    product = self._convert_woocommerce_product(wc_product)
                    products.append(product)
                except Exception as e:
                    logger.warning(f"Failed to convert WooCommerce product {wc_product.get('id')}: {str(e)}")
            
            logger.debug(f"Retrieved {len(products)} products from WooCommerce")
            return products
            
        except Exception as e:
            logger.error(f"Failed to get products from WooCommerce: {str(e)}")
            raise AdapterError(f"Failed to retrieve products: {str(e)}", adapter_type="woocommerce")
    
    async def get_product(self, product_id: str) -> Optional[Product]:
        """
        Retrieve a specific product by ID.
        
        Args:
            product_id: WooCommerce product ID
            
        Returns:
            Product object if found, None otherwise
        """
        self._ensure_connected()
        
        try:
            response = self.api.get(f"products/{product_id}")
            
            if response.status_code == 200:
                wc_product = response.json()
                return self._convert_woocommerce_product(wc_product)
            elif response.status_code == 404:
                return None
            else:
                raise AdapterError(
                    f"Failed to fetch product: HTTP {response.status_code}",
                    adapter_type="woocommerce"
                )
            
        except Exception as e:
            logger.error(f"Failed to get product {product_id} from WooCommerce: {str(e)}")
            return None
    
    async def sync_products(self) -> int:
        """
        Sync all products from WooCommerce store.
        
        Returns:
            Number of products synced
        """
        self._ensure_connected()
        
        try:
            total_synced = 0
            page = 1
            per_page = 100  # WooCommerce max limit
            
            while True:
                # Get products for current page
                response = self.api.get("products", params={
                    "per_page": per_page,
                    "page": page,
                    "status": "publish"
                })
                
                if response.status_code != 200:
                    raise AdapterError(
                        f"Failed to fetch products: HTTP {response.status_code}",
                        adapter_type="woocommerce"
                    )
                
                wc_products = response.json()
                
                if not wc_products:
                    break
                
                # Convert and count products
                for wc_product in wc_products:
                    try:
                        self._convert_woocommerce_product(wc_product)
                        total_synced += 1
                    except Exception as e:
                        logger.warning(f"Failed to sync product {wc_product.get('id')}: {str(e)}")
                
                # Check if we have more pages
                if len(wc_products) < per_page:
                    break
                
                page += 1
            
            logger.info(f"Synced {total_synced} products from WooCommerce")
            return total_synced
            
        except Exception as e:
            logger.error(f"Failed to sync products from WooCommerce: {str(e)}")
            raise AdapterError(f"Product sync failed: {str(e)}", adapter_type="woocommerce")
    
    async def webhook_handler(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Handle webhook events from WooCommerce.
        
        Args:
            event_type: Type of webhook event
            payload: Webhook payload
            
        Returns:
            True if event handled successfully
        """
        try:
            logger.debug(f"Handling WooCommerce webhook: {event_type}")
            
            if event_type in ["product.created", "product.updated"]:
                # Convert and return the product for further processing
                wc_product = payload
                product = self._convert_woocommerce_product(wc_product)
                action = "created" if event_type == "product.created" else "updated"
                logger.info(f"Product {action}: {product.id}")
                return True
                
            elif event_type == "product.deleted":
                product_id = str(payload.get("id"))
                logger.info(f"Product deleted: {product_id}")
                return True
                
            else:
                logger.debug(f"Unhandled webhook event type: {event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to handle WooCommerce webhook {event_type}: {str(e)}")
            return False
    
    def _convert_woocommerce_product(self, wc_product: Dict[str, Any]) -> Product:
        """
        Convert WooCommerce product data to KSE Product model.
        
        Args:
            wc_product: WooCommerce product data
            
        Returns:
            Product object
        """
        try:
            # Extract basic product information
            product_id = str(wc_product["id"])
            title = wc_product.get("name", "")
            description = wc_product.get("description", "")
            
            # Clean HTML from description
            import re
            description = re.sub(r'<[^>]+>', '', description)
            
            # Extract pricing
            price = None
            if wc_product.get("price"):
                price = float(wc_product["price"])
            
            # Extract images
            images = []
            for image in wc_product.get("images", []):
                if image.get("src"):
                    images.append(image["src"])
            
            # Extract tags
            tags = []
            for tag in wc_product.get("tags", []):
                if tag.get("name"):
                    tags.append(tag["name"])
            
            # Extract categories
            categories = []
            for category in wc_product.get("categories", []):
                if category.get("name"):
                    categories.append(category["name"])
            
            category = categories[0] if categories else None
            
            # Extract attributes as additional tags
            for attribute in wc_product.get("attributes", []):
                if attribute.get("options"):
                    tags.extend(attribute["options"])
            
            # Extract variations as variants
            variants = []
            if wc_product.get("variations"):
                # Note: In a real implementation, you'd fetch variation details
                # For now, we'll create a basic variant structure
                variants.append({
                    "id": product_id,
                    "title": "Default",
                    "price": price or 0,
                    "sku": wc_product.get("sku", ""),
                    "inventory_quantity": wc_product.get("stock_quantity"),
                })
            
            # Parse timestamps
            created_at = datetime.utcnow()
            updated_at = datetime.utcnow()
            
            if wc_product.get("date_created"):
                try:
                    created_at = datetime.fromisoformat(
                        wc_product["date_created"].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            if wc_product.get("date_modified"):
                try:
                    updated_at = datetime.fromisoformat(
                        wc_product["date_modified"].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            # Create Product object
            product = Product(
                id=product_id,
                title=title,
                description=description,
                price=price,
                currency="USD",  # Default, could be extracted from store settings
                category=category,
                brand=None,  # WooCommerce doesn't have a built-in brand field
                tags=tags,
                images=images,
                variants=variants,
                metadata={
                    "woocommerce_id": product_id,
                    "woocommerce_slug": wc_product.get("slug"),
                    "woocommerce_status": wc_product.get("status"),
                    "woocommerce_type": wc_product.get("type"),
                    "woocommerce_sku": wc_product.get("sku"),
                    "woocommerce_categories": categories,
                    "woocommerce_weight": wc_product.get("weight"),
                    "woocommerce_dimensions": wc_product.get("dimensions"),
                },
                created_at=created_at,
                updated_at=updated_at,
            )
            
            return product
            
        except Exception as e:
            raise AdapterError(
                f"Failed to convert WooCommerce product: {str(e)}",
                adapter_type="woocommerce",
                details={"woocommerce_product_id": wc_product.get("id")}
            )
    
    def _ensure_connected(self):
        """Ensure the adapter is connected."""
        if not self._connected:
            raise AdapterError("Not connected to WooCommerce. Call connect() first.", adapter_type="woocommerce")