"""
Generic adapter for KSE Memory SDK.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from ..core.interfaces import AdapterInterface
from ..core.models import Product
from ..exceptions import AdapterError


logger = logging.getLogger(__name__)


class GenericAdapter(AdapterInterface):
    """
    Generic platform adapter for KSE Memory SDK.
    
    Provides a flexible interface for custom data sources and platforms
    that don't have dedicated adapters.
    """
    
    def __init__(self):
        """Initialize generic adapter."""
        self._connected = False
        self._data_source: Optional[Callable] = None
        self._product_converter: Optional[Callable] = None
        self._webhook_handlers: Dict[str, Callable] = {}
        
        logger.info("Generic adapter initialized")
    
    async def connect(self, config: Dict[str, Any]) -> bool:
        """
        Connect to the data source.
        
        Args:
            config: Configuration dictionary with keys:
                - data_source: Callable that returns product data
                - product_converter: Callable to convert raw data to Product objects
                - webhook_handlers: Dict of event_type -> handler function
                
        Returns:
            True if connection successful
            
        Raises:
            AdapterError: If connection fails
        """
        try:
            # Validate required config
            if "data_source" not in config:
                raise AdapterError(
                    "Missing required 'data_source' in configuration",
                    adapter_type="generic"
                )
            
            self._data_source = config["data_source"]
            self._product_converter = config.get("product_converter", self._default_converter)
            self._webhook_handlers = config.get("webhook_handlers", {})
            
            # Test the data source
            try:
                test_data = await self._call_data_source(limit=1)
                logger.info(f"Connected to generic data source, test returned {len(test_data)} items")
                self._connected = True
                return True
                
            except Exception as e:
                raise AdapterError(
                    f"Failed to test data source: {str(e)}",
                    adapter_type="generic"
                )
                
        except Exception as e:
            logger.error(f"Failed to connect to generic data source: {str(e)}")
            if isinstance(e, AdapterError):
                raise
            raise AdapterError(f"Connection failed: {str(e)}", adapter_type="generic")
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the data source.
        
        Returns:
            True if disconnection successful
        """
        try:
            self._data_source = None
            self._product_converter = None
            self._webhook_handlers = {}
            self._connected = False
            logger.info("Disconnected from generic data source")
            return True
            
        except Exception as e:
            logger.error(f"Error during generic adapter disconnection: {str(e)}")
            return False
    
    async def get_products(self, limit: int = 100, offset: int = 0) -> List[Product]:
        """
        Retrieve products from the data source.
        
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
            # Get raw data from the data source
            raw_data = await self._call_data_source(limit=limit, offset=offset)
            
            # Convert to Product objects
            products = []
            for item in raw_data:
                try:
                    product = await self._convert_to_product(item)
                    products.append(product)
                except Exception as e:
                    logger.warning(f"Failed to convert item to product: {str(e)}")
            
            logger.debug(f"Retrieved {len(products)} products from generic data source")
            return products
            
        except Exception as e:
            logger.error(f"Failed to get products from generic data source: {str(e)}")
            raise AdapterError(f"Failed to retrieve products: {str(e)}", adapter_type="generic")
    
    async def get_product(self, product_id: str) -> Optional[Product]:
        """
        Retrieve a specific product by ID.
        
        Args:
            product_id: Product ID
            
        Returns:
            Product object if found, None otherwise
        """
        self._ensure_connected()
        
        try:
            # Try to get specific product from data source
            raw_data = await self._call_data_source(product_id=product_id)
            
            if raw_data:
                # Assume the first item is the requested product
                item = raw_data[0] if isinstance(raw_data, list) else raw_data
                return await self._convert_to_product(item)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get product {product_id} from generic data source: {str(e)}")
            return None
    
    async def sync_products(self) -> int:
        """
        Sync all products from the data source.
        
        Returns:
            Number of products synced
        """
        self._ensure_connected()
        
        try:
            total_synced = 0
            batch_size = 100
            offset = 0
            
            while True:
                # Get batch of products
                products = await self.get_products(limit=batch_size, offset=offset)
                
                if not products:
                    break
                
                total_synced += len(products)
                offset += batch_size
                
                # If we got fewer products than the batch size, we're done
                if len(products) < batch_size:
                    break
            
            logger.info(f"Synced {total_synced} products from generic data source")
            return total_synced
            
        except Exception as e:
            logger.error(f"Failed to sync products from generic data source: {str(e)}")
            raise AdapterError(f"Product sync failed: {str(e)}", adapter_type="generic")
    
    async def webhook_handler(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """
        Handle webhook events.
        
        Args:
            event_type: Type of webhook event
            payload: Webhook payload
            
        Returns:
            True if event handled successfully
        """
        try:
            logger.debug(f"Handling generic webhook: {event_type}")
            
            # Check if we have a handler for this event type
            if event_type in self._webhook_handlers:
                handler = self._webhook_handlers[event_type]
                
                # Call the handler
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(payload)
                else:
                    result = handler(payload)
                
                return bool(result)
            else:
                logger.debug(f"No handler registered for event type: {event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to handle generic webhook {event_type}: {str(e)}")
            return False
    
    async def _call_data_source(self, **kwargs) -> Any:
        """
        Call the data source function with the given parameters.
        
        Args:
            **kwargs: Parameters to pass to the data source
            
        Returns:
            Data from the data source
        """
        if not self._data_source:
            raise AdapterError("No data source configured", adapter_type="generic")
        
        try:
            if asyncio.iscoroutinefunction(self._data_source):
                return await self._data_source(**kwargs)
            else:
                return self._data_source(**kwargs)
        except Exception as e:
            raise AdapterError(f"Data source call failed: {str(e)}", adapter_type="generic")
    
    async def _convert_to_product(self, raw_item: Any) -> Product:
        """
        Convert raw data item to Product object.
        
        Args:
            raw_item: Raw data item
            
        Returns:
            Product object
        """
        try:
            if asyncio.iscoroutinefunction(self._product_converter):
                return await self._product_converter(raw_item)
            else:
                return self._product_converter(raw_item)
        except Exception as e:
            raise AdapterError(f"Product conversion failed: {str(e)}", adapter_type="generic")
    
    def _default_converter(self, raw_item: Any) -> Product:
        """
        Default converter that assumes the raw item is already a Product or dict.
        
        Args:
            raw_item: Raw data item
            
        Returns:
            Product object
        """
        if isinstance(raw_item, Product):
            return raw_item
        elif isinstance(raw_item, dict):
            return Product.from_dict(raw_item)
        else:
            # Try to convert to dict first
            try:
                if hasattr(raw_item, '__dict__'):
                    item_dict = raw_item.__dict__
                elif hasattr(raw_item, 'to_dict'):
                    item_dict = raw_item.to_dict()
                else:
                    raise ValueError(f"Cannot convert item of type {type(raw_item)} to Product")
                
                return Product.from_dict(item_dict)
                
            except Exception as e:
                raise AdapterError(
                    f"Default converter failed: {str(e)}",
                    adapter_type="generic",
                    details={"item_type": str(type(raw_item))}
                )
    
    def _ensure_connected(self):
        """Ensure the adapter is connected."""
        if not self._connected:
            raise AdapterError("Not connected to data source. Call connect() first.", adapter_type="generic")
    
    # Utility methods for common use cases
    
    @staticmethod
    def create_csv_data_source(file_path: str) -> Callable:
        """
        Create a data source function for CSV files.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Data source function
        """
        def csv_data_source(limit: int = 100, offset: int = 0, **kwargs):
            import pandas as pd
            
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Apply pagination
                end_idx = offset + limit
                paginated_df = df.iloc[offset:end_idx]
                
                # Convert to list of dictionaries
                return paginated_df.to_dict('records')
                
            except Exception as e:
                raise AdapterError(f"Failed to read CSV file {file_path}: {str(e)}")
        
        return csv_data_source
    
    @staticmethod
    def create_json_data_source(file_path: str, products_key: str = "products") -> Callable:
        """
        Create a data source function for JSON files.
        
        Args:
            file_path: Path to the JSON file
            products_key: Key in JSON that contains the products array
            
        Returns:
            Data source function
        """
        def json_data_source(limit: int = 100, offset: int = 0, **kwargs):
            import json
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract products array
                if products_key in data:
                    products = data[products_key]
                elif isinstance(data, list):
                    products = data
                else:
                    raise ValueError(f"Cannot find products in JSON file. Expected key: {products_key}")
                
                # Apply pagination
                end_idx = offset + limit
                return products[offset:end_idx]
                
            except Exception as e:
                raise AdapterError(f"Failed to read JSON file {file_path}: {str(e)}")
        
        return json_data_source
    
    @staticmethod
    def create_api_data_source(base_url: str, headers: Dict[str, str] = None) -> Callable:
        """
        Create a data source function for REST APIs.
        
        Args:
            base_url: Base URL of the API
            headers: HTTP headers to include in requests
            
        Returns:
            Data source function
        """
        async def api_data_source(limit: int = 100, offset: int = 0, product_id: str = None, **kwargs):
            import httpx
            
            try:
                async with httpx.AsyncClient() as client:
                    if product_id:
                        # Get specific product
                        url = f"{base_url}/products/{product_id}"
                        response = await client.get(url, headers=headers or {})
                        response.raise_for_status()
                        return response.json()
                    else:
                        # Get paginated products
                        url = f"{base_url}/products"
                        params = {"limit": limit, "offset": offset}
                        response = await client.get(url, headers=headers or {}, params=params)
                        response.raise_for_status()
                        data = response.json()
                        
                        # Handle different response formats
                        if isinstance(data, list):
                            return data
                        elif "products" in data:
                            return data["products"]
                        elif "data" in data:
                            return data["data"]
                        else:
                            return data
                        
            except Exception as e:
                raise AdapterError(f"Failed to fetch data from API {base_url}: {str(e)}")
        
        return api_data_source