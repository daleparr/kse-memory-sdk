"""
Platform adapters for KSE Memory SDK.
"""

from typing import Dict, Any
from ..core.interfaces import AdapterInterface
from ..exceptions import AdapterError


def get_adapter(adapter_type: str) -> AdapterInterface:
    """
    Factory function to get the appropriate adapter.
    
    Args:
        adapter_type: Type of adapter ('shopify', 'woocommerce', 'magento', 'generic')
        
    Returns:
        Adapter instance
        
    Raises:
        AdapterError: If adapter type is not supported
    """
    adapter_type = adapter_type.lower()
    
    if adapter_type == "shopify":
        from .shopify import ShopifyAdapter
        return ShopifyAdapter()
    elif adapter_type == "woocommerce":
        from .woocommerce import WooCommerceAdapter
        return WooCommerceAdapter()
    elif adapter_type == "magento":
        from .magento import MagentoAdapter
        return MagentoAdapter()
    elif adapter_type == "generic":
        from .generic import GenericAdapter
        return GenericAdapter()
    else:
        raise AdapterError(f"Unsupported adapter type: {adapter_type}")


# Import main adapter classes for direct access
from .shopify import ShopifyAdapter
from .woocommerce import WooCommerceAdapter
from .generic import GenericAdapter

__all__ = [
    "get_adapter",
    "ShopifyAdapter",
    "WooCommerceAdapter", 
    "GenericAdapter",
]