"""
Conceptual service for KSE Memory SDK.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.interfaces import ConceptualServiceInterface
from ..core.config import ConceptualConfig
from ..core.models import Product, ConceptualDimensions
from ..exceptions import ConceptualError


logger = logging.getLogger(__name__)


class ConceptualService(ConceptualServiceInterface):
    """
    Conceptual dimension computation service for KSE Memory SDK.
    
    Uses LLMs to automatically compute conceptual dimensions for products
    based on their descriptions, categories, and other attributes.
    """
    
    def __init__(self, config: ConceptualConfig):
        """
        Initialize conceptual service.
        
        Args:
            config: Conceptual configuration
        """
        self.config = config
        self._initialized = False
        
        # Standard conceptual dimensions with descriptions
        self.dimension_descriptions = {
            "elegance": "How refined, graceful, and sophisticated the product appears",
            "comfort": "How comfortable, cozy, and pleasant the product is to use",
            "boldness": "How striking, daring, and attention-grabbing the product is",
            "modernity": "How contemporary, current, and up-to-date the product feels",
            "minimalism": "How simple, clean, and uncluttered the product design is",
            "luxury": "How premium, exclusive, and high-end the product appears",
            "functionality": "How practical, useful, and purpose-driven the product is",
            "versatility": "How adaptable and suitable for multiple uses the product is",
            "seasonality": "How tied to specific seasons or weather the product is",
            "innovation": "How novel, creative, and technologically advanced the product is"
        }
        
        logger.info(f"Conceptual service initialized with {len(self.config.dimensions)} dimensions")
    
    async def _initialize_llm(self):
        """Initialize LLM for conceptual computation."""
        if self._initialized:
            return
        
        if not self.config.auto_compute:
            self._initialized = True
            return
        
        if not OPENAI_AVAILABLE:
            raise ConceptualError("OpenAI package not available. Install with: pip install openai")
        
        if not self.config.llm_api_key:
            raise ConceptualError("LLM API key required for automatic conceptual dimension computation")
        
        openai.api_key = self.config.llm_api_key
        self._initialized = True
        
        logger.info("LLM initialized for conceptual dimension computation")
    
    async def compute_dimensions(self, product: Product) -> ConceptualDimensions:
        """
        Compute conceptual dimensions for a product.
        
        Args:
            product: Product to analyze
            
        Returns:
            ConceptualDimensions object
            
        Raises:
            ConceptualError: If computation fails
        """
        await self._initialize_llm()
        
        try:
            if not self.config.auto_compute:
                # Return default dimensions if auto-compute is disabled
                return ConceptualDimensions()
            
            # Create prompt for LLM
            prompt = self._create_analysis_prompt(product)
            
            # Get LLM response
            response = await self._query_llm(prompt)
            
            # Parse response into dimensions
            dimensions = self._parse_llm_response(response)
            
            logger.debug(f"Computed conceptual dimensions for product {product.id}")
            return dimensions
            
        except Exception as e:
            logger.error(f"Failed to compute conceptual dimensions for product {product.id}: {str(e)}")
            raise ConceptualError(f"Dimension computation failed: {str(e)}", product_id=product.id)
    
    async def compute_batch_dimensions(self, products: List[Product]) -> List[ConceptualDimensions]:
        """
        Compute conceptual dimensions for multiple products.
        
        Args:
            products: List of products to analyze
            
        Returns:
            List of ConceptualDimensions objects
            
        Raises:
            ConceptualError: If batch computation fails
        """
        await self._initialize_llm()
        
        try:
            if not products:
                return []
            
            if not self.config.auto_compute:
                # Return default dimensions for all products
                return [ConceptualDimensions() for _ in products]
            
            dimensions_list = []
            
            # Process in batches to avoid API limits
            batch_size = 5  # Conservative batch size for LLM processing
            
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [self.compute_dimensions(product) for product in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results and exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Failed to compute dimensions for product {batch[j].id}: {str(result)}")
                        dimensions_list.append(ConceptualDimensions())  # Default dimensions
                    else:
                        dimensions_list.append(result)
            
            logger.debug(f"Computed conceptual dimensions for {len(products)} products")
            return dimensions_list
            
        except Exception as e:
            logger.error(f"Failed to compute batch conceptual dimensions: {str(e)}")
            raise ConceptualError(f"Batch dimension computation failed: {str(e)}")
    
    async def explain_dimensions(self, product: Product, dimensions: ConceptualDimensions) -> str:
        """
        Explain why specific dimensions were assigned.
        
        Args:
            product: Product that was analyzed
            dimensions: Computed dimensions
            
        Returns:
            Explanation text
            
        Raises:
            ConceptualError: If explanation generation fails
        """
        await self._initialize_llm()
        
        try:
            if not self.config.auto_compute:
                return "Automatic dimension computation is disabled."
            
            # Create explanation prompt
            prompt = self._create_explanation_prompt(product, dimensions)
            
            # Get LLM response
            explanation = await self._query_llm(prompt)
            
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate explanation for product {product.id}: {str(e)}")
            raise ConceptualError(f"Explanation generation failed: {str(e)}", product_id=product.id)
    
    def _create_analysis_prompt(self, product: Product) -> str:
        """Create prompt for LLM analysis."""
        
        # Build product context
        context_parts = [
            f"Title: {product.title}",
            f"Description: {product.description}",
        ]
        
        if product.category:
            context_parts.append(f"Category: {product.category}")
        
        if product.brand:
            context_parts.append(f"Brand: {product.brand}")
        
        if product.tags:
            context_parts.append(f"Tags: {', '.join(product.tags)}")
        
        if product.price:
            context_parts.append(f"Price: ${product.price}")
        
        product_context = "\n".join(context_parts)
        
        # Create dimension descriptions
        dimension_desc = "\n".join([
            f"- {dim}: {desc} (scale 0.0-1.0)"
            for dim, desc in self.dimension_descriptions.items()
            if dim in self.config.dimensions
        ])
        
        prompt = f"""
Analyze the following product and assign conceptual dimension scores on a scale of 0.0 to 1.0:

PRODUCT:
{product_context}

DIMENSIONS TO SCORE:
{dimension_desc}

Please analyze the product carefully and provide scores for each dimension. Consider:
- The product's intended use and target audience
- Design aesthetics and visual appeal
- Functional characteristics
- Market positioning and brand perception
- Cultural and social associations

Respond with ONLY a JSON object containing the dimension scores, like this:
{{
    "elegance": 0.7,
    "comfort": 0.8,
    "boldness": 0.3,
    "modernity": 0.6,
    "minimalism": 0.4,
    "luxury": 0.5,
    "functionality": 0.9,
    "versatility": 0.7,
    "seasonality": 0.2,
    "innovation": 0.4
}}
"""
        
        return prompt.strip()
    
    def _create_explanation_prompt(self, product: Product, dimensions: ConceptualDimensions) -> str:
        """Create prompt for dimension explanation."""
        
        # Get top dimensions
        dim_dict = dimensions.to_dict()
        sorted_dims = sorted(dim_dict.items(), key=lambda x: x[1], reverse=True)
        top_dims = sorted_dims[:3]
        
        top_dims_text = ", ".join([f"{dim} ({score:.2f})" for dim, score in top_dims])
        
        prompt = f"""
Explain why the product "{product.title}" received these conceptual dimension scores:

Top dimensions: {top_dims_text}

Product details:
- Description: {product.description}
- Category: {product.category or 'Not specified'}
- Brand: {product.brand or 'Not specified'}
- Tags: {', '.join(product.tags) if product.tags else 'None'}

Provide a brief, clear explanation (2-3 sentences) of why these particular dimensions scored highest for this product. Focus on the product's key characteristics that led to these scores.
"""
        
        return prompt.strip()
    
    async def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the given prompt."""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert product analyst specializing in conceptual dimension analysis. You provide accurate, objective assessments of products across multiple perceptual dimensions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise ConceptualError(f"LLM query failed: {str(e)}")
    
    def _parse_llm_response(self, response: str) -> ConceptualDimensions:
        """Parse LLM response into ConceptualDimensions."""
        try:
            # Extract JSON from response
            response = response.strip()
            
            # Find JSON object in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx]
            dimension_scores = json.loads(json_str)
            
            # Validate and clamp scores
            validated_scores = {}
            for dim in self.config.dimensions:
                if dim in dimension_scores:
                    score = float(dimension_scores[dim])
                    # Clamp to valid range
                    score = max(0.0, min(1.0, score))
                    validated_scores[dim] = score
                else:
                    # Default score if dimension not provided
                    validated_scores[dim] = 0.5
            
            return ConceptualDimensions(**validated_scores)
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {str(e)}. Using default dimensions.")
            # Return default dimensions if parsing fails
            return ConceptualDimensions()
    
    def compute_similarity(self, dims1: ConceptualDimensions, dims2: ConceptualDimensions) -> float:
        """
        Compute similarity between two sets of conceptual dimensions.
        
        Args:
            dims1: First set of dimensions
            dims2: Second set of dimensions
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            dict1 = dims1.to_dict()
            dict2 = dims2.to_dict()
            
            # Compute cosine similarity
            dot_product = sum(dict1[dim] * dict2[dim] for dim in self.config.dimensions)
            
            norm1 = sum(dict1[dim] ** 2 for dim in self.config.dimensions) ** 0.5
            norm2 = sum(dict2[dim] ** 2 for dim in self.config.dimensions) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in valid range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Failed to compute conceptual similarity: {str(e)}")
            return 0.0
    
    def get_dimension_weights(self, query_context: str = None) -> Dict[str, float]:
        """
        Get dimension weights based on query context.
        
        Args:
            query_context: Optional context for weighting dimensions
            
        Returns:
            Dictionary of dimension weights
        """
        # Default equal weights
        weights = {dim: 1.0 for dim in self.config.dimensions}
        
        if query_context:
            # Simple keyword-based weighting
            context_lower = query_context.lower()
            
            # Boost relevant dimensions based on keywords
            if any(word in context_lower for word in ["elegant", "sophisticated", "refined"]):
                weights["elegance"] = 1.5
            
            if any(word in context_lower for word in ["comfortable", "cozy", "soft"]):
                weights["comfort"] = 1.5
            
            if any(word in context_lower for word in ["bold", "striking", "dramatic"]):
                weights["boldness"] = 1.5
            
            if any(word in context_lower for word in ["modern", "contemporary", "current"]):
                weights["modernity"] = 1.5
            
            if any(word in context_lower for word in ["minimal", "simple", "clean"]):
                weights["minimalism"] = 1.5
            
            if any(word in context_lower for word in ["luxury", "premium", "high-end"]):
                weights["luxury"] = 1.5
            
            if any(word in context_lower for word in ["functional", "practical", "useful"]):
                weights["functionality"] = 1.5
            
            if any(word in context_lower for word in ["versatile", "adaptable", "flexible"]):
                weights["versatility"] = 1.5
            
            if any(word in context_lower for word in ["seasonal", "summer", "winter", "spring", "fall"]):
                weights["seasonality"] = 1.5
            
            if any(word in context_lower for word in ["innovative", "novel", "advanced"]):
                weights["innovation"] = 1.5
        
        return weights