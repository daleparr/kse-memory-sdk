"""
Embedding service for KSE Memory SDK.
"""

import asyncio
import logging
from typing import List, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image
    import requests
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from ..core.interfaces import EmbeddingServiceInterface
from ..core.config import EmbeddingConfig
from ..core.models import EmbeddingVector
from ..exceptions import EmbeddingError


logger = logging.getLogger(__name__)


class EmbeddingService(EmbeddingServiceInterface):
    """
    Embedding generation service for KSE Memory SDK.
    
    Supports multiple embedding models including Sentence Transformers,
    OpenAI embeddings, and CLIP for image embeddings.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding service.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self._text_model = None
        self._image_model = None
        self._clip_processor = None
        self._initialized = False
        
        logger.info(f"Embedding service initialized with text model: {config.text_model}")
    
    async def _initialize_models(self):
        """Initialize embedding models lazily."""
        if self._initialized:
            return
        
        try:
            # Initialize text embedding model
            await self._initialize_text_model()
            
            # Initialize image embedding model
            await self._initialize_image_model()
            
            self._initialized = True
            logger.info("Embedding models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {str(e)}")
            raise EmbeddingError(f"Model initialization failed: {str(e)}")
    
    async def _initialize_text_model(self):
        """Initialize text embedding model."""
        if self.config.text_model.startswith("text-embedding"):
            # OpenAI embedding model
            if not OPENAI_AVAILABLE:
                raise EmbeddingError("OpenAI package not available. Install with: pip install openai")
            
            if not self.config.openai_api_key:
                raise EmbeddingError("OpenAI API key required for OpenAI embedding models")
            
            openai.api_key = self.config.openai_api_key
            self._text_model = "openai"
            
        elif self.config.text_model.startswith("sentence-transformers"):
            # Sentence Transformers model
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise EmbeddingError("Sentence Transformers not available. Install with: pip install sentence-transformers")
            
            model_name = self.config.text_model
            self._text_model = SentenceTransformer(model_name)
            
        else:
            raise EmbeddingError(f"Unsupported text embedding model: {self.config.text_model}")
    
    async def _initialize_image_model(self):
        """Initialize image embedding model."""
        if self.config.image_model.startswith("openai/clip"):
            # CLIP model
            if not CLIP_AVAILABLE:
                raise EmbeddingError("CLIP dependencies not available. Install with: pip install transformers torch pillow")
            
            model_name = self.config.image_model.replace("openai/", "")
            self._image_model = CLIPModel.from_pretrained(model_name)
            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
            
        else:
            logger.warning(f"Unsupported image embedding model: {self.config.image_model}")
    
    async def generate_text_embedding(self, text: str) -> EmbeddingVector:
        """
        Generate text embedding.
        
        Args:
            text: Input text
            
        Returns:
            EmbeddingVector object
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        await self._initialize_models()
        
        try:
            if not text or not text.strip():
                raise EmbeddingError("Empty text provided for embedding")
            
            # Truncate text if too long
            if len(text) > self.config.max_length:
                text = text[:self.config.max_length]
            
            if self._text_model == "openai":
                # Use OpenAI API
                response = await self._generate_openai_embedding(text)
                vector = response["data"][0]["embedding"]
                dimension = len(vector)
                
            elif isinstance(self._text_model, SentenceTransformer):
                # Use Sentence Transformers
                vector = self._text_model.encode(text, normalize_embeddings=self.config.normalize)
                vector = vector.tolist()
                dimension = len(vector)
                
            else:
                raise EmbeddingError("No text embedding model available")
            
            return EmbeddingVector(
                vector=vector,
                model=self.config.text_model,
                dimension=dimension,
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {str(e)}")
            raise EmbeddingError(f"Text embedding generation failed: {str(e)}", model=self.config.text_model)
    
    async def generate_image_embedding(self, image_url: str) -> EmbeddingVector:
        """
        Generate image embedding.
        
        Args:
            image_url: URL of the image
            
        Returns:
            EmbeddingVector object
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        await self._initialize_models()
        
        try:
            if not self._image_model:
                raise EmbeddingError("No image embedding model available")
            
            # Download and process image
            image = await self._download_image(image_url)
            
            # Generate embedding using CLIP
            inputs = self._clip_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self._image_model.get_image_features(**inputs)
                
                if self.config.normalize:
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
                
                vector = image_features.squeeze().tolist()
            
            return EmbeddingVector(
                vector=vector,
                model=self.config.image_model,
                dimension=len(vector),
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {str(e)}")
            raise EmbeddingError(f"Image embedding generation failed: {str(e)}", model=self.config.image_model)
    
    async def generate_batch_text_embeddings(self, texts: List[str]) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of EmbeddingVector objects
            
        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        await self._initialize_models()
        
        try:
            if not texts:
                return []
            
            # Filter and truncate texts
            processed_texts = []
            for text in texts:
                if text and text.strip():
                    if len(text) > self.config.max_length:
                        text = text[:self.config.max_length]
                    processed_texts.append(text)
            
            if not processed_texts:
                return []
            
            embeddings = []
            
            if self._text_model == "openai":
                # Process in batches for OpenAI API
                batch_size = min(self.config.batch_size, 100)  # OpenAI limit
                
                for i in range(0, len(processed_texts), batch_size):
                    batch = processed_texts[i:i + batch_size]
                    response = await self._generate_openai_embeddings_batch(batch)
                    
                    for j, embedding_data in enumerate(response["data"]):
                        vector = embedding_data["embedding"]
                        embeddings.append(EmbeddingVector(
                            vector=vector,
                            model=self.config.text_model,
                            dimension=len(vector),
                            created_at=datetime.utcnow()
                        ))
                        
            elif isinstance(self._text_model, SentenceTransformer):
                # Use Sentence Transformers batch processing
                vectors = self._text_model.encode(
                    processed_texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize,
                    show_progress_bar=len(processed_texts) > 100
                )
                
                for vector in vectors:
                    embeddings.append(EmbeddingVector(
                        vector=vector.tolist(),
                        model=self.config.text_model,
                        dimension=len(vector),
                        created_at=datetime.utcnow()
                    ))
            
            else:
                raise EmbeddingError("No text embedding model available")
            
            logger.debug(f"Generated {len(embeddings)} text embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch text embeddings: {str(e)}")
            raise EmbeddingError(f"Batch text embedding generation failed: {str(e)}", model=self.config.text_model)
    
    async def generate_batch_image_embeddings(self, image_urls: List[str]) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple images.
        
        Args:
            image_urls: List of image URLs
            
        Returns:
            List of EmbeddingVector objects
            
        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        await self._initialize_models()
        
        try:
            if not image_urls:
                return []
            
            if not self._image_model:
                raise EmbeddingError("No image embedding model available")
            
            embeddings = []
            
            # Process images in batches
            batch_size = min(self.config.batch_size, 32)  # Reasonable batch size for images
            
            for i in range(0, len(image_urls), batch_size):
                batch_urls = image_urls[i:i + batch_size]
                
                # Download images
                images = []
                for url in batch_urls:
                    try:
                        image = await self._download_image(url)
                        images.append(image)
                    except Exception as e:
                        logger.warning(f"Failed to download image {url}: {str(e)}")
                        images.append(None)
                
                # Process valid images
                valid_images = [img for img in images if img is not None]
                if not valid_images:
                    continue
                
                # Generate embeddings
                inputs = self._clip_processor(images=valid_images, return_tensors="pt")
                
                with torch.no_grad():
                    image_features = self._image_model.get_image_features(**inputs)
                    
                    if self.config.normalize:
                        image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
                    
                    for features in image_features:
                        vector = features.tolist()
                        embeddings.append(EmbeddingVector(
                            vector=vector,
                            model=self.config.image_model,
                            dimension=len(vector),
                            created_at=datetime.utcnow()
                        ))
            
            logger.debug(f"Generated {len(embeddings)} image embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch image embeddings: {str(e)}")
            raise EmbeddingError(f"Batch image embedding generation failed: {str(e)}", model=self.config.image_model)
    
    async def _generate_openai_embedding(self, text: str) -> dict:
        """Generate single OpenAI embedding."""
        response = await openai.Embedding.acreate(
            model=self.config.text_model,
            input=text
        )
        return response
    
    async def _generate_openai_embeddings_batch(self, texts: List[str]) -> dict:
        """Generate batch OpenAI embeddings."""
        response = await openai.Embedding.acreate(
            model=self.config.text_model,
            input=texts
        )
        return response
    
    async def _download_image(self, image_url: str) -> Image.Image:
        """Download and process image from URL."""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(requests.get(image_url, stream=True).raw)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise EmbeddingError(f"Failed to download image from {image_url}: {str(e)}")