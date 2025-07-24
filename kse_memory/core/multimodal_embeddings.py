"""
Multi-Modal Embedding Engine for KSE Memory SDK

This module implements universal representation spaces for multimodal inputs
including text, vision, audio, and structured data with cross-modal similarity.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import base64
from datetime import datetime

# Import for different modality processing
try:
    import cv2
    import librosa
    from PIL import Image
    import torchvision.transforms as transforms
    from transformers import (
        AutoTokenizer, AutoModel, 
        CLIPProcessor, CLIPModel,
        Wav2Vec2Processor, Wav2Vec2Model
    )
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    logging.warning("Some multimodal dependencies not available. Install with: pip install opencv-python librosa pillow transformers")

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of modalities supported."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"
    TEMPORAL = "temporal"


class FusionStrategy(Enum):
    """Strategies for multi-modal fusion."""
    CONCATENATION = "concatenation"
    ATTENTION_WEIGHTED = "attention_weighted"
    CROSS_MODAL_ATTENTION = "cross_modal_attention"
    TRANSFORMER_FUSION = "transformer_fusion"
    CONTRASTIVE_LEARNING = "contrastive_learning"


@dataclass
class ModalityEmbedding:
    """Represents an embedding for a specific modality."""
    
    modality: ModalityType
    embedding: np.ndarray
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Normalize embedding after initialization."""
        if len(self.embedding.shape) == 1:
            # L2 normalize
            norm = np.linalg.norm(self.embedding)
            if norm > 0:
                self.embedding = self.embedding / norm


@dataclass
class MultiModalEmbedding:
    """Combined multi-modal embedding."""
    
    modality_embeddings: Dict[ModalityType, ModalityEmbedding]
    fused_embedding: np.ndarray
    fusion_strategy: FusionStrategy
    confidence_scores: Dict[ModalityType, float]
    alignment_scores: Dict[Tuple[ModalityType, ModalityType], float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_modalities(self) -> List[ModalityType]:
        """Get list of available modalities."""
        return list(self.modality_embeddings.keys())
    
    def get_cross_modal_similarity(self, other: 'MultiModalEmbedding') -> Dict[str, float]:
        """Calculate cross-modal similarities with another embedding."""
        similarities = {}
        
        for mod1 in self.modality_embeddings:
            for mod2 in other.modality_embeddings:
                if mod1 != mod2:  # Cross-modal only
                    emb1 = self.modality_embeddings[mod1].embedding
                    emb2 = other.modality_embeddings[mod2].embedding
                    
                    # Cosine similarity
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities[f"{mod1.value}_to_{mod2.value}"] = float(similarity)
        
        return similarities


class ModalityEncoder(ABC):
    """Abstract base class for modality encoders."""
    
    @abstractmethod
    async def encode(self, data: Any, **kwargs) -> ModalityEmbedding:
        """Encode data into embedding."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        pass
    
    @abstractmethod
    def supports_batch(self) -> bool:
        """Whether this encoder supports batch processing."""
        pass


class TextEncoder(ModalityEncoder):
    """Text encoder using transformer models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if MULTIMODAL_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        else:
            raise ImportError("Transformers not available. Install with: pip install transformers")
    
    async def encode(self, text: str, **kwargs) -> ModalityEmbedding:
        """Encode text into embedding."""
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                # Mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1)
        
        embedding_np = embedding.cpu().numpy().flatten()
        
        return ModalityEmbedding(
            modality=ModalityType.TEXT,
            embedding=embedding_np,
            confidence=1.0,  # Could be computed based on attention weights
            metadata={
                "text_length": len(text),
                "model": self.model_name,
                "truncated": len(text) > 512
            }
        )
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return 384  # Default for MiniLM
    
    def supports_batch(self) -> bool:
        """Supports batch processing."""
        return True


class ImageEncoder(ModalityEncoder):
    """Image encoder using CLIP or similar vision models."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if MULTIMODAL_AVAILABLE:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        else:
            raise ImportError("CLIP model not available. Install with: pip install transformers")
    
    async def encode(self, image_data: Union[str, np.ndarray, Image.Image], **kwargs) -> ModalityEmbedding:
        """Encode image into embedding."""
        
        # Handle different input formats
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                # Base64 encoded image
                image_data = self._decode_base64_image(image_data)
            else:
                # File path
                image_data = Image.open(image_data)
        elif isinstance(image_data, np.ndarray):
            image_data = Image.fromarray(image_data)
        
        # Process image
        inputs = self.processor(images=image_data, return_tensors="pt").to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        embedding_np = image_features.cpu().numpy().flatten()
        
        # Calculate confidence based on image quality metrics
        confidence = self._calculate_image_confidence(image_data)
        
        return ModalityEmbedding(
            modality=ModalityType.IMAGE,
            embedding=embedding_np,
            confidence=confidence,
            metadata={
                "image_size": image_data.size if hasattr(image_data, 'size') else None,
                "model": self.model_name,
                "channels": len(image_data.getbands()) if hasattr(image_data, 'getbands') else None
            }
        )
    
    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 image string."""
        header, data = base64_string.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(io.BytesIO(image_data))
    
    def _calculate_image_confidence(self, image: Image.Image) -> float:
        """Calculate confidence score based on image quality."""
        # Simple heuristic based on image size and variance
        if hasattr(image, 'size'):
            width, height = image.size
            area = width * height
            
            # Convert to numpy for variance calculation
            img_array = np.array(image)
            variance = np.var(img_array)
            
            # Normalize confidence (0.5 to 1.0 range)
            size_score = min(1.0, area / (512 * 512))  # Normalize to 512x512
            variance_score = min(1.0, variance / 10000)  # Normalize variance
            
            return 0.5 + 0.25 * size_score + 0.25 * variance_score
        
        return 0.8  # Default confidence
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return 512  # CLIP dimension
    
    def supports_batch(self) -> bool:
        """Supports batch processing."""
        return True


class AudioEncoder(ModalityEncoder):
    """Audio encoder using Wav2Vec2 or similar audio models."""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if MULTIMODAL_AVAILABLE:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            self.model.eval()
        else:
            raise ImportError("Wav2Vec2 not available. Install with: pip install transformers librosa")
    
    async def encode(self, audio_data: Union[str, np.ndarray], **kwargs) -> ModalityEmbedding:
        """Encode audio into embedding."""
        
        # Handle different input formats
        if isinstance(audio_data, str):
            # File path
            audio_array, sample_rate = librosa.load(audio_data, sr=16000)
        else:
            audio_array = audio_data
            sample_rate = kwargs.get('sample_rate', 16000)
        
        # Ensure correct sample rate
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        
        # Process audio
        inputs = self.processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling over sequence dimension
            embedding = outputs.last_hidden_state.mean(dim=1)
        
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Calculate confidence based on audio quality
        confidence = self._calculate_audio_confidence(audio_array)
        
        return ModalityEmbedding(
            modality=ModalityType.AUDIO,
            embedding=embedding_np,
            confidence=confidence,
            metadata={
                "duration": len(audio_array) / 16000,
                "sample_rate": 16000,
                "model": self.model_name,
                "rms_energy": float(np.sqrt(np.mean(audio_array**2)))
            }
        )
    
    def _calculate_audio_confidence(self, audio: np.ndarray) -> float:
        """Calculate confidence based on audio quality."""
        # Simple heuristics
        rms_energy = np.sqrt(np.mean(audio**2))
        snr_estimate = 20 * np.log10(rms_energy + 1e-8)  # Rough SNR estimate
        
        # Normalize to 0.5-1.0 range
        energy_score = min(1.0, rms_energy * 10)
        snr_score = min(1.0, max(0.0, (snr_estimate + 40) / 40))  # Assume -40dB to 0dB range
        
        return 0.5 + 0.25 * energy_score + 0.25 * snr_score
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return 768  # Wav2Vec2 dimension
    
    def supports_batch(self) -> bool:
        """Supports batch processing."""
        return True


class StructuredDataEncoder(ModalityEncoder):
    """Encoder for structured data (JSON, tables, etc.)."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        
        # Simple neural network for structured data
        self.encoder = nn.Sequential(
            nn.Linear(100, 256),  # Assume max 100 features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
            nn.Tanh()
        )
    
    async def encode(self, structured_data: Dict[str, Any], **kwargs) -> ModalityEmbedding:
        """Encode structured data into embedding."""
        
        # Convert structured data to feature vector
        feature_vector = self._extract_features(structured_data)
        
        # Pad or truncate to fixed size
        if len(feature_vector) < 100:
            feature_vector = np.pad(feature_vector, (0, 100 - len(feature_vector)))
        else:
            feature_vector = feature_vector[:100]
        
        # Generate embedding
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
            embedding = self.encoder(input_tensor)
        
        embedding_np = embedding.numpy().flatten()
        
        return ModalityEmbedding(
            modality=ModalityType.STRUCTURED,
            embedding=embedding_np,
            confidence=0.9,  # High confidence for structured data
            metadata={
                "num_fields": len(structured_data),
                "data_types": self._get_data_types(structured_data),
                "feature_count": len(feature_vector)
            }
        )
    
    def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from structured data."""
        features = []
        
        def process_value(value):
            if isinstance(value, (int, float)):
                return [float(value)]
            elif isinstance(value, str):
                return [float(len(value)), float(hash(value) % 1000)]
            elif isinstance(value, bool):
                return [float(value)]
            elif isinstance(value, list):
                return [float(len(value))] + [process_value(v)[0] for v in value[:3]]  # First 3 items
            elif isinstance(value, dict):
                return [float(len(value))]
            else:
                return [0.0]
        
        for key, value in data.items():
            features.extend(process_value(value))
        
        return np.array(features[:100])  # Limit to 100 features
    
    def _get_data_types(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Get data types for metadata."""
        return {key: type(value).__name__ for key, value in data.items()}
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def supports_batch(self) -> bool:
        """Supports batch processing."""
        return True


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusion."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Forward pass with cross-modal attention."""
        
        # Cross-modal attention
        attn_output, _ = self.attention(query, key, value)
        query = self.norm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output


class MultiModalFusionEngine:
    """Engine for fusing multiple modality embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fusion_strategy = FusionStrategy(config.get("fusion_strategy", "attention_weighted"))
        self.target_dim = config.get("target_dimension", 512)
        
        # Initialize fusion components
        if self.fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
            self.cross_attention = CrossModalAttention(self.target_dim)
        elif self.fusion_strategy == FusionStrategy.TRANSFORMER_FUSION:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.target_dim,
                    nhead=8,
                    batch_first=True
                ),
                num_layers=2
            )
        
        # Projection layers for different modalities
        self.projections = nn.ModuleDict()
        
        logger.info(f"Initialized MultiModalFusionEngine with strategy: {self.fusion_strategy.value}")
    
    def _ensure_projection(self, modality: ModalityType, input_dim: int):
        """Ensure projection layer exists for modality."""
        key = modality.value
        if key not in self.projections:
            self.projections[key] = nn.Linear(input_dim, self.target_dim)
    
    async def fuse_embeddings(
        self, 
        embeddings: Dict[ModalityType, ModalityEmbedding]
    ) -> MultiModalEmbedding:
        """Fuse multiple modality embeddings."""
        
        if not embeddings:
            raise ValueError("No embeddings provided for fusion")
        
        # Project all embeddings to common dimension
        projected_embeddings = {}
        confidence_scores = {}
        
        for modality, emb in embeddings.items():
            self._ensure_projection(modality, len(emb.embedding))
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(emb.embedding).unsqueeze(0)
                projected = self.projections[modality.value](input_tensor)
                projected_embeddings[modality] = projected.squeeze(0).numpy()
                confidence_scores[modality] = emb.confidence
        
        # Apply fusion strategy
        if self.fusion_strategy == FusionStrategy.CONCATENATION:
            fused = await self._fuse_concatenation(projected_embeddings)
        elif self.fusion_strategy == FusionStrategy.ATTENTION_WEIGHTED:
            fused = await self._fuse_attention_weighted(projected_embeddings, confidence_scores)
        elif self.fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
            fused = await self._fuse_cross_modal_attention(projected_embeddings)
        elif self.fusion_strategy == FusionStrategy.TRANSFORMER_FUSION:
            fused = await self._fuse_transformer(projected_embeddings)
        else:
            # Default to attention-weighted
            fused = await self._fuse_attention_weighted(projected_embeddings, confidence_scores)
        
        # Calculate alignment scores
        alignment_scores = self._calculate_alignment_scores(projected_embeddings)
        
        return MultiModalEmbedding(
            modality_embeddings=embeddings,
            fused_embedding=fused,
            fusion_strategy=self.fusion_strategy,
            confidence_scores=confidence_scores,
            alignment_scores=alignment_scores,
            metadata={
                "fusion_timestamp": datetime.now(),
                "num_modalities": len(embeddings),
                "target_dimension": self.target_dim
            }
        )
    
    async def _fuse_concatenation(self, embeddings: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Simple concatenation fusion."""
        concatenated = np.concatenate(list(embeddings.values()))
        
        # Project to target dimension if needed
        if len(concatenated) != self.target_dim:
            # Simple linear projection
            projection_matrix = np.random.randn(len(concatenated), self.target_dim) * 0.1
            fused = concatenated @ projection_matrix
        else:
            fused = concatenated
        
        return fused / np.linalg.norm(fused)  # L2 normalize
    
    async def _fuse_attention_weighted(
        self, 
        embeddings: Dict[ModalityType, np.ndarray],
        confidence_scores: Dict[ModalityType, float]
    ) -> np.ndarray:
        """Attention-weighted fusion based on confidence scores."""
        
        # Normalize confidence scores to weights
        total_confidence = sum(confidence_scores.values())
        weights = {mod: conf / total_confidence for mod, conf in confidence_scores.items()}
        
        # Weighted average
        fused = np.zeros(self.target_dim)
        for modality, embedding in embeddings.items():
            fused += weights[modality] * embedding
        
        return fused / np.linalg.norm(fused)  # L2 normalize
    
    async def _fuse_cross_modal_attention(self, embeddings: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Cross-modal attention fusion."""
        
        # Stack embeddings
        embedding_stack = torch.FloatTensor(list(embeddings.values())).unsqueeze(0)  # (1, num_modalities, dim)
        
        with torch.no_grad():
            # Use first embedding as query, all as key/value
            query = embedding_stack[:, :1, :]  # First modality as query
            key_value = embedding_stack
            
            fused_tensor = self.cross_attention(query, key_value, key_value)
            fused = fused_tensor.squeeze(0).mean(dim=0).numpy()  # Average over modalities
        
        return fused / np.linalg.norm(fused)
    
    async def _fuse_transformer(self, embeddings: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Transformer-based fusion."""
        
        # Stack embeddings
        embedding_stack = torch.FloatTensor(list(embeddings.values())).unsqueeze(0)  # (1, num_modalities, dim)
        
        with torch.no_grad():
            fused_tensor = self.transformer(embedding_stack)
            fused = fused_tensor.squeeze(0).mean(dim=0).numpy()  # Average over modalities
        
        return fused / np.linalg.norm(fused)
    
    def _calculate_alignment_scores(self, embeddings: Dict[ModalityType, np.ndarray]) -> Dict[Tuple[ModalityType, ModalityType], float]:
        """Calculate pairwise alignment scores between modalities."""
        
        alignment_scores = {}
        modalities = list(embeddings.keys())
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                emb1 = embeddings[mod1]
                emb2 = embeddings[mod2]
                
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                alignment_scores[(mod1, mod2)] = float(similarity)
        
        return alignment_scores


class UniversalMultiModalEncoder:
    """
    Universal encoder that handles multiple modalities and produces
    unified embeddings for cross-modal similarity and retrieval.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize modality encoders
        self.encoders: Dict[ModalityType, ModalityEncoder] = {}
        
        if config.get("enable_text", True):
            self.encoders[ModalityType.TEXT] = TextEncoder(
                config.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
            )
        
        if config.get("enable_image", True) and MULTIMODAL_AVAILABLE:
            self.encoders[ModalityType.IMAGE] = ImageEncoder(
                config.get("image_model", "openai/clip-vit-base-patch32")
            )
        
        if config.get("enable_audio", True) and MULTIMODAL_AVAILABLE:
            self.encoders[ModalityType.AUDIO] = AudioEncoder(
                config.get("audio_model", "facebook/wav2vec2-base")
            )
        
        if config.get("enable_structured", True):
            self.encoders[ModalityType.STRUCTURED] = StructuredDataEncoder(
                config.get("structured_dim", 384)
            )
        
        # Initialize fusion engine
        self.fusion_engine = MultiModalFusionEngine(config.get("fusion", {}))
        
        # Cross-modal similarity cache
        self.similarity_cache = {}
        self.cache_size = config.get("cache_size", 1000)
        
        logger.info(f"Initialized UniversalMultiModalEncoder with {len(self.encoders)} modalities")
    
    async def encode_multimodal(
        self, 
        data: Dict[ModalityType, Any],
        return_individual: bool = False
    ) -> Union[MultiModalEmbedding, Dict[ModalityType, ModalityEmbedding]]:
        """Encode multi-modal data into unified embedding."""
        
        individual_embeddings = {}
        
        # Encode each modality
        for modality, content in data.items():
            if modality in self.encoders:
                try:
                    embedding = await self.encoders[modality].encode(content)
                    individual_embeddings[modality] = embedding
                    logger.debug(f"Encoded {modality.value} with confidence {embedding.confidence:.3f}")
                except Exception as e:
                    logger.warning(f"Failed to encode {modality.value}: {e}")
                    continue
        
        if return_individual:
            return individual_embeddings
        
        if not individual_embeddings:
            raise ValueError("No modalities could be encoded")
        
        # Fuse embeddings
        fused_embedding = await self.fusion_engine.fuse_embeddings(individual_embeddings)
        
        return fused_embedding
    
    async def compute_cross_modal_similarity(
        self,
        embedding1: MultiModalEmbedding,
        embedding2: MultiModalEmbedding,
        include_intra_modal: bool = True
    ) -> Dict[str, float]:
        """Compute comprehensive cross-modal similarity scores."""
        
        similarities = {}
        
        # Cross-modal similarities
        cross_modal_sim = embedding1.get_cross_modal_similarity(embedding2)
        similarities.update(cross_modal_sim)
        
        # Intra-modal similarities (same modality)
        if include_intra_modal:
            for mod1 in embedding1.modality_embeddings:
                if mod1 in embedding2.modality_embeddings:
                    emb1 = embedding1.modality_embeddings[mod1].embedding
                    emb2 = embedding2.modality_embeddings[mod1].embedding
                    
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities[f"{mod1.value}_to_{mod1.value}"] = float(similarity)
        
        # Fused embedding similarity
        fused_sim = np.dot(embedding1.fused_embedding, embedding2.fused_embedding) / (
            np.linalg.norm(embedding1.fused_embedding) * np.linalg.norm(embedding2.fused_embedding)
        )
        similarities["fused_similarity"] = float(fused_sim)
        
        return similarities
    
    async def find_cross_modal_matches(
        self,
        query_embedding: MultiModalEmbedding,
        candidate_embeddings: List[MultiModalEmbedding],
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Find cross-modal matches for a query embedding."""
        
        matches = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarities = await self.compute_cross_modal_similarity(query_embedding, candidate)
            
            # Calculate overall similarity score
            overall_score = similarities.get("fused_similarity", 0.0)
            
            # Add cross-modal boost
            cross_modal_scores = [
                score for key, score in similarities.items() 
                if "to" in key and key.split("_to_")[0] != key.split("_to_")[1]
            ]
            
            if cross_modal_scores:
                cross_modal_boost = max(cross_modal_scores) * 0.2
                overall_score += cross_modal_boost
            
            if overall_score >= similarity_threshold:
                matches.append((i, {
                    "overall_score": overall_score,
                    **similarities
                }))
        
        # Sort by overall score
        matches.sort(key=lambda x: x[1]["overall_score"], reverse=True)
        
        return matches[:top_k]
    
    async def get_modality_statistics(self) -> Dict[str, Any]:
        """Get statistics about modality usage and performance."""
        
        return {
            "available_modalities": [mod.value for mod in self.encoders.keys()],
            "encoder_dimensions": {
                mod.value: encoder.get_embedding_dimension() 
                for mod, encoder in self.encoders.items()
            },
            "fusion_strategy": self.fusion_engine.fusion_strategy.value,
            "target_dimension": self.fusion_engine.target_dim,
            "cache_size": len(self.similarity_cache),
            "multimodal_support": MULTIMODAL_AVAILABLE
        }
    
    def supports_modality(self, modality: ModalityType) -> bool:
        """Check if a modality is supported."""
        return modality in self.encoders
    
    async def encode_single_modality(self, modality: ModalityType, data: Any) -> ModalityEmbedding:
        """Encode a single modality."""
        if modality not in self.encoders:
            raise ValueError(f"Modality {modality.value} not supported")
        
        return await self.encoders[modality].encode(data)