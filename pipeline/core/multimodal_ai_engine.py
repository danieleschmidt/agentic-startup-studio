"""
Multimodal AI Engine - Generation 3 Enhancement
Advanced multimodal AI system with cross-domain intelligence
"""

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

from pydantic import BaseModel, Field
from opentelemetry import trace

from ..config.settings import get_settings
from ..telemetry import get_tracer

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Types of data modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    GRAPH = "graph"
    GEOSPATIAL = "geospatial"


class AITaskType(str, Enum):
    """Types of AI tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION = "recommendation"
    OPTIMIZATION = "optimization"


class ConfidenceLevel(str, Enum):
    """AI confidence levels"""
    VERY_LOW = "very_low"  # < 0.3
    LOW = "low"           # 0.3 - 0.5
    MEDIUM = "medium"     # 0.5 - 0.7
    HIGH = "high"         # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9


@dataclass
class MultimodalInput:
    """Input data with multiple modalities"""
    input_id: str
    modalities: Dict[ModalityType, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def add_modality(self, modality_type: ModalityType, data: Any, confidence: float = 1.0) -> None:
        """Add a modality to the input"""
        self.modalities[modality_type] = {
            "data": data,
            "confidence": confidence,
            "added_at": datetime.utcnow()
        }
    
    def get_modality_types(self) -> List[ModalityType]:
        """Get list of available modality types"""
        return list(self.modalities.keys())
    
    def is_multimodal(self) -> bool:
        """Check if input has multiple modalities"""
        return len(self.modalities) > 1


@dataclass
class AIResult:
    """Result from AI processing"""
    result_id: str
    task_type: AITaskType
    result_data: Dict[str, Any]
    confidence: float
    confidence_level: ConfidenceLevel
    processing_time_ms: float
    model_info: Dict[str, Any] = field(default_factory=dict)
    input_modalities: List[ModalityType] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Set confidence level based on numerical confidence"""
        if self.confidence < 0.3:
            self.confidence_level = ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.5:
            self.confidence_level = ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            self.confidence_level = ConfidenceLevel.HIGH
        else:
            self.confidence_level = ConfidenceLevel.VERY_HIGH


class AttentionMechanism:
    """Advanced attention mechanism for multimodal fusion"""
    
    def __init__(self, hidden_dim: int = 512):
        self.hidden_dim = hidden_dim
        self.attention_weights: Dict[str, np.ndarray] = {}
        self.learned_patterns: List[Dict[str, Any]] = []
        
    def compute_attention(
        self, 
        modality_features: Dict[ModalityType, np.ndarray],
        query_context: Optional[str] = None
    ) -> Dict[ModalityType, float]:
        """Compute attention weights for each modality"""
        
        attention_scores = {}
        
        for modality, features in modality_features.items():
            # Simulate attention computation
            base_attention = self._compute_base_attention(features)
            
            # Context-aware attention adjustment
            context_adjustment = self._compute_context_attention(modality, query_context)
            
            # Cross-modal attention
            cross_modal_attention = self._compute_cross_modal_attention(
                modality, features, modality_features
            )
            
            # Combine attention scores
            final_attention = (
                0.5 * base_attention +
                0.3 * context_adjustment +
                0.2 * cross_modal_attention
            )
            
            attention_scores[modality] = final_attention
        
        # Normalize attention weights
        total_attention = sum(attention_scores.values())
        if total_attention > 0:
            attention_scores = {
                modality: score / total_attention
                for modality, score in attention_scores.items()
            }
        
        return attention_scores
    
    def _compute_base_attention(self, features: np.ndarray) -> float:
        """Compute base attention based on feature characteristics"""
        if len(features) == 0:
            return 0.0
        
        # Attention based on feature variance and magnitude
        variance = np.var(features)
        magnitude = np.mean(np.abs(features))
        
        return float(np.tanh(variance + magnitude))
    
    def _compute_context_attention(self, modality: ModalityType, context: Optional[str]) -> float:
        """Compute context-aware attention weights"""
        if context is None:
            return 0.5
        
        # Context relevance scoring
        context_relevance = {
            ModalityType.TEXT: ["text", "language", "nlp", "content"],
            ModalityType.IMAGE: ["visual", "image", "picture", "graphics"],
            ModalityType.AUDIO: ["sound", "audio", "speech", "music"],
            ModalityType.NUMERIC: ["number", "data", "metric", "value"],
            ModalityType.TEMPORAL: ["time", "sequence", "temporal", "trend"]
        }
        
        context_lower = context.lower()
        relevance_keywords = context_relevance.get(modality, [])
        
        relevance_score = sum(1 for keyword in relevance_keywords if keyword in context_lower)
        return min(1.0, relevance_score / len(relevance_keywords) if relevance_keywords else 0.5)
    
    def _compute_cross_modal_attention(
        self, 
        target_modality: ModalityType,
        target_features: np.ndarray,
        all_features: Dict[ModalityType, np.ndarray]
    ) -> float:
        """Compute cross-modal attention based on feature correlations"""
        if len(all_features) <= 1:
            return 0.5
        
        correlations = []
        for modality, features in all_features.items():
            if modality != target_modality and len(features) > 0 and len(target_features) > 0:
                # Simplified correlation computation
                min_len = min(len(target_features), len(features))
                if min_len > 0:
                    corr = np.corrcoef(
                        target_features[:min_len], 
                        features[:min_len]
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return float(np.mean(correlations)) if correlations else 0.5


class MultimodalFusion:
    """Advanced multimodal fusion strategies"""
    
    def __init__(self):
        self.attention_mechanism = AttentionMechanism()
        self.fusion_strategies = {
            "early": self._early_fusion,
            "late": self._late_fusion,
            "attention": self._attention_fusion,
            "hierarchical": self._hierarchical_fusion
        }
        
    async def fuse_modalities(
        self, 
        modality_features: Dict[ModalityType, np.ndarray],
        strategy: str = "attention",
        context: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fuse multiple modalities into unified representation"""
        
        if strategy not in self.fusion_strategies:
            strategy = "attention"
        
        fusion_func = self.fusion_strategies[strategy]
        fused_features, fusion_info = await fusion_func(modality_features, context)
        
        fusion_info.update({
            "strategy": strategy,
            "input_modalities": list(modality_features.keys()),
            "fusion_timestamp": datetime.utcnow().isoformat()
        })
        
        return fused_features, fusion_info
    
    async def _early_fusion(
        self, 
        modality_features: Dict[ModalityType, np.ndarray],
        context: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Early fusion: concatenate features before processing"""
        
        all_features = []
        feature_boundaries = {}
        current_pos = 0
        
        for modality, features in modality_features.items():
            all_features.extend(features)
            feature_boundaries[modality.value] = {
                "start": current_pos,
                "end": current_pos + len(features)
            }
            current_pos += len(features)
        
        fused_features = np.array(all_features)
        
        fusion_info = {
            "fusion_type": "early",
            "total_features": len(fused_features),
            "feature_boundaries": feature_boundaries
        }
        
        return fused_features, fusion_info
    
    async def _late_fusion(
        self, 
        modality_features: Dict[ModalityType, np.ndarray],
        context: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Late fusion: process each modality separately then combine"""
        
        processed_features = {}
        
        for modality, features in modality_features.items():
            # Simulate modality-specific processing
            processed = await self._process_modality_features(modality, features)
            processed_features[modality] = processed
        
        # Combine processed features
        combined_features = []
        weights = {}
        
        for modality, processed in processed_features.items():
            weight = 1.0 / len(processed_features)  # Equal weights for now
            weighted_features = processed * weight
            combined_features.extend(weighted_features)
            weights[modality.value] = weight
        
        fused_features = np.array(combined_features)
        
        fusion_info = {
            "fusion_type": "late",
            "modality_weights": weights,
            "processed_features_count": {
                modality.value: len(features) 
                for modality, features in processed_features.items()
            }
        }
        
        return fused_features, fusion_info
    
    async def _attention_fusion(
        self, 
        modality_features: Dict[ModalityType, np.ndarray],
        context: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Attention-based fusion using learned attention weights"""
        
        # Compute attention weights
        attention_weights = self.attention_mechanism.compute_attention(
            modality_features, context
        )
        
        # Apply attention weights to features
        weighted_features = []
        attention_info = {}
        
        for modality, features in modality_features.items():
            weight = attention_weights.get(modality, 0.0)
            weighted = features * weight
            weighted_features.extend(weighted)
            attention_info[modality.value] = {
                "weight": weight,
                "feature_count": len(features)
            }
        
        fused_features = np.array(weighted_features)
        
        fusion_info = {
            "fusion_type": "attention",
            "attention_weights": attention_info,
            "total_attention": sum(attention_weights.values())
        }
        
        return fused_features, fusion_info
    
    async def _hierarchical_fusion(
        self, 
        modality_features: Dict[ModalityType, np.ndarray],
        context: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Hierarchical fusion with multiple levels"""
        
        # Level 1: Group similar modalities
        modality_groups = self._group_similar_modalities(modality_features)
        
        # Level 2: Fuse within groups
        group_features = {}
        for group_name, group_modalities in modality_groups.items():
            group_data = {mod: modality_features[mod] for mod in group_modalities}
            group_fused, _ = await self._attention_fusion(group_data, context)
            group_features[group_name] = group_fused
        
        # Level 3: Fuse across groups
        final_features = []
        group_info = {}
        
        for group_name, features in group_features.items():
            final_features.extend(features)
            group_info[group_name] = len(features)
        
        fused_features = np.array(final_features)
        
        fusion_info = {
            "fusion_type": "hierarchical",
            "modality_groups": {
                group: [mod.value for mod in mods] 
                for group, mods in modality_groups.items()
            },
            "group_feature_counts": group_info
        }
        
        return fused_features, fusion_info
    
    def _group_similar_modalities(
        self, 
        modality_features: Dict[ModalityType, np.ndarray]
    ) -> Dict[str, List[ModalityType]]:
        """Group similar modalities for hierarchical fusion"""
        
        groups = {
            "visual": [ModalityType.IMAGE, ModalityType.VIDEO],
            "sequential": [ModalityType.TEXT, ModalityType.AUDIO, ModalityType.TEMPORAL],
            "structured": [ModalityType.NUMERIC, ModalityType.GRAPH, ModalityType.GEOSPATIAL]
        }
        
        # Filter groups to only include available modalities
        filtered_groups = {}
        for group_name, modalities in groups.items():
            available_modalities = [mod for mod in modalities if mod in modality_features]
            if available_modalities:
                filtered_groups[group_name] = available_modalities
        
        # Handle ungrouped modalities
        grouped_modalities = set()
        for modalities in filtered_groups.values():
            grouped_modalities.update(modalities)
        
        ungrouped = [mod for mod in modality_features.keys() if mod not in grouped_modalities]
        if ungrouped:
            filtered_groups["other"] = ungrouped
        
        return filtered_groups
    
    async def _process_modality_features(
        self, 
        modality: ModalityType, 
        features: np.ndarray
    ) -> np.ndarray:
        """Process features specific to modality type"""
        
        if modality == ModalityType.TEXT:
            # Text-specific processing (embedding, normalization, etc.)
            return self._process_text_features(features)
        elif modality == ModalityType.IMAGE:
            # Image-specific processing (CNN features, etc.)
            return self._process_image_features(features)
        elif modality == ModalityType.AUDIO:
            # Audio-specific processing (spectral features, etc.)
            return self._process_audio_features(features)
        elif modality == ModalityType.NUMERIC:
            # Numeric processing (normalization, scaling, etc.)
            return self._process_numeric_features(features)
        else:
            # Default processing
            return self._normalize_features(features)
    
    def _process_text_features(self, features: np.ndarray) -> np.ndarray:
        """Process text features"""
        # Simulate text feature processing
        normalized = self._normalize_features(features)
        # Apply text-specific transformations
        return normalized * 1.1  # Slight amplification for text
    
    def _process_image_features(self, features: np.ndarray) -> np.ndarray:
        """Process image features"""
        # Simulate image feature processing
        normalized = self._normalize_features(features)
        # Apply image-specific transformations (e.g., spatial attention)
        return normalized * 0.95  # Slight dampening for images
    
    def _process_audio_features(self, features: np.ndarray) -> np.ndarray:
        """Process audio features"""
        # Simulate audio feature processing
        normalized = self._normalize_features(features)
        # Apply audio-specific transformations (e.g., temporal smoothing)
        return np.convolve(normalized, np.ones(3)/3, mode='same')  # Simple smoothing
    
    def _process_numeric_features(self, features: np.ndarray) -> np.ndarray:
        """Process numeric features"""
        # Standard numeric preprocessing
        return self._normalize_features(features)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to unit scale"""
        if len(features) == 0:
            return features
        
        features_array = np.array(features)
        
        # Min-max normalization
        min_val = np.min(features_array)
        max_val = np.max(features_array)
        
        if max_val > min_val:
            normalized = (features_array - min_val) / (max_val - min_val)
        else:
            normalized = features_array
        
        return normalized


class MultimodalEmbedding:
    """Create unified embeddings from multimodal data"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.modality_projectors: Dict[ModalityType, np.ndarray] = {}
        self._initialize_projectors()
        
    def _initialize_projectors(self) -> None:
        """Initialize projection matrices for each modality"""
        for modality in ModalityType:
            # Random projection matrix (in real system would be learned)
            self.modality_projectors[modality] = np.random.normal(
                0, 0.1, (self.embedding_dim, self.embedding_dim)
            )
    
    async def create_embedding(
        self, 
        multimodal_input: MultimodalInput,
        fusion_strategy: str = "attention"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create unified embedding from multimodal input"""
        
        # Extract features from each modality
        modality_features = {}
        extraction_info = {}
        
        for modality_type, modality_data in multimodal_input.modalities.items():
            features = await self._extract_features(modality_type, modality_data["data"])
            modality_features[modality_type] = features
            extraction_info[modality_type.value] = {
                "feature_count": len(features),
                "confidence": modality_data.get("confidence", 1.0)
            }
        
        # Fuse modalities
        fusion_engine = MultimodalFusion()
        fused_features, fusion_info = await fusion_engine.fuse_modalities(
            modality_features, fusion_strategy
        )
        
        # Project to embedding space
        embedding = await self._project_to_embedding_space(fused_features)
        
        embedding_info = {
            "input_id": multimodal_input.input_id,
            "embedding_dim": self.embedding_dim,
            "modality_extraction": extraction_info,
            "fusion_info": fusion_info,
            "embedding_norm": float(np.linalg.norm(embedding))
        }
        
        return embedding, embedding_info
    
    async def _extract_features(self, modality_type: ModalityType, data: Any) -> np.ndarray:
        """Extract features from specific modality data"""
        
        if modality_type == ModalityType.TEXT:
            return await self._extract_text_features(data)
        elif modality_type == ModalityType.IMAGE:
            return await self._extract_image_features(data)
        elif modality_type == ModalityType.AUDIO:
            return await self._extract_audio_features(data)
        elif modality_type == ModalityType.NUMERIC:
            return await self._extract_numeric_features(data)
        elif modality_type == ModalityType.TEMPORAL:
            return await self._extract_temporal_features(data)
        else:
            # Default feature extraction
            return await self._extract_default_features(data)
    
    async def _extract_text_features(self, text_data: Any) -> np.ndarray:
        """Extract features from text data"""
        if isinstance(text_data, str):
            # Simulate text feature extraction (TF-IDF, embeddings, etc.)
            text_length = len(text_data)
            word_count = len(text_data.split())
            
            # Simple text features
            features = [
                text_length / 1000,  # Normalized length
                word_count / 100,    # Normalized word count
                text_data.count('.') / max(1, word_count),  # Sentence density
                len(set(text_data.split())) / max(1, word_count),  # Vocabulary richness
            ]
            
            # Pad to fixed size
            features.extend([0.0] * (64 - len(features)))
            return np.array(features[:64])
        else:
            return np.zeros(64)
    
    async def _extract_image_features(self, image_data: Any) -> np.ndarray:
        """Extract features from image data"""
        # Simulate image feature extraction (CNN features, color histograms, etc.)
        if isinstance(image_data, dict) and "dimensions" in image_data:
            width = image_data.get("width", 100)
            height = image_data.get("height", 100)
            channels = image_data.get("channels", 3)
            
            features = [
                width / 1000,   # Normalized width
                height / 1000,  # Normalized height
                channels / 4,   # Normalized channels
                (width * height) / 1000000,  # Normalized area
            ]
            
            # Simulate additional CNN-like features
            features.extend(np.random.normal(0, 0.1, 60).tolist())
            return np.array(features[:64])
        else:
            # Default image features
            return np.random.normal(0, 0.1, 64)
    
    async def _extract_audio_features(self, audio_data: Any) -> np.ndarray:
        """Extract features from audio data"""
        # Simulate audio feature extraction (MFCC, spectral features, etc.)
        if isinstance(audio_data, dict):
            duration = audio_data.get("duration", 1.0)
            sample_rate = audio_data.get("sample_rate", 44100)
            
            features = [
                duration / 10,  # Normalized duration
                sample_rate / 44100,  # Normalized sample rate
            ]
            
            # Simulate MFCC-like features
            features.extend(np.random.normal(0, 0.1, 62).tolist())
            return np.array(features[:64])
        else:
            return np.random.normal(0, 0.1, 64)
    
    async def _extract_numeric_features(self, numeric_data: Any) -> np.ndarray:
        """Extract features from numeric data"""
        if isinstance(numeric_data, (list, np.ndarray)):
            data_array = np.array(numeric_data)
            
            # Statistical features
            features = [
                np.mean(data_array),
                np.std(data_array),
                np.min(data_array),
                np.max(data_array),
                np.median(data_array),
                len(data_array) / 1000,  # Normalized length
            ]
            
            # Pad or truncate to fixed size
            features.extend([0.0] * (64 - len(features)))
            return np.array(features[:64])
        else:
            return np.zeros(64)
    
    async def _extract_temporal_features(self, temporal_data: Any) -> np.ndarray:
        """Extract features from temporal data"""
        if isinstance(temporal_data, list):
            # Time series features
            series = np.array(temporal_data)
            
            features = [
                len(series) / 1000,  # Normalized length
                np.mean(series),
                np.std(series),
                np.mean(np.diff(series)) if len(series) > 1 else 0,  # Trend
            ]
            
            # Add autocorrelation-like features
            if len(series) > 1:
                for lag in [1, 5, 10]:
                    if lag < len(series):
                        correlation = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                        features.append(correlation if not np.isnan(correlation) else 0)
                    else:
                        features.append(0)
            
            # Pad to fixed size
            features.extend([0.0] * (64 - len(features)))
            return np.array(features[:64])
        else:
            return np.zeros(64)
    
    async def _extract_default_features(self, data: Any) -> np.ndarray:
        """Default feature extraction for unknown data types"""
        # Convert to string and extract basic features
        str_data = str(data)
        features = [
            len(str_data) / 1000,  # Normalized length
            str_data.count(' ') / max(1, len(str_data)),  # Space density
        ]
        
        features.extend([0.0] * (64 - len(features)))
        return np.array(features[:64])
    
    async def _project_to_embedding_space(self, features: np.ndarray) -> np.ndarray:
        """Project features to unified embedding space"""
        # Ensure features are right size for projection
        if len(features) < self.embedding_dim:
            # Pad with zeros
            padded_features = np.zeros(self.embedding_dim)
            padded_features[:len(features)] = features
            features = padded_features
        elif len(features) > self.embedding_dim:
            # Truncate
            features = features[:self.embedding_dim]
        
        # Apply projection (simplified)
        embedding = np.tanh(features)  # Non-linear activation
        
        # Normalize to unit sphere
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class MultimodalAI:
    """Main multimodal AI processing engine"""
    
    def __init__(self):
        self.embedding_engine = MultimodalEmbedding()
        self.task_processors: Dict[AITaskType, callable] = {}
        self.model_cache: Dict[str, Any] = {}
        self.processing_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        self._initialize_task_processors()
        
    def _initialize_task_processors(self) -> None:
        """Initialize processors for different AI tasks"""
        self.task_processors = {
            AITaskType.CLASSIFICATION: self._process_classification,
            AITaskType.REGRESSION: self._process_regression,
            AITaskType.GENERATION: self._process_generation,
            AITaskType.SUMMARIZATION: self._process_summarization,
            AITaskType.QUESTION_ANSWERING: self._process_question_answering,
            AITaskType.ANOMALY_DETECTION: self._process_anomaly_detection,
            AITaskType.RECOMMENDATION: self._process_recommendation,
            AITaskType.OPTIMIZATION: self._process_optimization
        }
    
    async def process_multimodal_task(
        self,
        task_type: AITaskType,
        multimodal_input: MultimodalInput,
        task_parameters: Dict[str, Any] = None
    ) -> AIResult:
        """Process a multimodal AI task"""
        
        start_time = time.time()
        task_parameters = task_parameters or {}
        
        with tracer.start_as_current_span("process_multimodal_task") as span:
            span.set_attributes({
                "task_type": task_type.value,
                "input_id": multimodal_input.input_id,
                "modality_count": len(multimodal_input.modalities)
            })
            
            try:
                # Create unified embedding
                embedding, embedding_info = await self.embedding_engine.create_embedding(
                    multimodal_input
                )
                
                # Process task with appropriate processor
                processor = self.task_processors.get(task_type)
                if not processor:
                    raise ValueError(f"No processor available for task type: {task_type}")
                
                result_data = await processor(embedding, multimodal_input, task_parameters)
                
                # Calculate confidence
                confidence = self._calculate_confidence(result_data, embedding_info)
                
                processing_time = (time.time() - start_time) * 1000
                
                # Create result
                result = AIResult(
                    result_id=f"{task_type.value}_{int(time.time() * 1000)}",
                    task_type=task_type,
                    result_data=result_data,
                    confidence=confidence,
                    confidence_level=ConfidenceLevel.MEDIUM,  # Will be set in __post_init__
                    processing_time_ms=processing_time,
                    model_info={
                        "embedding_info": embedding_info,
                        "task_parameters": task_parameters
                    },
                    input_modalities=multimodal_input.get_modality_types()
                )
                
                # Record processing history
                self.processing_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "task_type": task_type.value,
                    "input_modalities": [m.value for m in multimodal_input.get_modality_types()],
                    "processing_time_ms": processing_time,
                    "confidence": confidence,
                    "success": True
                })
                
                # Update performance metrics
                self._update_performance_metrics(task_type, processing_time, confidence)
                
                return result
                
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                
                # Record error
                self.processing_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "task_type": task_type.value,
                    "processing_time_ms": processing_time,
                    "error": str(e),
                    "success": False
                })
                
                logger.error(f"Error processing multimodal task {task_type.value}: {e}")
                raise
    
    async def _process_classification(
        self,
        embedding: np.ndarray,
        multimodal_input: MultimodalInput,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process classification task"""
        
        num_classes = parameters.get("num_classes", 5)
        class_names = parameters.get("class_names", [f"class_{i}" for i in range(num_classes)])
        
        # Simulate classification using embedding
        logits = np.random.normal(0, 1, num_classes)
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        
        return {
            "predicted_class": predicted_class,
            "predicted_class_idx": int(predicted_class_idx),
            "probabilities": {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            "top_3_predictions": [
                {
                    "class": class_names[idx],
                    "probability": float(probabilities[idx])
                }
                for idx in np.argsort(probabilities)[-3:][::-1]
            ]
        }
    
    async def _process_regression(
        self,
        embedding: np.ndarray,
        multimodal_input: MultimodalInput,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process regression task"""
        
        target_range = parameters.get("target_range", [0, 1])
        target_name = parameters.get("target_name", "value")
        
        # Simulate regression using embedding
        raw_prediction = np.mean(embedding) + np.random.normal(0, 0.1)
        
        # Scale to target range
        min_val, max_val = target_range
        predicted_value = min_val + (raw_prediction + 1) / 2 * (max_val - min_val)
        predicted_value = np.clip(predicted_value, min_val, max_val)
        
        # Estimate uncertainty
        uncertainty = abs(np.std(embedding)) * 0.1
        
        return {
            target_name: float(predicted_value),
            "uncertainty": float(uncertainty),
            "confidence_interval": {
                "lower": float(predicted_value - uncertainty),
                "upper": float(predicted_value + uncertainty)
            },
            "target_range": target_range
        }
    
    async def _process_generation(
        self,
        embedding: np.ndarray,
        multimodal_input: MultimodalInput,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process generation task"""
        
        generation_type = parameters.get("type", "text")
        max_length = parameters.get("max_length", 100)
        
        if generation_type == "text":
            # Simulate text generation based on embedding
            base_words = ["innovative", "efficient", "advanced", "intelligent", "automated"]
            embedding_influenced_words = []
            
            for i in range(min(max_length // 10, len(embedding))):
                if embedding[i] > 0:
                    embedding_influenced_words.append(base_words[i % len(base_words)])
            
            generated_text = f"Based on multimodal analysis: {' '.join(embedding_influenced_words)}"
            
            return {
                "generated_text": generated_text,
                "generation_type": generation_type,
                "length": len(generated_text),
                "diversity_score": len(set(generated_text.split())) / len(generated_text.split())
            }
        else:
            return {
                "generated_content": "Generated content placeholder",
                "generation_type": generation_type,
                "metadata": {"note": "Generation type not fully implemented"}
            }
    
    async def _process_summarization(
        self,
        embedding: np.ndarray,
        multimodal_input: MultimodalInput,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process summarization task"""
        
        max_summary_length = parameters.get("max_length", 50)
        
        # Analyze input modalities for summarization
        modality_insights = []
        
        for modality_type in multimodal_input.get_modality_types():
            if modality_type == ModalityType.TEXT:
                modality_insights.append("textual content analysis")
            elif modality_type == ModalityType.IMAGE:
                modality_insights.append("visual content analysis")
            elif modality_type == ModalityType.NUMERIC:
                modality_insights.append("numerical data analysis")
            else:
                modality_insights.append(f"{modality_type.value} analysis")
        
        # Generate summary
        summary = f"Multimodal analysis incorporating {', '.join(modality_insights)}."
        if len(summary) > max_summary_length:
            summary = summary[:max_summary_length-3] + "..."
        
        # Extract key points
        key_points = [
            f"Contains {len(multimodal_input.modalities)} modalities",
            f"Primary focus: {modality_insights[0] if modality_insights else 'general analysis'}",
            f"Confidence: {np.mean(embedding):.2f}"
        ]
        
        return {
            "summary": summary,
            "key_points": key_points,
            "summary_length": len(summary),
            "compression_ratio": len(summary) / 1000,  # Assuming 1000 char input
            "modalities_analyzed": [m.value for m in multimodal_input.get_modality_types()]
        }
    
    async def _process_question_answering(
        self,
        embedding: np.ndarray,
        multimodal_input: MultimodalInput,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process question answering task"""
        
        question = parameters.get("question", "What can you tell me about this content?")
        answer_type = parameters.get("answer_type", "extractive")
        
        # Analyze question and generate answer
        question_embedding = await self._get_question_embedding(question)
        
        # Calculate relevance between question and content
        relevance = np.dot(embedding, question_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(question_embedding)
        )
        
        # Generate answer based on modalities
        answer_components = []
        
        for modality_type in multimodal_input.get_modality_types():
            if modality_type == ModalityType.TEXT:
                answer_components.append("Based on textual analysis")
            elif modality_type == ModalityType.IMAGE:
                answer_components.append("visual inspection reveals")
            elif modality_type == ModalityType.NUMERIC:
                answer_components.append("numerical data indicates")
        
        answer = f"{', '.join(answer_components)} relevant information for your question."
        
        return {
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "relevance_score": float(relevance),
            "confidence": min(1.0, abs(relevance) + 0.5),
            "supporting_modalities": [m.value for m in multimodal_input.get_modality_types()]
        }
    
    async def _process_anomaly_detection(
        self,
        embedding: np.ndarray,
        multimodal_input: MultimodalInput,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process anomaly detection task"""
        
        threshold = parameters.get("threshold", 0.8)
        
        # Calculate anomaly score based on embedding characteristics
        embedding_norm = np.linalg.norm(embedding)
        embedding_mean = np.mean(embedding)
        embedding_std = np.std(embedding)
        
        # Anomaly indicators
        norm_anomaly = abs(embedding_norm - 1.0)  # Embeddings should be normalized
        distribution_anomaly = abs(embedding_mean)  # Should be centered around 0
        variance_anomaly = abs(embedding_std - 0.5)  # Expected standard deviation
        
        # Combined anomaly score
        anomaly_score = (norm_anomaly + distribution_anomaly + variance_anomaly) / 3
        
        is_anomaly = anomaly_score > threshold
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(anomaly_score),
            "threshold": threshold,
            "anomaly_indicators": {
                "norm_deviation": float(norm_anomaly),
                "distribution_deviation": float(distribution_anomaly),
                "variance_deviation": float(variance_anomaly)
            },
            "severity": "high" if anomaly_score > 0.9 else "medium" if anomaly_score > 0.6 else "low"
        }
    
    async def _process_recommendation(
        self,
        embedding: np.ndarray,
        multimodal_input: MultimodalInput,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process recommendation task"""
        
        num_recommendations = parameters.get("num_recommendations", 5)
        recommendation_type = parameters.get("type", "content")
        
        # Generate recommendations based on embedding similarity
        candidate_items = parameters.get("candidates", [
            f"Item_{i}" for i in range(num_recommendations * 2)
        ])
        
        # Simulate item embeddings and calculate similarity
        recommendations = []
        
        for item in candidate_items[:num_recommendations * 2]:
            # Simulate item embedding
            item_embedding = np.random.normal(0, 0.1, len(embedding))
            
            # Calculate similarity
            similarity = np.dot(embedding, item_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(item_embedding)
            )
            
            recommendations.append({
                "item": item,
                "similarity_score": float(similarity),
                "confidence": min(1.0, abs(similarity) + 0.3)
            })
        
        # Sort by similarity and take top recommendations
        recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_recommendations = recommendations[:num_recommendations]
        
        return {
            "recommendations": top_recommendations,
            "recommendation_type": recommendation_type,
            "num_recommendations": len(top_recommendations),
            "average_confidence": np.mean([r["confidence"] for r in top_recommendations]),
            "input_modalities": [m.value for m in multimodal_input.get_modality_types()]
        }
    
    async def _process_optimization(
        self,
        embedding: np.ndarray,
        multimodal_input: MultimodalInput,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process optimization task"""
        
        optimization_target = parameters.get("target", "performance")
        constraints = parameters.get("constraints", {})
        
        # Use embedding to guide optimization
        current_values = embedding[:min(len(embedding), 10)]  # Use first 10 dimensions
        
        # Simulate optimization process
        optimized_values = []
        improvements = []
        
        for i, value in enumerate(current_values):
            # Apply optimization (gradient-like improvement)
            improvement_factor = 1 + (value * 0.1)  # Small improvement based on current value
            optimized_value = value * improvement_factor
            
            # Apply constraints
            if f"param_{i}_min" in constraints:
                optimized_value = max(optimized_value, constraints[f"param_{i}_min"])
            if f"param_{i}_max" in constraints:
                optimized_value = min(optimized_value, constraints[f"param_{i}_max"])
            
            optimized_values.append(float(optimized_value))
            improvements.append(float(abs(optimized_value - value)))
        
        total_improvement = sum(improvements)
        
        return {
            "optimization_target": optimization_target,
            "original_values": [float(v) for v in current_values],
            "optimized_values": optimized_values,
            "improvements": improvements,
            "total_improvement": total_improvement,
            "optimization_success": total_improvement > 0.1,
            "constraints_applied": list(constraints.keys())
        }
    
    async def _get_question_embedding(self, question: str) -> np.ndarray:
        """Get embedding for a question"""
        # Simulate question embedding (in real system would use proper NLP)
        question_length = len(question)
        word_count = len(question.split())
        
        # Simple question features
        features = np.array([
            question_length / 100,
            word_count / 20,
            question.count('?') / max(1, word_count),
            len(set(question.split())) / max(1, word_count)
        ])
        
        # Pad to match embedding dimension
        if len(features) < 768:
            padded = np.zeros(768)
            padded[:len(features)] = features
            features = padded
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _calculate_confidence(
        self, 
        result_data: Dict[str, Any], 
        embedding_info: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for AI result"""
        
        base_confidence = 0.7  # Base confidence
        
        # Adjust based on embedding quality
        embedding_norm = embedding_info.get("embedding_norm", 1.0)
        norm_factor = min(1.0, embedding_norm)
        
        # Adjust based on modality count (more modalities = higher confidence)
        modality_count = len(embedding_info.get("modality_extraction", {}))
        modality_factor = min(1.0, 0.5 + (modality_count * 0.1))
        
        # Adjust based on fusion quality
        fusion_info = embedding_info.get("fusion_info", {})
        total_attention = fusion_info.get("total_attention", 0.5)
        attention_factor = min(1.0, total_attention)
        
        # Combine factors
        confidence = base_confidence * norm_factor * modality_factor * attention_factor
        
        return max(0.1, min(0.95, confidence))
    
    def _update_performance_metrics(
        self, 
        task_type: AITaskType, 
        processing_time: float, 
        confidence: float
    ) -> None:
        """Update performance metrics"""
        
        metric_key = f"{task_type.value}_avg_time"
        if metric_key in self.performance_metrics:
            # Exponential moving average
            self.performance_metrics[metric_key] = (
                0.9 * self.performance_metrics[metric_key] + 0.1 * processing_time
            )
        else:
            self.performance_metrics[metric_key] = processing_time
        
        confidence_key = f"{task_type.value}_avg_confidence"
        if confidence_key in self.performance_metrics:
            self.performance_metrics[confidence_key] = (
                0.9 * self.performance_metrics[confidence_key] + 0.1 * confidence
            )
        else:
            self.performance_metrics[confidence_key] = confidence
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get comprehensive AI engine status"""
        
        # Processing statistics
        total_tasks = len(self.processing_history)
        successful_tasks = sum(1 for task in self.processing_history if task.get("success", False))
        
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Task type distribution
        task_distribution = {}
        for task in self.processing_history:
            task_type = task.get("task_type", "unknown")
            task_distribution[task_type] = task_distribution.get(task_type, 0) + 1
        
        # Recent performance
        recent_tasks = [
            task for task in self.processing_history[-50:]  # Last 50 tasks
            if task.get("success", False)
        ]
        
        if recent_tasks:
            avg_processing_time = np.mean([task["processing_time_ms"] for task in recent_tasks])
            avg_confidence = np.mean([task.get("confidence", 0.5) for task in recent_tasks])
        else:
            avg_processing_time = 0
            avg_confidence = 0
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tasks_processed": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate_percent": success_rate,
            "task_distribution": task_distribution,
            "recent_performance": {
                "avg_processing_time_ms": avg_processing_time,
                "avg_confidence": avg_confidence,
                "recent_task_count": len(recent_tasks)
            },
            "available_task_types": [task_type.value for task_type in AITaskType],
            "supported_modalities": [modality.value for modality in ModalityType],
            "performance_metrics": dict(self.performance_metrics),
            "model_cache_size": len(self.model_cache)
        }


# Global AI engine instance
_multimodal_ai: Optional[MultimodalAI] = None


async def get_multimodal_ai() -> MultimodalAI:
    """Get or create the global multimodal AI engine"""
    global _multimodal_ai
    if _multimodal_ai is None:
        _multimodal_ai = MultimodalAI()
    return _multimodal_ai


async def process_text_and_image(
    text: str,
    image_data: Dict[str, Any],
    task_type: AITaskType = AITaskType.CLASSIFICATION
) -> AIResult:
    """Convenience function to process text and image together"""
    
    ai_engine = await get_multimodal_ai()
    
    multimodal_input = MultimodalInput(
        input_id=f"text_image_{int(time.time() * 1000)}"
    )
    
    multimodal_input.add_modality(ModalityType.TEXT, text)
    multimodal_input.add_modality(ModalityType.IMAGE, image_data)
    
    return await ai_engine.process_multimodal_task(task_type, multimodal_input)


async def analyze_startup_idea_multimodal(
    idea_text: str,
    market_data: Dict[str, Any],
    additional_context: Dict[str, Any] = None
) -> AIResult:
    """Analyze startup idea using multimodal approach"""
    
    ai_engine = await get_multimodal_ai()
    
    multimodal_input = MultimodalInput(
        input_id=f"startup_analysis_{int(time.time() * 1000)}"
    )
    
    # Add text modality
    multimodal_input.add_modality(ModalityType.TEXT, idea_text)
    
    # Add numeric market data
    if market_data:
        multimodal_input.add_modality(ModalityType.NUMERIC, market_data)
    
    # Add additional context as structured data
    if additional_context:
        multimodal_input.add_modality(ModalityType.GRAPH, additional_context)
    
    # Classify the startup idea
    return await ai_engine.process_multimodal_task(
        AITaskType.CLASSIFICATION,
        multimodal_input,
        {
            "num_classes": 3,
            "class_names": ["high_potential", "medium_potential", "low_potential"]
        }
    )