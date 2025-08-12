#!/usr/bin/env python3
"""
Adaptive Learning System - Self-Improving AI Core
=================================================

An advanced adaptive learning system that continuously improves performance
through experience, pattern recognition, and quantum-inspired optimization.

Features:
- Continuous learning from execution data
- Pattern recognition and classification
- Performance prediction and optimization
- Self-adaptive algorithms
- Memory consolidation and forgetting
"""

import asyncio
import json
import logging
import math
import pickle
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import uuid

logger = logging.getLogger(__name__)


class LearningType(str, Enum):
    """Types of learning the system can perform"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    QUANTUM_INSPIRED = "quantum_inspired"


class PatternCategory(str, Enum):
    """Categories of patterns the system can learn"""
    PERFORMANCE = "performance"
    ERROR = "error"
    USER_BEHAVIOR = "user_behavior"
    RESOURCE_USAGE = "resource_usage"
    WORKFLOW = "workflow"
    OPTIMIZATION = "optimization"


class LearningMode(str, Enum):
    """Learning modes for different scenarios"""
    EXPLORATION = "exploration"     # Focus on discovering new patterns
    EXPLOITATION = "exploitation"   # Focus on using known patterns
    BALANCED = "balanced"          # Balance between exploration and exploitation
    ADAPTIVE = "adaptive"          # Dynamically adjust based on context


@dataclass
class LearningExample:
    """Individual learning example"""
    example_id: str
    input_features: Dict[str, float]
    target_output: Dict[str, float]
    context: Dict[str, Any]
    category: PatternCategory
    timestamp: datetime = field(default_factory=datetime.now)
    quality_score: float = 1.0
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnedPattern:
    """Represents a learned pattern"""
    pattern_id: str
    category: PatternCategory
    pattern_data: Dict[str, Any]
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    prediction_accuracy: float = 0.0


@dataclass
class LearningMetrics:
    """Metrics for learning performance"""
    total_examples: int = 0
    patterns_learned: int = 0
    average_confidence: float = 0.0
    learning_rate: float = 0.01
    prediction_accuracy: float = 0.0
    pattern_usage_rate: float = 0.0
    memory_efficiency: float = 0.0
    adaptation_score: float = 0.0


class QuantumFeatureExtractor:
    """Quantum-inspired feature extraction"""
    
    def __init__(self, feature_dim: int = 50):
        self.feature_dim = feature_dim
        self.quantum_weights = [random.uniform(-1, 1) for _ in range(feature_dim)]
        self.phase_shifts = [random.uniform(0, 2*math.pi) for _ in range(feature_dim)]
        self.entanglement_matrix = [[random.uniform(-0.1, 0.1) for _ in range(feature_dim)] 
                                   for _ in range(feature_dim)]
    
    def extract_quantum_features(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract quantum-inspired features from raw data"""
        # Convert raw data to numerical vector
        numerical_vector = self._data_to_vector(raw_data)
        
        # Apply quantum transformations
        quantum_features = {}
        
        for i in range(self.feature_dim):
            # Base feature computation
            base_value = sum(self.quantum_weights[j] * numerical_vector[j % len(numerical_vector)] 
                           for j in range(len(numerical_vector)))
            
            # Quantum phase modulation
            phase_modulated = base_value * math.cos(self.phase_shifts[i] + time.time() * 0.001)
            
            # Entanglement effects
            entanglement_effect = sum(self.entanglement_matrix[i][j] * numerical_vector[j % len(numerical_vector)]
                                    for j in range(len(numerical_vector)))
            
            # Quantum interference
            interference = math.sin(phase_modulated + entanglement_effect) * 0.1
            
            quantum_features[f"q_feature_{i}"] = math.tanh(phase_modulated + interference)
        
        return quantum_features
    
    def _data_to_vector(self, data: Dict[str, Any]) -> List[float]:
        """Convert dictionary data to numerical vector"""
        vector = []
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, str):
                # String to hash-based feature
                vector.append((hash(value) % 1000) / 1000.0)
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            elif isinstance(value, list) and value:
                if isinstance(value[0], (int, float)):
                    vector.extend([float(v) for v in value[:5]])  # Limit to 5 elements
                else:
                    vector.append(len(value))
            elif isinstance(value, dict):
                vector.append(len(value))
            else:
                vector.append(0.0)
        
        # Ensure minimum vector size
        while len(vector) < 10:
            vector.append(0.0)
        
        return vector[:20]  # Limit vector size


class PatternClassifier:
    """Advanced pattern classification system"""
    
    def __init__(self):
        self.patterns = {}
        self.feature_importance = defaultdict(float)
        self.classification_history = deque(maxlen=1000)
        self.confidence_threshold = 0.7
    
    def classify_pattern(self, features: Dict[str, float], context: Dict[str, Any]) -> Tuple[PatternCategory, float]:
        """Classify a pattern based on features and context"""
        # Calculate similarity to known patterns
        best_match = None
        best_score = 0.0
        
        for pattern_id, pattern in self.patterns.items():
            similarity = self._calculate_similarity(features, pattern.feature_weights)
            context_relevance = self._calculate_context_relevance(context, pattern.pattern_data)
            
            combined_score = similarity * 0.7 + context_relevance * 0.3
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = pattern
        
        if best_match and best_score > self.confidence_threshold:
            return best_match.category, best_score
        else:
            # Infer category from context
            inferred_category = self._infer_category_from_context(context)
            return inferred_category, 0.5
    
    def _calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate cosine similarity between feature vectors"""
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        dot_product = sum(features1[key] * features2[key] for key in common_keys)
        norm1 = math.sqrt(sum(features1[key]**2 for key in common_keys))
        norm2 = math.sqrt(sum(features2[key]**2 for key in common_keys))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_context_relevance(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate relevance based on context similarity"""
        # Simple context matching for now
        common_keys = set(context1.keys()) & set(context2.keys())
        
        if not common_keys:
            return 0.0
        
        matches = sum(1 for key in common_keys if str(context1[key]) == str(context2[key]))
        return matches / len(common_keys)
    
    def _infer_category_from_context(self, context: Dict[str, Any]) -> PatternCategory:
        """Infer pattern category from context"""
        context_str = str(context).lower()
        
        if any(word in context_str for word in ['error', 'failure', 'exception', 'bug']):
            return PatternCategory.ERROR
        elif any(word in context_str for word in ['performance', 'speed', 'latency', 'throughput']):
            return PatternCategory.PERFORMANCE
        elif any(word in context_str for word in ['user', 'behavior', 'interaction', 'usage']):
            return PatternCategory.USER_BEHAVIOR
        elif any(word in context_str for word in ['resource', 'memory', 'cpu', 'disk']):
            return PatternCategory.RESOURCE_USAGE
        elif any(word in context_str for word in ['workflow', 'process', 'step', 'pipeline']):
            return PatternCategory.WORKFLOW
        else:
            return PatternCategory.OPTIMIZATION
    
    def add_pattern(self, pattern: LearnedPattern) -> None:
        """Add a new learned pattern"""
        self.patterns[pattern.pattern_id] = pattern
        
        # Update feature importance
        for feature, weight in pattern.feature_weights.items():
            self.feature_importance[feature] += abs(weight) * pattern.confidence


class AdaptiveLearningSystem:
    """Main adaptive learning system"""
    
    def __init__(self, storage_path: Path = None):
        self.system_id = str(uuid.uuid4())[:8]
        self.storage_path = storage_path or Path("adaptive_learning_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Core components
        self.feature_extractor = QuantumFeatureExtractor()
        self.pattern_classifier = PatternClassifier()
        
        # Learning data
        self.learning_examples = deque(maxlen=10000)
        self.learned_patterns = {}
        self.active_patterns = {}
        self.prediction_cache = {}
        
        # Learning parameters
        self.learning_mode = LearningMode.BALANCED
        self.exploration_rate = 0.2
        self.learning_rate = 0.01
        self.forgetting_rate = 0.001
        self.consolidation_threshold = 0.8
        
        # Performance tracking
        self.metrics = LearningMetrics()
        self.performance_history = deque(maxlen=1000)
        self.adaptation_history = []
        
        # Load existing data
        self._load_learning_data()
        
        logger.info(f"Adaptive Learning System initialized [ID: {self.system_id}]")
    
    async def learn_from_example(
        self, 
        input_data: Dict[str, Any], 
        output_data: Dict[str, Any],
        context: Dict[str, Any] = None,
        quality_score: float = 1.0
    ) -> str:
        """Learn from a new example"""
        
        # Extract quantum features
        quantum_features = self.feature_extractor.extract_quantum_features(input_data)
        
        # Create learning example
        example = LearningExample(
            example_id=str(uuid.uuid4())[:8],
            input_features=quantum_features,
            target_output={str(k): float(v) if isinstance(v, (int, float)) else hash(str(v)) % 100 / 100.0 
                          for k, v in output_data.items()},
            context=context or {},
            category=PatternCategory.OPTIMIZATION,  # Default category
            quality_score=quality_score,
            importance=self._calculate_importance(input_data, output_data, context)
        )
        
        # Classify the pattern
        pattern_category, confidence = self.pattern_classifier.classify_pattern(
            quantum_features, example.context
        )
        example.category = pattern_category
        
        # Add to learning examples
        self.learning_examples.append(example)
        self.metrics.total_examples += 1
        
        # Learn pattern if confident enough
        if confidence > self.consolidation_threshold or len(self.learning_examples) % 100 == 0:
            await self._consolidate_patterns()
        
        # Adaptive learning rate adjustment
        await self._adapt_learning_parameters(example, confidence)
        
        logger.info(f"Learned example: {pattern_category.value} (confidence: {confidence:.3f})")
        return example.example_id
    
    async def predict_outcome(
        self, 
        input_data: Dict[str, Any], 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Predict outcome based on learned patterns"""
        
        # Extract features
        quantum_features = self.feature_extractor.extract_quantum_features(input_data)
        
        # Create cache key
        cache_key = hash(str(sorted(quantum_features.items())) + str(sorted((context or {}).items())))
        
        # Check cache
        if cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            if (datetime.now() - cached_result["timestamp"]).seconds < 300:  # 5 minute cache
                return cached_result["prediction"]
        
        # Find matching patterns
        matching_patterns = await self._find_matching_patterns(quantum_features, context or {})
        
        if not matching_patterns:
            # No matching patterns - return default prediction
            prediction = {
                "success_probability": 0.5,
                "confidence": 0.1,
                "predicted_outcome": "unknown",
                "reasoning": "No matching patterns found"
            }
        else:
            # Weighted prediction based on matching patterns
            prediction = await self._generate_weighted_prediction(matching_patterns, quantum_features)
        
        # Cache result
        self.prediction_cache[cache_key] = {
            "prediction": prediction,
            "timestamp": datetime.now()
        }
        
        return prediction
    
    async def adapt_to_feedback(
        self, 
        example_id: str, 
        actual_outcome: Dict[str, Any],
        success: bool = True
    ) -> None:
        """Adapt learning based on feedback"""
        
        # Find the original example
        original_example = None
        for example in self.learning_examples:
            if example.example_id == example_id:
                original_example = example
                break
        
        if not original_example:
            logger.warning(f"Example {example_id} not found for feedback")
            return
        
        # Update pattern success rates
        await self._update_pattern_performance(original_example, actual_outcome, success)
        
        # Adjust learning parameters
        if success:
            self._reinforce_successful_patterns(original_example)
        else:
            self._penalize_failed_patterns(original_example)
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": datetime.now(),
            "example_id": example_id,
            "success": success,
            "learning_rate_before": self.learning_rate,
            "exploration_rate_before": self.exploration_rate
        })
        
        logger.info(f"Adapted to feedback: {example_id} ({'success' if success else 'failure'})")
    
    async def _consolidate_patterns(self) -> None:
        """Consolidate recent examples into learned patterns"""
        logger.info("🧠 Consolidating patterns from recent examples")
        
        # Group examples by category
        category_examples = defaultdict(list)
        for example in list(self.learning_examples)[-100:]:  # Last 100 examples
            category_examples[example.category].append(example)
        
        # Create patterns for each category
        for category, examples in category_examples.items():
            if len(examples) < 3:  # Need minimum examples
                continue
            
            pattern = await self._create_pattern_from_examples(category, examples)
            if pattern:
                self.learned_patterns[pattern.pattern_id] = pattern
                self.pattern_classifier.add_pattern(pattern)
                self.metrics.patterns_learned += 1
                
                logger.info(f"Consolidated pattern: {category.value} (confidence: {pattern.confidence:.3f})")
    
    async def _create_pattern_from_examples(
        self, 
        category: PatternCategory, 
        examples: List[LearningExample]
    ) -> Optional[LearnedPattern]:
        """Create a learned pattern from examples"""
        
        if not examples:
            return None
        
        # Aggregate features
        feature_sums = defaultdict(float)
        feature_counts = defaultdict(int)
        
        for example in examples:
            for feature, value in example.input_features.items():
                feature_sums[feature] += value * example.quality_score
                feature_counts[feature] += 1
        
        # Calculate average features and weights
        feature_weights = {}
        for feature in feature_sums:
            if feature_counts[feature] > 0:
                feature_weights[feature] = feature_sums[feature] / feature_counts[feature]
        
        # Calculate pattern confidence
        confidence = min(0.95, len(examples) / 20.0 + 0.3)  # More examples = higher confidence
        
        # Aggregate target outputs
        output_patterns = defaultdict(list)
        for example in examples:
            for key, value in example.target_output.items():
                output_patterns[key].append(value)
        
        # Calculate average outputs
        pattern_outputs = {}
        for key, values in output_patterns.items():
            pattern_outputs[key] = statistics.mean(values)
        
        # Create pattern
        pattern = LearnedPattern(
            pattern_id=str(uuid.uuid4())[:8],
            category=category,
            pattern_data={
                "input_patterns": feature_weights,
                "output_patterns": pattern_outputs,
                "example_count": len(examples),
                "quality_scores": [e.quality_score for e in examples]
            },
            confidence=confidence,
            feature_weights=feature_weights
        )
        
        return pattern
    
    async def _find_matching_patterns(
        self, 
        features: Dict[str, float], 
        context: Dict[str, Any]
    ) -> List[Tuple[LearnedPattern, float]]:
        """Find patterns matching the given features and context"""
        
        matching_patterns = []
        
        for pattern in self.learned_patterns.values():
            # Calculate feature similarity
            similarity = self._calculate_feature_similarity(features, pattern.feature_weights)
            
            # Calculate context relevance
            context_relevance = self._calculate_context_match(context, pattern.pattern_data)
            
            # Combined matching score
            match_score = similarity * 0.7 + context_relevance * 0.3
            
            if match_score > 0.3:  # Minimum matching threshold
                matching_patterns.append((pattern, match_score))
        
        # Sort by match score
        matching_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return matching_patterns[:5]  # Top 5 matches
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between feature vectors"""
        if not features1 or not features2:
            return 0.0
        
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        # Cosine similarity
        dot_product = sum(features1[k] * features2[k] for k in common_keys)
        norm1 = math.sqrt(sum(features1[k]**2 for k in common_keys))
        norm2 = math.sqrt(sum(features2[k]**2 for k in common_keys))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_context_match(self, context1: Dict[str, Any], pattern_data: Dict[str, Any]) -> float:
        """Calculate context matching score"""
        # Simple implementation - can be enhanced
        return 0.5  # Default medium relevance
    
    async def _generate_weighted_prediction(
        self, 
        matching_patterns: List[Tuple[LearnedPattern, float]], 
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate prediction based on weighted patterns"""
        
        if not matching_patterns:
            return {"success_probability": 0.5, "confidence": 0.0}
        
        # Calculate weighted averages
        total_weight = sum(score * pattern.confidence for pattern, score in matching_patterns)
        weighted_success = 0.0
        weighted_confidence = 0.0
        
        predictions = {}
        
        for pattern, match_score in matching_patterns:
            weight = match_score * pattern.confidence
            
            # Get pattern predictions
            pattern_outputs = pattern.pattern_data.get("output_patterns", {})
            
            for key, value in pattern_outputs.items():
                if key not in predictions:
                    predictions[key] = 0.0
                predictions[key] += value * weight
            
            weighted_success += pattern.success_rate * weight
            weighted_confidence += pattern.confidence * weight
        
        # Normalize predictions
        if total_weight > 0:
            for key in predictions:
                predictions[key] /= total_weight
            weighted_success /= total_weight
            weighted_confidence /= total_weight
        
        # Generate reasoning
        top_pattern = matching_patterns[0][0]
        reasoning = f"Based on {len(matching_patterns)} matching patterns, primarily {top_pattern.category.value}"
        
        return {
            "success_probability": max(0.0, min(1.0, weighted_success)),
            "confidence": max(0.0, min(1.0, weighted_confidence)),
            "predicted_outcome": "success" if weighted_success > 0.6 else "uncertain",
            "reasoning": reasoning,
            "pattern_matches": len(matching_patterns),
            **predictions
        }
    
    def _calculate_importance(
        self, 
        input_data: Dict[str, Any], 
        output_data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> float:
        """Calculate importance score for a learning example"""
        
        importance = 1.0
        
        # Increase importance for error-related examples
        if context and any(word in str(context).lower() for word in ['error', 'failure', 'exception']):
            importance += 0.5
        
        # Increase importance for performance-related examples
        if any(word in str(input_data).lower() for word in ['performance', 'speed', 'time']):
            importance += 0.3
        
        # Increase importance for novel examples (different from recent patterns)
        # Simplified implementation
        importance += random.uniform(0, 0.2)
        
        return min(2.0, importance)
    
    async def _adapt_learning_parameters(self, example: LearningExample, confidence: float) -> None:
        """Adapt learning parameters based on new example"""
        
        # Adjust learning rate based on confidence
        if confidence < 0.3:
            self.learning_rate = min(0.1, self.learning_rate * 1.05)  # Increase for uncertain examples
        elif confidence > 0.8:
            self.learning_rate = max(0.001, self.learning_rate * 0.98)  # Decrease for confident examples
        
        # Adjust exploration rate based on learning mode
        if self.learning_mode == LearningMode.EXPLORATION:
            self.exploration_rate = min(0.5, self.exploration_rate + 0.01)
        elif self.learning_mode == LearningMode.EXPLOITATION:
            self.exploration_rate = max(0.05, self.exploration_rate - 0.01)
        
        # Update metrics
        self.metrics.learning_rate = self.learning_rate
        self.metrics.average_confidence = (
            self.metrics.average_confidence * 0.9 + confidence * 0.1
        )
    
    async def _update_pattern_performance(
        self, 
        example: LearningExample, 
        actual_outcome: Dict[str, Any], 
        success: bool
    ) -> None:
        """Update pattern performance based on actual outcomes"""
        
        # Find patterns that contributed to this prediction
        matching_patterns = await self._find_matching_patterns(example.input_features, example.context)
        
        for pattern, match_score in matching_patterns:
            pattern.usage_count += 1
            
            # Update success rate
            old_success_rate = pattern.success_rate
            pattern.success_rate = (
                old_success_rate * (pattern.usage_count - 1) + (1.0 if success else 0.0)
            ) / pattern.usage_count
            
            pattern.last_used = datetime.now()
            
            # Update prediction accuracy if we have numerical targets
            if isinstance(actual_outcome, dict):
                # Calculate prediction accuracy (simplified)
                predicted = pattern.pattern_data.get("output_patterns", {})
                if predicted:
                    accuracy = 1.0 - abs(
                        list(predicted.values())[0] - list(actual_outcome.values())[0]
                    ) if predicted and actual_outcome else 0.5
                    
                    pattern.prediction_accuracy = (
                        pattern.prediction_accuracy * 0.8 + accuracy * 0.2
                    )
    
    def _reinforce_successful_patterns(self, example: LearningExample) -> None:
        """Reinforce patterns that led to successful outcomes"""
        # Increase confidence slightly for related patterns
        for pattern in self.learned_patterns.values():
            if pattern.category == example.category:
                pattern.confidence = min(0.98, pattern.confidence * 1.01)
    
    def _penalize_failed_patterns(self, example: LearningExample) -> None:
        """Penalize patterns that led to failed outcomes"""
        # Decrease confidence slightly for related patterns
        for pattern in self.learned_patterns.values():
            if pattern.category == example.category:
                pattern.confidence = max(0.1, pattern.confidence * 0.99)
    
    def _save_learning_data(self) -> None:
        """Save learning data to disk"""
        try:
            data = {
                "learned_patterns": self.learned_patterns,
                "metrics": self.metrics,
                "learning_parameters": {
                    "learning_rate": self.learning_rate,
                    "exploration_rate": self.exploration_rate,
                    "learning_mode": self.learning_mode.value
                },
                "saved_at": datetime.now().isoformat()
            }
            
            with open(self.storage_path / "learning_data.pkl", "wb") as f:
                pickle.dump(data, f)
                
            logger.info(f"Learning data saved to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def _load_learning_data(self) -> None:
        """Load learning data from disk"""
        try:
            data_file = self.storage_path / "learning_data.pkl"
            if data_file.exists():
                with open(data_file, "rb") as f:
                    data = pickle.load(f)
                
                self.learned_patterns = data.get("learned_patterns", {})
                self.metrics = data.get("metrics", LearningMetrics())
                
                params = data.get("learning_parameters", {})
                self.learning_rate = params.get("learning_rate", 0.01)
                self.exploration_rate = params.get("exploration_rate", 0.2)
                
                # Add patterns to classifier
                for pattern in self.learned_patterns.values():
                    self.pattern_classifier.add_pattern(pattern)
                
                logger.info(f"Loaded {len(self.learned_patterns)} learned patterns")
                
        except Exception as e:
            logger.warning(f"Failed to load learning data: {e}")
    
    async def optimize_memory(self) -> None:
        """Optimize memory usage by forgetting less important patterns"""
        logger.info("🧹 Optimizing memory usage")
        
        if len(self.learned_patterns) < 1000:
            return  # No need to optimize yet
        
        # Score patterns by importance
        pattern_scores = []
        for pattern_id, pattern in self.learned_patterns.items():
            # Calculate importance score
            recency = (datetime.now() - (pattern.last_used or pattern.created_at)).days
            importance = (
                pattern.confidence * 0.3 +
                pattern.success_rate * 0.3 +
                min(1.0, pattern.usage_count / 100.0) * 0.2 +
                max(0.0, 1.0 - recency / 365.0) * 0.2  # Recency factor
            )
            
            pattern_scores.append((pattern_id, importance))
        
        # Sort by importance and keep top 80%
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        keep_count = int(len(pattern_scores) * 0.8)
        
        patterns_to_remove = [pid for pid, _ in pattern_scores[keep_count:]]
        
        # Remove low-importance patterns
        for pattern_id in patterns_to_remove:
            del self.learned_patterns[pattern_id]
        
        # Clear old prediction cache
        self.prediction_cache.clear()
        
        logger.info(f"Memory optimization complete: kept {keep_count}/{len(pattern_scores)} patterns")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning system status"""
        
        # Calculate recent performance
        recent_patterns = [p for p in self.learned_patterns.values() 
                          if p.last_used and (datetime.now() - p.last_used).days < 7]
        
        return {
            "system_id": self.system_id,
            "learning_mode": self.learning_mode.value,
            "metrics": {
                "total_examples": self.metrics.total_examples,
                "patterns_learned": len(self.learned_patterns),
                "active_patterns": len(recent_patterns),
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate,
                "average_confidence": self.metrics.average_confidence,
            },
            "performance": {
                "pattern_usage_rate": len(recent_patterns) / max(1, len(self.learned_patterns)),
                "memory_efficiency": 1.0 - len(self.prediction_cache) / 10000.0,
                "adaptation_count": len(self.adaptation_history),
            },
            "pattern_categories": {
                category.value: len([p for p in self.learned_patterns.values() if p.category == category])
                for category in PatternCategory
            }
        }


# Global adaptive learning instance
_learning_system: Optional[AdaptiveLearningSystem] = None


def get_adaptive_learning_system() -> AdaptiveLearningSystem:
    """Get or create global adaptive learning system"""
    global _learning_system
    if _learning_system is None:
        _learning_system = AdaptiveLearningSystem()
    return _learning_system


async def learn_from_execution(
    input_data: Dict[str, Any],
    output_data: Dict[str, Any], 
    context: Dict[str, Any] = None,
    success: bool = True
) -> str:
    """Convenience function for learning from execution data"""
    system = get_adaptive_learning_system()
    
    quality_score = 1.0 if success else 0.3
    return await system.learn_from_example(input_data, output_data, context, quality_score)


async def predict_execution_outcome(
    input_data: Dict[str, Any],
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Convenience function for predicting execution outcomes"""
    system = get_adaptive_learning_system()
    return await system.predict_outcome(input_data, context)


async def demo_adaptive_learning():
    """Demonstrate adaptive learning capabilities"""
    print("🧠 Adaptive Learning System Demo")
    print("=" * 50)
    
    system = get_adaptive_learning_system()
    
    # Demo 1: Learn from examples
    print("\n1. Learning from execution examples:")
    examples = [
        ({"task_count": 5, "complexity": 0.7}, {"duration": 120, "success": True}, {"type": "build"}),
        ({"task_count": 3, "complexity": 0.3}, {"duration": 60, "success": True}, {"type": "test"}),
        ({"task_count": 8, "complexity": 0.9}, {"duration": 300, "success": False}, {"type": "deploy"}),
        ({"task_count": 4, "complexity": 0.5}, {"duration": 90, "success": True}, {"type": "build"}),
    ]
    
    for i, (input_data, output_data, context) in enumerate(examples):
        example_id = await system.learn_from_example(input_data, output_data, context)
        print(f"   Learned example {i+1}: {example_id}")
    
    # Demo 2: Make predictions
    print("\n2. Making predictions:")
    predictions = [
        {"task_count": 6, "complexity": 0.6},
        {"task_count": 2, "complexity": 0.2}, 
        {"task_count": 10, "complexity": 0.8}
    ]
    
    for pred_input in predictions:
        prediction = await system.predict_outcome(pred_input, {"type": "build"})
        print(f"   Input: {pred_input}")
        print(f"   Prediction: {prediction['predicted_outcome']} "
              f"(confidence: {prediction['confidence']:.2%})")
    
    # Demo 3: System status
    print(f"\n3. Learning System Status:")
    status = system.get_learning_status()
    print(f"   Learning Mode: {status['learning_mode']}")
    print(f"   Total Examples: {status['metrics']['total_examples']}")
    print(f"   Patterns Learned: {status['metrics']['patterns_learned']}")
    print(f"   Learning Rate: {status['metrics']['learning_rate']:.4f}")
    print(f"   Pattern Categories: {status['pattern_categories']}")
    
    return {
        "status": status,
        "predictions": predictions
    }


if __name__ == "__main__":
    asyncio.run(demo_adaptive_learning())