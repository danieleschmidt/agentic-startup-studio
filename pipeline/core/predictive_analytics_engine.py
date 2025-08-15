"""
Predictive Analytics Engine - Generation 1 Enhancement
Advanced prediction system with neural evolution and quantum insights
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

from pydantic import BaseModel, Field
from opentelemetry import trace

from ..config.settings import get_settings
from ..telemetry import get_tracer
from .neural_evolution_engine import get_evolution_engine, NeuralNetworkType
from .adaptive_intelligence import get_intelligence, PatternType

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class PredictionType(str, Enum):
    """Types of predictions the system can make"""
    IDEA_SUCCESS = "idea_success"
    MARKET_TREND = "market_trend"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_PERFORMANCE = "system_performance"
    BUSINESS_METRIC = "business_metric"
    RISK_ASSESSMENT = "risk_assessment"
    OPPORTUNITY_DETECTION = "opportunity_detection"


class PredictionHorizon(str, Enum):
    """Time horizons for predictions"""
    IMMEDIATE = "immediate"  # Next 1-6 hours
    SHORT_TERM = "short_term"  # Next 1-7 days
    MEDIUM_TERM = "medium_term"  # Next 1-4 weeks
    LONG_TERM = "long_term"  # Next 1-12 months
    STRATEGIC = "strategic"  # Next 1-5 years


class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions"""
    VERY_LOW = "very_low"  # < 30%
    LOW = "low"  # 30-50%
    MEDIUM = "medium"  # 50-70%
    HIGH = "high"  # 70-90%
    VERY_HIGH = "very_high"  # > 90%


@dataclass
class PredictionFeature:
    """Individual feature for prediction models"""
    feature_name: str
    feature_type: str  # numerical, categorical, temporal, textual
    value: Any
    importance: float = 0.0
    confidence: float = 0.0
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def normalize_value(self) -> float:
        """Normalize feature value to 0-1 range"""
        if self.feature_type == "numerical":
            if isinstance(self.value, (int, float)):
                # Simple min-max normalization (in practice, would use learned bounds)
                return max(0.0, min(1.0, self.value / 100.0))
        elif self.feature_type == "categorical":
            # Hash-based normalization
            return hash(str(self.value)) % 1000 / 1000.0
        elif self.feature_type == "temporal":
            if isinstance(self.value, datetime):
                # Normalize based on time of day, day of week, etc.
                hour_norm = self.value.hour / 24.0
                day_norm = self.value.weekday() / 7.0
                return (hour_norm + day_norm) / 2.0
        
        return 0.5  # Default for unknown types


@dataclass
class Prediction:
    """Individual prediction result"""
    prediction_id: str
    prediction_type: PredictionType
    horizon: PredictionHorizon
    target_value: Any
    confidence: float
    confidence_level: ConfidenceLevel
    features_used: List[PredictionFeature]
    model_info: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    actual_outcome: Optional[Any] = None
    accuracy: Optional[float] = None
    
    def __post_init__(self):
        """Set expiration based on horizon"""
        if self.expires_at is None:
            if self.horizon == PredictionHorizon.IMMEDIATE:
                self.expires_at = self.created_at + timedelta(hours=6)
            elif self.horizon == PredictionHorizon.SHORT_TERM:
                self.expires_at = self.created_at + timedelta(days=7)
            elif self.horizon == PredictionHorizon.MEDIUM_TERM:
                self.expires_at = self.created_at + timedelta(weeks=4)
            elif self.horizon == PredictionHorizon.LONG_TERM:
                self.expires_at = self.created_at + timedelta(days=365)
            else:  # STRATEGIC
                self.expires_at = self.created_at + timedelta(days=1825)
    
    def classify_confidence(self) -> ConfidenceLevel:
        """Classify numerical confidence into level"""
        if self.confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.5:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def update_accuracy(self, actual_outcome: Any) -> None:
        """Update prediction accuracy when actual outcome is known"""
        self.actual_outcome = actual_outcome
        
        if self.prediction_type in [PredictionType.IDEA_SUCCESS, PredictionType.RISK_ASSESSMENT]:
            # Binary classification accuracy
            predicted_class = 1 if self.target_value > 0.5 else 0
            actual_class = 1 if actual_outcome > 0.5 else 0
            self.accuracy = 1.0 if predicted_class == actual_class else 0.0
        else:
            # Regression accuracy (inverse of normalized error)
            if isinstance(self.target_value, (int, float)) and isinstance(actual_outcome, (int, float)):
                error = abs(self.target_value - actual_outcome)
                max_possible_error = max(abs(self.target_value), abs(actual_outcome), 1.0)
                self.accuracy = max(0.0, 1.0 - error / max_possible_error)


class PredictiveModel:
    """Advanced predictive model with neural evolution"""
    
    def __init__(self, model_id: str, prediction_type: PredictionType):
        self.model_id = model_id
        self.prediction_type = prediction_type
        self.neural_genome_id: Optional[str] = None
        self.feature_importance: Dict[str, float] = {}
        self.training_data: List[Dict[str, Any]] = []
        self.predictions_made: List[Prediction] = []
        self.performance_metrics: Dict[str, float] = {}
        self.last_training: Optional[datetime] = None
        self.version = 1
        
    async def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Train the predictive model"""
        with tracer.start_as_current_span("train_predictive_model") as span:
            span.set_attributes({
                "model_id": self.model_id,
                "prediction_type": self.prediction_type.value,
                "training_samples": len(training_data)
            })
            
            self.training_data.extend(training_data)
            
            # Get optimal neural architecture from evolution engine
            evolution_engine = await get_evolution_engine()
            
            # Find best genome for this prediction type
            best_genome = None
            best_fitness = 0.0
            
            for genome in evolution_engine.population.values():
                if genome.fitness > best_fitness:
                    best_genome = genome
                    best_fitness = genome.fitness
            
            if best_genome:
                self.neural_genome_id = best_genome.genome_id
                logger.info(f"Model {self.model_id} using neural genome {self.neural_genome_id}")
            
            # Calculate feature importance
            await self._calculate_feature_importance(training_data)
            
            # Update performance metrics
            await self._evaluate_model_performance()
            
            self.last_training = datetime.utcnow()
            self.version += 1
    
    async def _calculate_feature_importance(self, data: List[Dict[str, Any]]) -> None:
        """Calculate importance of each feature"""
        if not data:
            return
        
        # Extract all feature names
        all_features = set()
        for sample in data:
            features = sample.get("features", {})
            all_features.update(features.keys())
        
        # Simple feature importance based on variance and correlation
        for feature_name in all_features:
            values = []
            targets = []
            
            for sample in data:
                if feature_name in sample.get("features", {}):
                    feature_val = sample["features"][feature_name]
                    target_val = sample.get("target", 0)
                    
                    if isinstance(feature_val, (int, float)) and isinstance(target_val, (int, float)):
                        values.append(feature_val)
                        targets.append(target_val)
            
            if len(values) > 1:
                # Calculate correlation-based importance
                correlation = np.corrcoef(values, targets)[0, 1] if len(values) == len(targets) else 0
                variance = np.var(values)
                
                importance = abs(correlation) * (1 + variance)
                self.feature_importance[feature_name] = importance
            else:
                self.feature_importance[feature_name] = 0.1  # Default low importance
    
    async def _evaluate_model_performance(self) -> None:
        """Evaluate model performance on historical predictions"""
        if not self.predictions_made:
            return
        
        # Calculate accuracy for predictions with known outcomes
        accuracies = [p.accuracy for p in self.predictions_made if p.accuracy is not None]
        
        if accuracies:
            self.performance_metrics = {
                "average_accuracy": np.mean(accuracies),
                "accuracy_std": np.std(accuracies),
                "predictions_evaluated": len(accuracies),
                "total_predictions": len(self.predictions_made)
            }
        
        # Calculate confidence calibration
        self._calculate_confidence_calibration()
    
    def _calculate_confidence_calibration(self) -> None:
        """Calculate how well-calibrated the model's confidence estimates are"""
        evaluated_predictions = [p for p in self.predictions_made if p.accuracy is not None]
        
        if len(evaluated_predictions) < 10:
            return
        
        # Group predictions by confidence level
        confidence_groups = {}
        for pred in evaluated_predictions:
            conf_level = pred.classify_confidence().value
            if conf_level not in confidence_groups:
                confidence_groups[conf_level] = []
            confidence_groups[conf_level].append(pred)
        
        # Calculate calibration for each group
        calibration_scores = {}
        for conf_level, predictions in confidence_groups.items():
            if len(predictions) >= 3:
                avg_confidence = np.mean([p.confidence for p in predictions])
                avg_accuracy = np.mean([p.accuracy for p in predictions])
                calibration_error = abs(avg_confidence - avg_accuracy)
                calibration_scores[conf_level] = 1.0 - calibration_error
        
        if calibration_scores:
            self.performance_metrics["calibration_score"] = np.mean(list(calibration_scores.values()))
    
    async def predict(self, features: Dict[str, Any], horizon: PredictionHorizon) -> Prediction:
        """Make a prediction using the model"""
        with tracer.start_as_current_span("make_prediction") as span:
            span.set_attributes({
                "model_id": self.model_id,
                "prediction_type": self.prediction_type.value,
                "horizon": horizon.value
            })
            
            # Convert features to PredictionFeature objects
            prediction_features = []
            for name, value in features.items():
                feature_type = self._infer_feature_type(value)
                importance = self.feature_importance.get(name, 0.1)
                
                pred_feature = PredictionFeature(
                    feature_name=name,
                    feature_type=feature_type,
                    value=value,
                    importance=importance,
                    confidence=min(1.0, importance * 2),  # Simple confidence estimate
                    source="input"
                )
                prediction_features.append(pred_feature)
            
            # Generate prediction using neural network simulation
            target_value = await self._neural_network_predict(prediction_features)
            
            # Calculate confidence based on feature importance and model performance
            confidence = self._calculate_prediction_confidence(prediction_features)
            
            # Create prediction object
            prediction = Prediction(
                prediction_id=f"{self.model_id}_{int(time.time())}",
                prediction_type=self.prediction_type,
                horizon=horizon,
                target_value=target_value,
                confidence=confidence,
                confidence_level=ConfidenceLevel.MEDIUM,  # Will be calculated
                features_used=prediction_features,
                model_info={
                    "model_id": self.model_id,
                    "version": self.version,
                    "neural_genome_id": self.neural_genome_id,
                    "features_count": len(prediction_features)
                }
            )
            
            prediction.confidence_level = prediction.classify_confidence()
            self.predictions_made.append(prediction)
            
            return prediction
    
    def _infer_feature_type(self, value: Any) -> str:
        """Infer the type of a feature value"""
        if isinstance(value, (int, float)):
            return "numerical"
        elif isinstance(value, datetime):
            return "temporal"
        elif isinstance(value, str):
            try:
                float(value)
                return "numerical"
            except ValueError:
                return "categorical"
        else:
            return "categorical"
    
    async def _neural_network_predict(self, features: List[PredictionFeature]) -> float:
        """Simulate neural network prediction"""
        # Normalize features
        feature_values = [f.normalize_value() for f in features]
        
        if not feature_values:
            return 0.5  # Default prediction
        
        # Weight by feature importance
        weighted_sum = sum(
            val * f.importance 
            for val, f in zip(feature_values, features)
        )
        
        total_importance = sum(f.importance for f in features)
        if total_importance > 0:
            weighted_average = weighted_sum / total_importance
        else:
            weighted_average = np.mean(feature_values)
        
        # Apply neural network simulation
        if self.prediction_type == PredictionType.IDEA_SUCCESS:
            # Sigmoid activation for binary classification
            return 1 / (1 + np.exp(-5 * (weighted_average - 0.5)))
        elif self.prediction_type == PredictionType.MARKET_TREND:
            # Tanh activation for trend prediction
            return (np.tanh(3 * (weighted_average - 0.5)) + 1) / 2
        else:
            # Linear with some non-linearity
            return max(0.0, min(1.0, weighted_average + 0.1 * np.sin(weighted_average * np.pi)))
    
    def _calculate_prediction_confidence(self, features: List[PredictionFeature]) -> float:
        """Calculate confidence for a prediction"""
        if not features:
            return 0.1
        
        # Base confidence from feature quality
        feature_confidences = [f.confidence for f in features]
        base_confidence = np.mean(feature_confidences)
        
        # Adjust based on model performance
        if "average_accuracy" in self.performance_metrics:
            model_confidence = self.performance_metrics["average_accuracy"]
            combined_confidence = 0.6 * base_confidence + 0.4 * model_confidence
        else:
            combined_confidence = base_confidence
        
        # Adjust based on feature coverage
        important_features_used = sum(1 for f in features if f.importance > 0.5)
        coverage_bonus = min(0.2, important_features_used * 0.05)
        
        final_confidence = min(0.95, combined_confidence + coverage_bonus)
        return max(0.05, final_confidence)


class PredictiveAnalyticsEngine:
    """
    Advanced predictive analytics system with neural evolution
    """
    
    def __init__(self):
        self.models: Dict[str, PredictiveModel] = {}
        self.prediction_history: List[Prediction] = []
        self.feature_store: Dict[str, List[PredictionFeature]] = {}
        self.model_registry: Dict[PredictionType, List[str]] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.active_learning_queue: List[Dict[str, Any]] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._analytics_active = True
        
    async def initialize(self) -> None:
        """Initialize the predictive analytics engine"""
        with tracer.start_as_current_span("initialize_analytics_engine"):
            logger.info("Initializing Predictive Analytics Engine")
            
            # Create models for different prediction types
            for pred_type in PredictionType:
                await self._create_model_for_type(pred_type)
            
            # Start background processes
            asyncio.create_task(self._feature_collection_loop())
            asyncio.create_task(self._model_retraining_loop())
            asyncio.create_task(self._prediction_validation_loop())
            asyncio.create_task(self._ensemble_optimization_loop())
            
            logger.info(f"Initialized {len(self.models)} predictive models")
    
    async def _create_model_for_type(self, prediction_type: PredictionType) -> None:
        """Create and initialize a model for a specific prediction type"""
        model_id = f"{prediction_type.value}_model_v1"
        model = PredictiveModel(model_id, prediction_type)
        
        self.models[model_id] = model
        
        if prediction_type not in self.model_registry:
            self.model_registry[prediction_type] = []
        self.model_registry[prediction_type].append(model_id)
        
        # Initialize with some synthetic training data
        await self._generate_synthetic_training_data(model)
    
    async def _generate_synthetic_training_data(self, model: PredictiveModel) -> None:
        """Generate synthetic training data for initial model training"""
        synthetic_data = []
        
        for i in range(100):  # Generate 100 synthetic samples
            if model.prediction_type == PredictionType.IDEA_SUCCESS:
                # Synthetic startup idea success data
                features = {
                    "market_size": np.random.uniform(1, 10),
                    "team_experience": np.random.uniform(0, 5),
                    "funding_amount": np.random.uniform(0, 1000000),
                    "competition_level": np.random.uniform(1, 10),
                    "innovation_score": np.random.uniform(0, 1)
                }
                # Success probability based on features
                success_prob = (
                    features["market_size"] * 0.2 +
                    features["team_experience"] * 0.3 +
                    np.log(features["funding_amount"] + 1) * 0.1 +
                    (10 - features["competition_level"]) * 0.1 +
                    features["innovation_score"] * 0.3
                ) / 10.0
                target = min(1.0, max(0.0, success_prob + np.random.normal(0, 0.1)))
                
            elif model.prediction_type == PredictionType.MARKET_TREND:
                # Synthetic market trend data
                features = {
                    "search_volume": np.random.uniform(100, 10000),
                    "social_mentions": np.random.uniform(10, 1000),
                    "competitor_activity": np.random.uniform(0, 1),
                    "economic_indicators": np.random.uniform(-0.1, 0.1),
                    "seasonal_factor": np.sin(i * np.pi / 50)  # Simulate seasonality
                }
                trend = (
                    np.log(features["search_volume"]) * 0.3 +
                    np.log(features["social_mentions"]) * 0.2 +
                    features["competitor_activity"] * 0.2 +
                    features["economic_indicators"] * 0.1 +
                    features["seasonal_factor"] * 0.2
                ) / 5.0
                target = max(0.0, min(1.0, trend + np.random.normal(0, 0.05)))
                
            else:
                # Generic synthetic data
                features = {
                    f"feature_{j}": np.random.uniform(0, 10)
                    for j in range(5)
                }
                target = np.mean(list(features.values())) / 10.0
            
            synthetic_data.append({
                "features": features,
                "target": target,
                "timestamp": datetime.utcnow() - timedelta(days=np.random.randint(1, 365))
            })
        
        await model.train(synthetic_data)
    
    async def add_training_data(
        self, 
        prediction_type: PredictionType, 
        features: Dict[str, Any], 
        target: float
    ) -> None:
        """Add new training data for a specific prediction type"""
        with tracer.start_as_current_span("add_training_data"):
            training_sample = {
                "features": features,
                "target": target,
                "timestamp": datetime.utcnow()
            }
            
            # Add to active learning queue for batch processing
            self.active_learning_queue.append({
                "prediction_type": prediction_type,
                "sample": training_sample
            })
    
    async def predict(
        self, 
        prediction_type: PredictionType, 
        features: Dict[str, Any], 
        horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
        use_ensemble: bool = True
    ) -> Union[Prediction, List[Prediction]]:
        """Make a prediction using the specified type"""
        with tracer.start_as_current_span("predict") as span:
            span.set_attributes({
                "prediction_type": prediction_type.value,
                "horizon": horizon.value,
                "use_ensemble": use_ensemble
            })
            
            if use_ensemble:
                return await self._ensemble_predict(prediction_type, features, horizon)
            else:
                # Use single best model
                model_ids = self.model_registry.get(prediction_type, [])
                if not model_ids:
                    raise ValueError(f"No models available for {prediction_type.value}")
                
                # Use the best performing model
                best_model_id = self._get_best_model(prediction_type)
                model = self.models[best_model_id]
                
                prediction = await model.predict(features, horizon)
                self.prediction_history.append(prediction)
                
                return prediction
    
    async def _ensemble_predict(
        self, 
        prediction_type: PredictionType, 
        features: Dict[str, Any], 
        horizon: PredictionHorizon
    ) -> List[Prediction]:
        """Make ensemble predictions using multiple models"""
        model_ids = self.model_registry.get(prediction_type, [])
        if not model_ids:
            raise ValueError(f"No models available for {prediction_type.value}")
        
        # Get predictions from all models
        individual_predictions = []
        for model_id in model_ids:
            model = self.models[model_id]
            prediction = await model.predict(features, horizon)
            individual_predictions.append(prediction)
        
        # Create ensemble prediction
        ensemble_prediction = await self._combine_predictions(
            individual_predictions, prediction_type, features, horizon
        )
        
        self.prediction_history.extend(individual_predictions + [ensemble_prediction])
        
        return individual_predictions + [ensemble_prediction]
    
    async def _combine_predictions(
        self, 
        predictions: List[Prediction], 
        prediction_type: PredictionType,
        features: Dict[str, Any],
        horizon: PredictionHorizon
    ) -> Prediction:
        """Combine multiple predictions into an ensemble prediction"""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        # Weight predictions by model performance and confidence
        weighted_values = []
        weighted_confidences = []
        total_weight = 0.0
        
        for pred in predictions:
            model_id = pred.model_info["model_id"]
            
            # Get model weight (based on performance)
            model_weight = self.ensemble_weights.get(model_id, 1.0)
            
            # Combine model weight with prediction confidence
            combined_weight = model_weight * pred.confidence
            
            weighted_values.append(pred.target_value * combined_weight)
            weighted_confidences.append(pred.confidence * combined_weight)
            total_weight += combined_weight
        
        if total_weight > 0:
            ensemble_value = sum(weighted_values) / total_weight
            ensemble_confidence = sum(weighted_confidences) / total_weight
        else:
            ensemble_value = np.mean([p.target_value for p in predictions])
            ensemble_confidence = np.mean([p.confidence for p in predictions])
        
        # Create ensemble prediction
        ensemble_features = []
        for name, value in features.items():
            feature_type = predictions[0].features_used[0].feature_type if predictions[0].features_used else "numerical"
            ensemble_features.append(PredictionFeature(
                feature_name=name,
                feature_type=feature_type,
                value=value,
                importance=1.0,
                confidence=ensemble_confidence,
                source="ensemble"
            ))
        
        ensemble_prediction = Prediction(
            prediction_id=f"ensemble_{prediction_type.value}_{int(time.time())}",
            prediction_type=prediction_type,
            horizon=horizon,
            target_value=ensemble_value,
            confidence=ensemble_confidence,
            confidence_level=ConfidenceLevel.MEDIUM,
            features_used=ensemble_features,
            model_info={
                "model_id": "ensemble",
                "version": 1,
                "component_models": [p.model_info["model_id"] for p in predictions],
                "ensemble_method": "weighted_average"
            }
        )
        
        ensemble_prediction.confidence_level = ensemble_prediction.classify_confidence()
        
        return ensemble_prediction
    
    def _get_best_model(self, prediction_type: PredictionType) -> str:
        """Get the best performing model for a prediction type"""
        model_ids = self.model_registry.get(prediction_type, [])
        if not model_ids:
            raise ValueError(f"No models available for {prediction_type.value}")
        
        best_model_id = model_ids[0]
        best_performance = 0.0
        
        for model_id in model_ids:
            model = self.models[model_id]
            performance = model.performance_metrics.get("average_accuracy", 0.0)
            if performance > best_performance:
                best_performance = performance
                best_model_id = model_id
        
        return best_model_id
    
    async def update_prediction_outcome(
        self, 
        prediction_id: str, 
        actual_outcome: Any
    ) -> None:
        """Update a prediction with its actual outcome"""
        with tracer.start_as_current_span("update_prediction_outcome"):
            # Find the prediction
            prediction = None
            for pred in self.prediction_history:
                if pred.prediction_id == prediction_id:
                    prediction = pred
                    break
            
            if prediction:
                prediction.update_accuracy(actual_outcome)
                
                # Update model performance
                model_id = prediction.model_info["model_id"]
                if model_id in self.models:
                    model = self.models[model_id]
                    await model._evaluate_model_performance()
                
                logger.info(f"Updated prediction {prediction_id} with accuracy {prediction.accuracy}")
            else:
                logger.warning(f"Prediction {prediction_id} not found")
    
    async def _feature_collection_loop(self) -> None:
        """Continuously collect features from various sources"""
        while self._analytics_active:
            try:
                await self._collect_system_features()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                logger.error(f"Error in feature collection: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_features(self) -> None:
        """Collect features from system metrics and adaptive intelligence"""
        with tracer.start_as_current_span("collect_system_features"):
            # Get intelligence data
            intelligence = await get_intelligence()
            intelligence_report = intelligence.get_intelligence_report()
            
            # Convert intelligence metrics to features
            system_features = []
            
            if "patterns_detected" in intelligence_report:
                feature = PredictionFeature(
                    feature_name="patterns_detected",
                    feature_type="numerical",
                    value=intelligence_report["patterns_detected"],
                    importance=0.7,
                    source="adaptive_intelligence"
                )
                system_features.append(feature)
            
            if "adaptation_success_rate" in intelligence_report:
                feature = PredictionFeature(
                    feature_name="adaptation_success_rate",
                    feature_type="numerical",
                    value=intelligence_report["adaptation_success_rate"],
                    importance=0.8,
                    source="adaptive_intelligence"
                )
                system_features.append(feature)
            
            # Store features in feature store
            timestamp_key = datetime.utcnow().strftime("%Y-%m-%d-%H")
            if timestamp_key not in self.feature_store:
                self.feature_store[timestamp_key] = []
            self.feature_store[timestamp_key].extend(system_features)
            
            # Clean up old features (keep only last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            keys_to_remove = [
                key for key in self.feature_store.keys()
                if datetime.strptime(key, "%Y-%m-%d-%H") < cutoff_time
            ]
            for key in keys_to_remove:
                del self.feature_store[key]
    
    async def _model_retraining_loop(self) -> None:
        """Continuously retrain models with new data"""
        while self._analytics_active:
            try:
                await self._process_active_learning_queue()
                await asyncio.sleep(300)  # Process every 5 minutes
            except Exception as e:
                logger.error(f"Error in model retraining: {e}")
                await asyncio.sleep(300)
    
    async def _process_active_learning_queue(self) -> None:
        """Process the active learning queue"""
        if not self.active_learning_queue:
            return
        
        with tracer.start_as_current_span("process_active_learning"):
            # Group samples by prediction type
            samples_by_type = {}
            for item in self.active_learning_queue:
                pred_type = item["prediction_type"]
                if pred_type not in samples_by_type:
                    samples_by_type[pred_type] = []
                samples_by_type[pred_type].append(item["sample"])
            
            # Retrain models with new data
            for pred_type, samples in samples_by_type.items():
                model_ids = self.model_registry.get(pred_type, [])
                for model_id in model_ids:
                    model = self.models[model_id]
                    await model.train(samples)
            
            # Clear the queue
            self.active_learning_queue.clear()
            
            logger.info(f"Processed {sum(len(samples) for samples in samples_by_type.values())} training samples")
    
    async def _prediction_validation_loop(self) -> None:
        """Validate predictions that have expired"""
        while self._analytics_active:
            try:
                await self._validate_expired_predictions()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Error in prediction validation: {e}")
                await asyncio.sleep(3600)
    
    async def _validate_expired_predictions(self) -> None:
        """Check for expired predictions and attempt validation"""
        current_time = datetime.utcnow()
        expired_predictions = [
            pred for pred in self.prediction_history
            if pred.expires_at and pred.expires_at < current_time and pred.actual_outcome is None
        ]
        
        for prediction in expired_predictions:
            # Attempt to validate prediction (in real system, would check external sources)
            if prediction.prediction_type == PredictionType.IDEA_SUCCESS:
                # Simulate checking if idea succeeded
                simulated_outcome = np.random.random()  # Would be real data
                await self.update_prediction_outcome(prediction.prediction_id, simulated_outcome)
    
    async def _ensemble_optimization_loop(self) -> None:
        """Optimize ensemble weights based on model performance"""
        while self._analytics_active:
            try:
                await self._optimize_ensemble_weights()
                await asyncio.sleep(1800)  # Optimize every 30 minutes
            except Exception as e:
                logger.error(f"Error in ensemble optimization: {e}")
                await asyncio.sleep(1800)
    
    async def _optimize_ensemble_weights(self) -> None:
        """Optimize weights for ensemble models"""
        with tracer.start_as_current_span("optimize_ensemble_weights"):
            # Calculate weights based on recent model performance
            for model_id, model in self.models.items():
                if "average_accuracy" in model.performance_metrics:
                    accuracy = model.performance_metrics["average_accuracy"]
                    # Weight based on accuracy and recency
                    recency_factor = 1.0
                    if model.last_training:
                        days_since_training = (datetime.utcnow() - model.last_training).days
                        recency_factor = max(0.1, 1.0 - days_since_training * 0.1)
                    
                    weight = accuracy * recency_factor
                    self.ensemble_weights[model_id] = weight
                else:
                    self.ensemble_weights[model_id] = 0.5  # Default weight
            
            logger.debug(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        with tracer.start_as_current_span("analytics_report"):
            # Model performance summary
            model_performance = {}
            for model_id, model in self.models.items():
                model_performance[model_id] = {
                    "prediction_type": model.prediction_type.value,
                    "version": model.version,
                    "predictions_made": len(model.predictions_made),
                    "performance_metrics": model.performance_metrics,
                    "feature_importance": dict(list(model.feature_importance.items())[:5])  # Top 5
                }
            
            # Prediction statistics
            total_predictions = len(self.prediction_history)
            predictions_by_type = {}
            predictions_by_confidence = {}
            
            for pred in self.prediction_history:
                pred_type = pred.prediction_type.value
                predictions_by_type[pred_type] = predictions_by_type.get(pred_type, 0) + 1
                
                conf_level = pred.confidence_level.value
                predictions_by_confidence[conf_level] = predictions_by_confidence.get(conf_level, 0) + 1
            
            # Recent performance trends
            recent_predictions = [
                pred for pred in self.prediction_history
                if pred.created_at > datetime.utcnow() - timedelta(days=7)
            ]
            
            recent_accuracies = [
                pred.accuracy for pred in recent_predictions
                if pred.accuracy is not None
            ]
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_models": len(self.models),
                "total_predictions": total_predictions,
                "recent_predictions": len(recent_predictions),
                "model_performance": model_performance,
                "prediction_distribution": {
                    "by_type": predictions_by_type,
                    "by_confidence": predictions_by_confidence
                },
                "performance_trends": {
                    "recent_average_accuracy": np.mean(recent_accuracies) if recent_accuracies else 0.0,
                    "recent_accuracy_count": len(recent_accuracies),
                    "ensemble_weights": self.ensemble_weights
                },
                "feature_store_size": sum(len(features) for features in self.feature_store.values()),
                "active_learning_queue_size": len(self.active_learning_queue)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the analytics engine"""
        logger.info("Shutting down Predictive Analytics Engine")
        self._analytics_active = False
        self.executor.shutdown(wait=True)


# Global analytics engine instance
_analytics_engine: Optional[PredictiveAnalyticsEngine] = None


async def get_analytics_engine() -> PredictiveAnalyticsEngine:
    """Get or create the global predictive analytics engine"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = PredictiveAnalyticsEngine()
        await _analytics_engine.initialize()
    return _analytics_engine


async def predict_idea_success(idea_features: Dict[str, Any]) -> Prediction:
    """Convenience function to predict startup idea success"""
    engine = await get_analytics_engine()
    return await engine.predict(
        PredictionType.IDEA_SUCCESS,
        idea_features,
        PredictionHorizon.MEDIUM_TERM
    )


async def predict_market_trend(market_features: Dict[str, Any]) -> Prediction:
    """Convenience function to predict market trends"""
    engine = await get_analytics_engine()
    return await engine.predict(
        PredictionType.MARKET_TREND,
        market_features,
        PredictionHorizon.SHORT_TERM
    )