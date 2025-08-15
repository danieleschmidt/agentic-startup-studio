"""
Test suite for Predictive Analytics Engine - Generation 1 Enhancement
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from pipeline.core.predictive_analytics_engine import (
    PredictiveAnalyticsEngine,
    PredictiveModel,
    Prediction,
    PredictionFeature,
    PredictionType,
    PredictionHorizon,
    ConfidenceLevel,
    get_analytics_engine,
    predict_idea_success,
    predict_market_trend
)


class TestPredictionFeature:
    """Test prediction feature functionality"""
    
    def test_feature_creation(self):
        """Test prediction feature creation"""
        feature = PredictionFeature(
            feature_name="test_feature",
            feature_type="numerical",
            value=42.5,
            importance=0.8
        )
        
        assert feature.feature_name == "test_feature"
        assert feature.feature_type == "numerical"
        assert feature.value == 42.5
        assert feature.importance == 0.8
        assert isinstance(feature.timestamp, datetime)
    
    def test_numerical_normalization(self):
        """Test numerical value normalization"""
        feature = PredictionFeature(
            feature_name="numerical",
            feature_type="numerical",
            value=50.0
        )
        
        normalized = feature.normalize_value()
        assert 0 <= normalized <= 1
        assert normalized == 0.5  # 50/100
    
    def test_categorical_normalization(self):
        """Test categorical value normalization"""
        feature = PredictionFeature(
            feature_name="category",
            feature_type="categorical",
            value="test_category"
        )
        
        normalized = feature.normalize_value()
        assert 0 <= normalized <= 1
        # Should be consistent for same value
        assert normalized == feature.normalize_value()
    
    def test_temporal_normalization(self):
        """Test temporal value normalization"""
        test_time = datetime(2024, 6, 15, 14, 30)  # 2:30 PM on Saturday
        feature = PredictionFeature(
            feature_name="timestamp",
            feature_type="temporal",
            value=test_time
        )
        
        normalized = feature.normalize_value()
        assert 0 <= normalized <= 1
        # Should incorporate hour and day information
    
    def test_unknown_type_normalization(self):
        """Test unknown type normalization"""
        feature = PredictionFeature(
            feature_name="unknown",
            feature_type="unknown",
            value="whatever"
        )
        
        normalized = feature.normalize_value()
        assert normalized == 0.5  # Default value


class TestPrediction:
    """Test prediction functionality"""
    
    def test_prediction_creation(self):
        """Test prediction creation"""
        features = [
            PredictionFeature("feature1", "numerical", 0.8),
            PredictionFeature("feature2", "categorical", "high")
        ]
        
        prediction = Prediction(
            prediction_id="test_pred",
            prediction_type=PredictionType.IDEA_SUCCESS,
            horizon=PredictionHorizon.SHORT_TERM,
            target_value=0.75,
            confidence=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            features_used=features,
            model_info={"model_id": "test_model"}
        )
        
        assert prediction.prediction_id == "test_pred"
        assert prediction.prediction_type == PredictionType.IDEA_SUCCESS
        assert prediction.horizon == PredictionHorizon.SHORT_TERM
        assert prediction.target_value == 0.75
        assert prediction.confidence == 0.85
        assert len(prediction.features_used) == 2
        assert prediction.actual_outcome is None
        assert prediction.accuracy is None
    
    def test_expiration_setting(self):
        """Test automatic expiration setting"""
        prediction = Prediction(
            prediction_id="test",
            prediction_type=PredictionType.MARKET_TREND,
            horizon=PredictionHorizon.IMMEDIATE,
            target_value=0.5,
            confidence=0.7,
            confidence_level=ConfidenceLevel.MEDIUM,
            features_used=[],
            model_info={}
        )
        
        assert prediction.expires_at is not None
        assert prediction.expires_at > prediction.created_at
        # Should expire in 6 hours for immediate horizon
        expected_expiry = prediction.created_at + timedelta(hours=6)
        assert abs((prediction.expires_at - expected_expiry).total_seconds()) < 60
    
    def test_confidence_classification(self):
        """Test confidence level classification"""
        low_pred = Prediction(
            "test", PredictionType.IDEA_SUCCESS, PredictionHorizon.SHORT_TERM,
            0.5, 0.4, ConfidenceLevel.LOW, [], {}
        )
        assert low_pred.classify_confidence() == ConfidenceLevel.LOW
        
        high_pred = Prediction(
            "test", PredictionType.IDEA_SUCCESS, PredictionHorizon.SHORT_TERM,
            0.5, 0.85, ConfidenceLevel.HIGH, [], {}
        )
        assert high_pred.classify_confidence() == ConfidenceLevel.HIGH
    
    def test_binary_accuracy_update(self):
        """Test accuracy update for binary classification"""
        prediction = Prediction(
            "test", PredictionType.IDEA_SUCCESS, PredictionHorizon.SHORT_TERM,
            0.8, 0.7, ConfidenceLevel.HIGH, [], {}
        )
        
        # Test correct prediction (both > 0.5)
        prediction.update_accuracy(0.9)
        assert prediction.accuracy == 1.0
        assert prediction.actual_outcome == 0.9
        
        # Test incorrect prediction
        prediction2 = Prediction(
            "test2", PredictionType.IDEA_SUCCESS, PredictionHorizon.SHORT_TERM,
            0.8, 0.7, ConfidenceLevel.HIGH, [], {}
        )
        prediction2.update_accuracy(0.2)
        assert prediction2.accuracy == 0.0
    
    def test_regression_accuracy_update(self):
        """Test accuracy update for regression"""
        prediction = Prediction(
            "test", PredictionType.MARKET_TREND, PredictionHorizon.SHORT_TERM,
            0.6, 0.7, ConfidenceLevel.HIGH, [], {}
        )
        
        # Close prediction should have high accuracy
        prediction.update_accuracy(0.65)
        assert prediction.accuracy > 0.9
        
        # Far prediction should have low accuracy
        prediction2 = Prediction(
            "test2", PredictionType.MARKET_TREND, PredictionHorizon.SHORT_TERM,
            0.2, 0.7, ConfidenceLevel.HIGH, [], {}
        )
        prediction2.update_accuracy(0.9)
        assert prediction2.accuracy < 0.5


@pytest.mark.asyncio
class TestPredictiveModel:
    """Test predictive model functionality"""
    
    async def test_model_creation(self):
        """Test predictive model creation"""
        model = PredictiveModel("test_model", PredictionType.IDEA_SUCCESS)
        
        assert model.model_id == "test_model"
        assert model.prediction_type == PredictionType.IDEA_SUCCESS
        assert model.version == 1
        assert len(model.training_data) == 0
        assert len(model.predictions_made) == 0
        assert model.neural_genome_id is None
    
    async def test_model_training(self):
        """Test model training"""
        model = PredictiveModel("test_model", PredictionType.IDEA_SUCCESS)
        
        training_data = [
            {
                "features": {"feature1": 0.8, "feature2": 0.6},
                "target": 0.75
            },
            {
                "features": {"feature1": 0.3, "feature2": 0.4},
                "target": 0.25
            }
        ]
        
        with patch('pipeline.core.predictive_analytics_engine.get_evolution_engine') as mock_evolution:
            # Mock the evolution engine
            mock_engine = Mock()
            mock_genome = Mock()
            mock_genome.genome_id = "test_genome"
            mock_genome.fitness = 0.8
            mock_engine.population.values.return_value = [mock_genome]
            mock_evolution.return_value = mock_engine
            
            await model.train(training_data)
        
        assert len(model.training_data) == 2
        assert model.version == 2
        assert model.neural_genome_id == "test_genome"
        assert len(model.feature_importance) > 0
        assert model.last_training is not None
    
    async def test_feature_importance_calculation(self):
        """Test feature importance calculation"""
        model = PredictiveModel("test_model", PredictionType.MARKET_TREND)
        
        training_data = [
            {"features": {"feature1": 1.0, "feature2": 0.5}, "target": 0.8},
            {"features": {"feature1": 0.5, "feature2": 1.0}, "target": 0.6},
            {"features": {"feature1": 0.2, "feature2": 0.3}, "target": 0.2}
        ]
        
        await model._calculate_feature_importance(training_data)
        
        assert "feature1" in model.feature_importance
        assert "feature2" in model.feature_importance
        assert all(importance >= 0 for importance in model.feature_importance.values())
    
    async def test_model_prediction(self):
        """Test model prediction"""
        model = PredictiveModel("test_model", PredictionType.IDEA_SUCCESS)
        model.feature_importance = {"feature1": 0.8, "feature2": 0.6}
        
        features = {"feature1": 0.7, "feature2": 0.5}
        
        prediction = await model.predict(features, PredictionHorizon.SHORT_TERM)
        
        assert isinstance(prediction, Prediction)
        assert prediction.prediction_type == PredictionType.IDEA_SUCCESS
        assert prediction.horizon == PredictionHorizon.SHORT_TERM
        assert 0 <= prediction.target_value <= 1
        assert 0 <= prediction.confidence <= 1
        assert len(prediction.features_used) == 2
        assert prediction.model_info["model_id"] == "test_model"
    
    async def test_neural_network_simulation(self):
        """Test neural network prediction simulation"""
        model = PredictiveModel("test_model", PredictionType.IDEA_SUCCESS)
        
        features = [
            PredictionFeature("feat1", "numerical", 0.8, importance=0.9),
            PredictionFeature("feat2", "numerical", 0.6, importance=0.7)
        ]
        
        result = await model._neural_network_predict(features)
        
        assert 0 <= result <= 1
        # For idea success, should use sigmoid activation
        assert isinstance(result, float)
    
    def test_feature_type_inference(self):
        """Test feature type inference"""
        model = PredictiveModel("test", PredictionType.MARKET_TREND)
        
        assert model._infer_feature_type(42) == "numerical"
        assert model._infer_feature_type(3.14) == "numerical"
        assert model._infer_feature_type(datetime.now()) == "temporal"
        assert model._infer_feature_type("category") == "categorical"
        assert model._infer_feature_type("42.5") == "numerical"  # String number
    
    async def test_confidence_calculation(self):
        """Test prediction confidence calculation"""
        model = PredictiveModel("test_model", PredictionType.MARKET_TREND)
        model.performance_metrics = {"average_accuracy": 0.8}
        
        features = [
            PredictionFeature("feat1", "numerical", 0.5, confidence=0.7, importance=0.8),
            PredictionFeature("feat2", "numerical", 0.3, confidence=0.6, importance=0.6)
        ]
        
        confidence = model._calculate_prediction_confidence(features)
        
        assert 0 <= confidence <= 1
        # Should be influenced by both feature confidence and model performance


@pytest.mark.asyncio
class TestPredictiveAnalyticsEngine:
    """Test predictive analytics engine functionality"""
    
    async def test_engine_initialization(self):
        """Test analytics engine initialization"""
        engine = PredictiveAnalyticsEngine()
        
        with patch('pipeline.core.predictive_analytics_engine.get_intelligence') as mock_intelligence:
            mock_intel = Mock()
            mock_intel.get_intelligence_report.return_value = {}
            mock_intelligence.return_value = mock_intel
            
            await engine.initialize()
        
        assert len(engine.models) > 0
        assert len(engine.model_registry) > 0
        # Should have models for each prediction type
        assert len(engine.models) >= len(PredictionType)
    
    async def test_model_creation_for_types(self):
        """Test model creation for different prediction types"""
        engine = PredictiveAnalyticsEngine()
        
        with patch.object(engine, '_generate_synthetic_training_data') as mock_synthetic:
            mock_synthetic.return_value = None
            await engine._create_model_for_type(PredictionType.IDEA_SUCCESS)
        
        assert PredictionType.IDEA_SUCCESS in engine.model_registry
        assert len(engine.model_registry[PredictionType.IDEA_SUCCESS]) == 1
        
        model_id = engine.model_registry[PredictionType.IDEA_SUCCESS][0]
        assert model_id in engine.models
        
        model = engine.models[model_id]
        assert model.prediction_type == PredictionType.IDEA_SUCCESS
    
    async def test_synthetic_data_generation(self):
        """Test synthetic training data generation"""
        engine = PredictiveAnalyticsEngine()
        model = PredictiveModel("test", PredictionType.IDEA_SUCCESS)
        
        with patch.object(model, 'train') as mock_train:
            await engine._generate_synthetic_training_data(model)
            mock_train.assert_called_once()
            
            # Check that training data was generated
            training_data = mock_train.call_args[0][0]
            assert len(training_data) == 100
            assert "features" in training_data[0]
            assert "target" in training_data[0]
    
    async def test_single_model_prediction(self):
        """Test single model prediction"""
        engine = PredictiveAnalyticsEngine()
        
        # Create and register a model
        model = PredictiveModel("test_model", PredictionType.IDEA_SUCCESS)
        engine.models["test_model"] = model
        engine.model_registry[PredictionType.IDEA_SUCCESS] = ["test_model"]
        
        features = {"market_size": 5.0, "team_experience": 3.0}
        
        with patch.object(model, 'predict') as mock_predict:
            mock_prediction = Mock(spec=Prediction)
            mock_predict.return_value = mock_prediction
            
            result = await engine.predict(
                PredictionType.IDEA_SUCCESS, 
                features, 
                PredictionHorizon.SHORT_TERM,
                use_ensemble=False
            )
            
            assert result == mock_prediction
            mock_predict.assert_called_once_with(features, PredictionHorizon.SHORT_TERM)
    
    async def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        engine = PredictiveAnalyticsEngine()
        
        # Create multiple models
        model1 = PredictiveModel("model1", PredictionType.MARKET_TREND)
        model2 = PredictiveModel("model2", PredictionType.MARKET_TREND)
        
        engine.models["model1"] = model1
        engine.models["model2"] = model2
        engine.model_registry[PredictionType.MARKET_TREND] = ["model1", "model2"]
        
        features = {"search_volume": 1000, "social_mentions": 500}
        
        # Mock predictions from individual models
        pred1 = Prediction(
            "pred1", PredictionType.MARKET_TREND, PredictionHorizon.SHORT_TERM,
            0.6, 0.8, ConfidenceLevel.HIGH, [], {"model_id": "model1"}
        )
        pred2 = Prediction(
            "pred2", PredictionType.MARKET_TREND, PredictionHorizon.SHORT_TERM,
            0.7, 0.7, ConfidenceLevel.HIGH, [], {"model_id": "model2"}
        )
        
        with patch.object(model1, 'predict', return_value=pred1), \
             patch.object(model2, 'predict', return_value=pred2):
            
            results = await engine.predict(
                PredictionType.MARKET_TREND,
                features,
                PredictionHorizon.SHORT_TERM,
                use_ensemble=True
            )
            
            assert len(results) == 3  # 2 individual + 1 ensemble
            ensemble_pred = results[-1]
            assert ensemble_pred.model_info["model_id"] == "ensemble"
    
    async def test_ensemble_combination(self):
        """Test ensemble prediction combination"""
        engine = PredictiveAnalyticsEngine()
        engine.ensemble_weights = {"model1": 0.8, "model2": 0.6}
        
        pred1 = Prediction(
            "pred1", PredictionType.IDEA_SUCCESS, PredictionHorizon.SHORT_TERM,
            0.8, 0.9, ConfidenceLevel.HIGH, [], {"model_id": "model1"}
        )
        pred2 = Prediction(
            "pred2", PredictionType.IDEA_SUCCESS, PredictionHorizon.SHORT_TERM,
            0.6, 0.7, ConfidenceLevel.MEDIUM, [], {"model_id": "model2"}
        )
        
        features = {"feature1": 0.5}
        
        ensemble_pred = await engine._combine_predictions(
            [pred1, pred2], PredictionType.IDEA_SUCCESS, features, PredictionHorizon.SHORT_TERM
        )
        
        assert isinstance(ensemble_pred, Prediction)
        assert ensemble_pred.model_info["model_id"] == "ensemble"
        # Should be weighted average closer to pred1 (higher weight and confidence)
        assert 0.6 < ensemble_pred.target_value < 0.8
    
    async def test_training_data_addition(self):
        """Test adding training data"""
        engine = PredictiveAnalyticsEngine()
        
        features = {"feature1": 0.7, "feature2": 0.5}
        target = 0.8
        
        await engine.add_training_data(PredictionType.IDEA_SUCCESS, features, target)
        
        assert len(engine.active_learning_queue) == 1
        queued_item = engine.active_learning_queue[0]
        assert queued_item["prediction_type"] == PredictionType.IDEA_SUCCESS
        assert queued_item["sample"]["features"] == features
        assert queued_item["sample"]["target"] == target
    
    async def test_active_learning_processing(self):
        """Test active learning queue processing"""
        engine = PredictiveAnalyticsEngine()
        
        # Add items to queue
        engine.active_learning_queue = [
            {
                "prediction_type": PredictionType.IDEA_SUCCESS,
                "sample": {"features": {"f1": 0.5}, "target": 0.6}
            },
            {
                "prediction_type": PredictionType.MARKET_TREND,
                "sample": {"features": {"f2": 0.7}, "target": 0.8}
            }
        ]
        
        # Mock models
        model1 = Mock()
        model1.train = AsyncMock()
        model2 = Mock()
        model2.train = AsyncMock()
        
        engine.models = {"model1": model1, "model2": model2}
        engine.model_registry = {
            PredictionType.IDEA_SUCCESS: ["model1"],
            PredictionType.MARKET_TREND: ["model2"]
        }
        
        await engine._process_active_learning_queue()
        
        assert len(engine.active_learning_queue) == 0
        model1.train.assert_called_once()
        model2.train.assert_called_once()
    
    async def test_prediction_outcome_update(self):
        """Test updating prediction outcomes"""
        engine = PredictiveAnalyticsEngine()
        
        # Create a prediction in history
        prediction = Prediction(
            "test_pred", PredictionType.IDEA_SUCCESS, PredictionHorizon.SHORT_TERM,
            0.7, 0.8, ConfidenceLevel.HIGH, [], {"model_id": "test_model"}
        )
        engine.prediction_history.append(prediction)
        
        # Mock model
        model = Mock()
        model._evaluate_model_performance = AsyncMock()
        engine.models["test_model"] = model
        
        await engine.update_prediction_outcome("test_pred", 0.85)
        
        assert prediction.actual_outcome == 0.85
        assert prediction.accuracy == 1.0  # Both > 0.5, so correct classification
        model._evaluate_model_performance.assert_called_once()
    
    async def test_feature_collection(self):
        """Test system feature collection"""
        engine = PredictiveAnalyticsEngine()
        
        # Mock intelligence
        mock_intel = Mock()
        mock_intel.get_intelligence_report.return_value = {
            "patterns_detected": 5,
            "adaptation_success_rate": 0.75
        }
        
        with patch('pipeline.core.predictive_analytics_engine.get_intelligence', return_value=mock_intel):
            await engine._collect_system_features()
        
        # Should have collected features
        assert len(engine.feature_store) > 0
        
        # Check feature content
        timestamp_key = list(engine.feature_store.keys())[0]
        features = engine.feature_store[timestamp_key]
        
        feature_names = [f.feature_name for f in features]
        assert "patterns_detected" in feature_names
        assert "adaptation_success_rate" in feature_names
    
    async def test_ensemble_weights_optimization(self):
        """Test ensemble weights optimization"""
        engine = PredictiveAnalyticsEngine()
        
        # Create models with different performance
        model1 = Mock()
        model1.performance_metrics = {"average_accuracy": 0.8}
        model1.last_training = datetime.utcnow() - timedelta(days=1)
        
        model2 = Mock()
        model2.performance_metrics = {"average_accuracy": 0.6}
        model2.last_training = datetime.utcnow() - timedelta(days=5)
        
        engine.models = {"model1": model1, "model2": model2}
        
        await engine._optimize_ensemble_weights()
        
        # Model1 should have higher weight (better performance, more recent)
        assert engine.ensemble_weights["model1"] > engine.ensemble_weights["model2"]
        assert 0 <= engine.ensemble_weights["model1"] <= 1
        assert 0 <= engine.ensemble_weights["model2"] <= 1
    
    async def test_best_model_selection(self):
        """Test best model selection"""
        engine = PredictiveAnalyticsEngine()
        
        # Create models with different performance
        model1 = PredictiveModel("model1", PredictionType.IDEA_SUCCESS)
        model1.performance_metrics = {"average_accuracy": 0.6}
        
        model2 = PredictiveModel("model2", PredictionType.IDEA_SUCCESS)
        model2.performance_metrics = {"average_accuracy": 0.8}
        
        engine.models = {"model1": model1, "model2": model2}
        engine.model_registry[PredictionType.IDEA_SUCCESS] = ["model1", "model2"]
        
        best_model_id = engine._get_best_model(PredictionType.IDEA_SUCCESS)
        
        assert best_model_id == "model2"
    
    async def test_expired_prediction_validation(self):
        """Test validation of expired predictions"""
        engine = PredictiveAnalyticsEngine()
        
        # Create expired prediction
        expired_pred = Prediction(
            "expired", PredictionType.IDEA_SUCCESS, PredictionHorizon.IMMEDIATE,
            0.7, 0.8, ConfidenceLevel.HIGH, [], {}
        )
        expired_pred.expires_at = datetime.utcnow() - timedelta(hours=1)
        
        engine.prediction_history.append(expired_pred)
        
        with patch.object(engine, 'update_prediction_outcome') as mock_update:
            await engine._validate_expired_predictions()
            mock_update.assert_called_once()
    
    def test_analytics_report(self):
        """Test analytics report generation"""
        engine = PredictiveAnalyticsEngine()
        
        # Add some mock data
        model = PredictiveModel("test_model", PredictionType.IDEA_SUCCESS)
        model.performance_metrics = {"average_accuracy": 0.75}
        model.feature_importance = {"feature1": 0.8, "feature2": 0.6}
        engine.models["test_model"] = model
        
        prediction = Prediction(
            "test_pred", PredictionType.MARKET_TREND, PredictionHorizon.SHORT_TERM,
            0.6, 0.7, ConfidenceLevel.MEDIUM, [], {}
        )
        prediction.accuracy = 0.8
        engine.prediction_history.append(prediction)
        
        engine.ensemble_weights = {"test_model": 0.75}
        
        report = engine.get_analytics_report()
        
        assert "timestamp" in report
        assert "total_models" in report
        assert "total_predictions" in report
        assert "model_performance" in report
        assert "prediction_distribution" in report
        assert "performance_trends" in report
        
        # Check model performance section
        assert "test_model" in report["model_performance"]
        model_info = report["model_performance"]["test_model"]
        assert model_info["prediction_type"] == PredictionType.IDEA_SUCCESS.value
        assert model_info["performance_metrics"]["average_accuracy"] == 0.75


@pytest.mark.asyncio
class TestGlobalAnalyticsEngine:
    """Test global analytics engine singleton"""
    
    async def test_get_analytics_engine(self):
        """Test global analytics engine creation"""
        # Reset global instance
        import pipeline.core.predictive_analytics_engine as module
        module._analytics_engine = None
        
        with patch('pipeline.core.predictive_analytics_engine.get_intelligence') as mock_intelligence:
            mock_intel = Mock()
            mock_intel.get_intelligence_report.return_value = {}
            mock_intelligence.return_value = mock_intel
            
            engine1 = await get_analytics_engine()
            engine2 = await get_analytics_engine()
        
        assert engine1 is engine2  # Should be same instance
        assert len(engine1.models) > 0
    
    async def test_convenience_functions(self):
        """Test convenience prediction functions"""
        with patch('pipeline.core.predictive_analytics_engine.get_analytics_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_prediction = Mock(spec=Prediction)
            mock_engine.predict = AsyncMock(return_value=mock_prediction)
            mock_get_engine.return_value = mock_engine
            
            # Test idea success prediction
            idea_features = {"market_size": 5, "team_experience": 3}
            result = await predict_idea_success(idea_features)
            
            assert result == mock_prediction
            mock_engine.predict.assert_called_with(
                PredictionType.IDEA_SUCCESS,
                idea_features,
                PredictionHorizon.MEDIUM_TERM
            )
            
            # Test market trend prediction
            market_features = {"search_volume": 1000, "social_mentions": 500}
            result = await predict_market_trend(market_features)
            
            mock_engine.predict.assert_called_with(
                PredictionType.MARKET_TREND,
                market_features,
                PredictionHorizon.SHORT_TERM
            )


@pytest.mark.integration
class TestPredictiveAnalyticsIntegration:
    """Integration tests for predictive analytics system"""
    
    async def test_full_prediction_lifecycle(self):
        """Test complete prediction lifecycle"""
        engine = PredictiveAnalyticsEngine()
        
        with patch('pipeline.core.predictive_analytics_engine.get_intelligence') as mock_intelligence:
            mock_intel = Mock()
            mock_intel.get_intelligence_report.return_value = {
                "patterns_detected": 3,
                "adaptation_success_rate": 0.7
            }
            mock_intelligence.return_value = mock_intel
            
            await engine.initialize()
        
        # Add training data
        features = {"market_size": 7, "team_experience": 4, "funding_amount": 500000}
        await engine.add_training_data(PredictionType.IDEA_SUCCESS, features, 0.85)
        
        # Process training data
        await engine._process_active_learning_queue()
        
        # Make prediction
        prediction_result = await engine.predict(
            PredictionType.IDEA_SUCCESS,
            features,
            PredictionHorizon.SHORT_TERM,
            use_ensemble=False
        )
        
        assert isinstance(prediction_result, Prediction)
        assert 0 <= prediction_result.target_value <= 1
        assert 0 <= prediction_result.confidence <= 1
        
        # Update with outcome
        await engine.update_prediction_outcome(prediction_result.prediction_id, 0.9)
        
        # Check accuracy was calculated
        updated_pred = next(
            p for p in engine.prediction_history 
            if p.prediction_id == prediction_result.prediction_id
        )
        assert updated_pred.accuracy is not None
    
    async def test_ensemble_prediction_integration(self):
        """Test ensemble prediction with multiple models"""
        engine = PredictiveAnalyticsEngine()
        
        with patch('pipeline.core.predictive_analytics_engine.get_intelligence') as mock_intelligence:
            mock_intel = Mock()
            mock_intel.get_intelligence_report.return_value = {}
            mock_intelligence.return_value = mock_intel
            
            await engine.initialize()
        
        # Make ensemble prediction
        features = {"search_volume": 5000, "social_mentions": 200}
        results = await engine.predict(
            PredictionType.MARKET_TREND,
            features,
            PredictionHorizon.SHORT_TERM,
            use_ensemble=True
        )
        
        # Should have multiple predictions (individual + ensemble)
        assert len(results) >= 2
        
        # Last should be ensemble
        ensemble_pred = results[-1]
        assert ensemble_pred.model_info["model_id"] == "ensemble"
        assert "component_models" in ensemble_pred.model_info
    
    async def test_continuous_learning_integration(self):
        """Test continuous learning and model improvement"""
        engine = PredictiveAnalyticsEngine()
        
        with patch('pipeline.core.predictive_analytics_engine.get_intelligence') as mock_intelligence:
            mock_intel = Mock()
            mock_intel.get_intelligence_report.return_value = {}
            mock_intelligence.return_value = mock_intel
            
            await engine.initialize()
        
        # Add multiple training samples
        training_samples = [
            ({"feature1": 0.8, "feature2": 0.6}, 0.75),
            ({"feature1": 0.3, "feature2": 0.4}, 0.25),
            ({"feature1": 0.9, "feature2": 0.8}, 0.85),
            ({"feature1": 0.2, "feature2": 0.3}, 0.15)
        ]
        
        for features, target in training_samples:
            await engine.add_training_data(PredictionType.IDEA_SUCCESS, features, target)
        
        # Process training data
        await engine._process_active_learning_queue()
        
        # Check that models were updated
        model_id = engine.model_registry[PredictionType.IDEA_SUCCESS][0]
        model = engine.models[model_id]
        
        assert len(model.training_data) >= 4  # Original synthetic + new samples
        assert model.version > 1  # Should have been retrained
    
    async def test_performance_tracking_integration(self):
        """Test performance tracking across predictions"""
        engine = PredictiveAnalyticsEngine()
        
        with patch('pipeline.core.predictive_analytics_engine.get_intelligence') as mock_intelligence:
            mock_intel = Mock()
            mock_intel.get_intelligence_report.return_value = {}
            mock_intelligence.return_value = mock_intel
            
            await engine.initialize()
        
        # Make several predictions and update outcomes
        features_list = [
            {"feature1": 0.8, "feature2": 0.6},
            {"feature1": 0.3, "feature2": 0.4},
            {"feature1": 0.7, "feature2": 0.5}
        ]
        
        outcomes = [0.8, 0.2, 0.6]
        
        predictions = []
        for features in features_list:
            pred = await engine.predict(
                PredictionType.IDEA_SUCCESS,
                features,
                PredictionHorizon.SHORT_TERM,
                use_ensemble=False
            )
            predictions.append(pred)
        
        # Update outcomes
        for pred, outcome in zip(predictions, outcomes):
            await engine.update_prediction_outcome(pred.prediction_id, outcome)
        
        # Check analytics report
        report = engine.get_analytics_report()
        
        assert report["total_predictions"] >= 3
        assert "performance_trends" in report
        
        # Should have calculated some accuracies
        recent_accuracies = [
            p.accuracy for p in engine.prediction_history 
            if p.accuracy is not None
        ]
        assert len(recent_accuracies) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])