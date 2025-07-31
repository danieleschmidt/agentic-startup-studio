"""
Model Versioning and Lifecycle Management for AI Agents
Provides comprehensive model version control, A/B testing, and deployment management
"""

import os
import json
import hashlib
import pickle
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "Development"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelType(Enum):
    """Types of models in the system"""
    LANGUAGE_MODEL = "language_model"
    EMBEDDING_MODEL = "embedding_model"
    CLASSIFICATION_MODEL = "classification_model"
    AGENT_PROMPT = "agent_prompt"
    WORKFLOW_CONFIG = "workflow_config"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    stage: ModelStage
    created_at: datetime
    created_by: str
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    
    # Performance metrics
    accuracy: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    cost_per_request: Optional[float] = None
    
    # Technical specifications
    model_size_mb: Optional[float] = None
    framework: Optional[str] = None
    runtime_version: Optional[str] = None
    hardware_requirements: Optional[Dict[str, Any]] = None
    
    # Business metrics
    business_impact_score: Optional[float] = None
    user_satisfaction_score: Optional[float] = None
    conversion_rate: Optional[float] = None
    revenue_impact: Optional[float] = None


class ModelVersionManager:
    """Advanced model version management with lifecycle control"""
    
    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5000",
        model_registry_name: str = "agentic-startup-studio-models"
    ):
        self.tracking_uri = mlflow_tracking_uri
        self.registry_name = model_registry_name
        self.client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Initialize model registry
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the model registry structure"""
        try:
            # Create main registry if it doesn't exist
            self.client.create_registered_model(
                self.registry_name,
                tags={
                    "project": "agentic-startup-studio",
                    "purpose": "ai-agent-models",
                    "created_at": datetime.now().isoformat()
                },
                description="Central registry for all AI agent models and configurations"
            )
        except mlflow.exceptions.MlflowException:
            # Registry already exists
            pass
    
    def create_model_version(
        self,
        model_name: str,
        model_type: ModelType,
        model_artifact_path: str,
        metadata: Optional[ModelMetadata] = None,
        run_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Create a new model version with comprehensive metadata"""
        
        # Generate unique model ID if not provided in metadata
        model_id = str(uuid.uuid4()) if not metadata else metadata.model_id
        
        # Create default metadata if not provided
        if not metadata:
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version="1",
                stage=ModelStage.DEVELOPMENT,
                created_at=datetime.now(),
                created_by=os.getenv("USER", "system"),
                description=description,
                tags=tags or {}
            )
        
        try:
            # Create registered model if it doesn't exist
            try:
                self.client.create_registered_model(
                    model_name,
                    tags=metadata.tags or {},
                    description=metadata.description
                )
            except mlflow.exceptions.MlflowException:
                # Model already exists
                pass
            
            # Create model version
            model_version = self.client.create_model_version(
                model_name,
                source=model_artifact_path,
                run_id=run_id,
                tags={
                    "model_id": metadata.model_id,
                    "model_type": metadata.model_type.value,
                    "created_by": metadata.created_by,
                    "framework": metadata.framework or "unknown",
                    **(metadata.tags or {})
                },
                description=metadata.description
            )
            
            # Store comprehensive metadata
            self._store_model_metadata(model_name, model_version.version, metadata)
            
            return model_version.version
            
        except Exception as e:
            raise Exception(f"Failed to create model version: {e}")
    
    def _store_model_metadata(
        self,
        model_name: str,
        version: str,
        metadata: ModelMetadata
    ):
        """Store comprehensive model metadata as MLflow artifacts"""
        
        # Convert metadata to dictionary
        metadata_dict = asdict(metadata)
        
        # Handle datetime serialization
        for key, value in metadata_dict.items():
            if isinstance(value, datetime):
                metadata_dict[key] = value.isoformat()
            elif isinstance(value, Enum):
                metadata_dict[key] = value.value
        
        # Create temporary metadata file
        metadata_file = f"/tmp/model_metadata_{model_name}_v{version}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Store as artifact in the model version
        try:
            # Note: This requires the model version to be created first
            # In practice, you might want to store this in a separate metadata store
            model_version_uri = f"models:/{model_name}/{version}"
            
            # Create a temporary run to store metadata
            with mlflow.start_run():
                mlflow.log_artifact(metadata_file, "model_metadata")
                
        except Exception as e:
            print(f"Warning: Could not store metadata as artifact: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage,
        archive_existing: bool = True
    ) -> bool:
        """Promote a model version to a higher stage"""
        
        try:
            # Archive existing production models if requested
            if archive_existing and target_stage == ModelStage.PRODUCTION:
                self._archive_existing_production_models(model_name)
            
            # Transition to target stage
            self.client.transition_model_version_stage(
                model_name,
                version,
                target_stage.value,
                archive_existing_versions=archive_existing
            )
            
            # Log promotion event
            self._log_model_event(
                model_name,
                version,
                "promotion",
                {
                    "target_stage": target_stage.value,
                    "promoted_at": datetime.now().isoformat(),
                    "promoted_by": os.getenv("USER", "system")
                }
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to promote model: {e}")
            return False
    
    def _archive_existing_production_models(self, model_name: str):
        """Archive existing production models"""
        
        try:
            # Get current production models
            production_versions = self.client.get_latest_versions(
                model_name,
                stages=["Production"]
            )
            
            # Archive each production version
            for version in production_versions:
                self.client.transition_model_version_stage(
                    model_name,
                    version.version,
                    ModelStage.ARCHIVED.value
                )
                
        except Exception as e:
            print(f"Warning: Could not archive existing production models: {e}")
    
    def _log_model_event(
        self,
        model_name: str,
        version: str,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Log model lifecycle events"""
        
        event_record = {
            "model_name": model_name,
            "model_version": version,
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store event log (in practice, you might use a dedicated event store)
        event_file = f"/tmp/model_event_{uuid.uuid4().hex[:8]}.json"
        with open(event_file, "w") as f:
            json.dump(event_record, f, indent=2)
        
        try:
            with mlflow.start_run():
                mlflow.log_artifact(event_file, "model_events")
        except Exception as e:
            print(f"Warning: Could not log model event: {e}")
        finally:
            if os.path.exists(event_file):
                os.remove(event_file)
    
    def get_model_metadata(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelMetadata]:
        """Retrieve comprehensive model metadata"""
        
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            elif stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage.value])
                if not versions:
                    return None
                model_version = versions[0]
            else:
                # Get latest version
                versions = self.client.get_latest_versions(model_name)
                if not versions:
                    return None
                model_version = versions[0]
            
            # Extract metadata from tags and model version info
            tags = model_version.tags or {}
            
            metadata = ModelMetadata(
                model_id=tags.get("model_id", str(uuid.uuid4())),
                model_name=model_name,
                model_type=ModelType(tags.get("model_type", "language_model")),
                version=model_version.version,
                stage=ModelStage(model_version.current_stage),
                created_at=datetime.fromtimestamp(model_version.creation_timestamp / 1000),
                created_by=tags.get("created_by", "unknown"),
                description=model_version.description,
                tags=tags,
                framework=tags.get("framework")
            )
            
            return metadata
            
        except Exception as e:
            print(f"Failed to retrieve model metadata: {e}")
            return None
    
    def compare_model_versions(
        self,
        model_name: str,
        version1: str,
        version2: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare two model versions across specified metrics"""
        
        try:
            # Get model versions
            mv1 = self.client.get_model_version(model_name, version1)
            mv2 = self.client.get_model_version(model_name, version2)
            
            # Get associated run metrics
            run1_metrics = self._get_run_metrics(mv1.run_id, metrics) if mv1.run_id else {}
            run2_metrics = self._get_run_metrics(mv2.run_id, metrics) if mv2.run_id else {}
            
            comparison = {
                "model_name": model_name,
                "version1": {
                    "version": version1,
                    "stage": mv1.current_stage,
                    "created_at": datetime.fromtimestamp(mv1.creation_timestamp / 1000).isoformat(),
                    "metrics": run1_metrics
                },
                "version2": {
                    "version": version2,
                    "stage": mv2.current_stage,
                    "created_at": datetime.fromtimestamp(mv2.creation_timestamp / 1000).isoformat(),
                    "metrics": run2_metrics
                },
                "comparison": {}
            }
            
            # Calculate metric differences
            for metric in metrics:
                val1 = run1_metrics.get(metric, 0)
                val2 = run2_metrics.get(metric, 0)
                
                if val1 and val2:
                    diff = val2 - val1
                    pct_change = (diff / val1) * 100 if val1 != 0 else 0
                    
                    comparison["comparison"][metric] = {
                        "version1_value": val1,
                        "version2_value": val2,
                        "difference": diff,
                        "percent_change": pct_change,
                        "better_version": version2 if val2 > val1 else version1
                    }
            
            return comparison
            
        except Exception as e:
            print(f"Failed to compare model versions: {e}")
            return {}
    
    def _get_run_metrics(self, run_id: str, metrics: List[str]) -> Dict[str, float]:
        """Get metrics from an MLflow run"""
        
        try:
            run = self.client.get_run(run_id)
            return {
                metric: run.data.metrics.get(metric, 0.0)
                for metric in metrics
                if metric in run.data.metrics
            }
        except Exception as e:
            print(f"Failed to get run metrics: {e}")
            return {}
    
    def get_model_performance_trend(
        self,
        model_name: str,
        metric: str,
        time_range_days: int = 30
    ) -> pd.DataFrame:
        """Get performance trend for a model over time"""
        
        try:
            # Get all versions of the model
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            # Filter by time range
            cutoff_time = datetime.now() - timedelta(days=time_range_days)
            recent_versions = [
                v for v in all_versions
                if datetime.fromtimestamp(v.creation_timestamp / 1000) >= cutoff_time
            ]
            
            # Extract performance data
            performance_data = []
            for version in recent_versions:
                if version.run_id:
                    run_metrics = self._get_run_metrics(version.run_id, [metric])
                    
                    if metric in run_metrics:
                        performance_data.append({
                            "version": version.version,
                            "stage": version.current_stage,
                            "created_at": datetime.fromtimestamp(version.creation_timestamp / 1000),
                            "metric_value": run_metrics[metric]
                        })
            
            return pd.DataFrame(performance_data).sort_values("created_at")
            
        except Exception as e:
            print(f"Failed to get performance trend: {e}")
            return pd.DataFrame()
    
    def rollback_model(
        self,
        model_name: str,
        target_version: str,
        rollback_reason: str
    ) -> bool:
        """Rollback to a previous model version"""
        
        try:
            # Get current production version
            current_versions = self.client.get_latest_versions(
                model_name,
                stages=["Production"]
            )
            
            current_version = current_versions[0].version if current_versions else None
            
            # Archive current production version
            if current_version:
                self.client.transition_model_version_stage(
                    model_name,
                    current_version,
                    ModelStage.ARCHIVED.value
                )
            
            # Promote target version to production
            self.client.transition_model_version_stage(
                model_name,
                target_version,
                ModelStage.PRODUCTION.value
            )
            
            # Log rollback event
            self._log_model_event(
                model_name,
                target_version,
                "rollback",
                {
                    "previous_version": current_version,
                    "rollback_reason": rollback_reason,
                    "rolled_back_at": datetime.now().isoformat(),
                    "rolled_back_by": os.getenv("USER", "system")
                }
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to rollback model: {e}")
            return False
    
    def cleanup_old_versions(
        self,
        model_name: str,
        keep_versions: int = 10,
        keep_days: int = 90
    ) -> int:
        """Clean up old model versions based on retention policy"""
        
        try:
            # Get all versions
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            # Sort by creation time (newest first)
            sorted_versions = sorted(
                all_versions,
                key=lambda v: v.creation_timestamp,
                reverse=True
            )
            
            # Determine versions to keep
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            versions_to_delete = []
            
            for i, version in enumerate(sorted_versions):
                # Always keep production and staging versions
                if version.current_stage in ["Production", "Staging"]:
                    continue
                
                # Keep recent versions up to the limit
                if i < keep_versions:
                    continue
                
                # Keep versions newer than the cutoff
                version_time = datetime.fromtimestamp(version.creation_timestamp / 1000)
                if version_time >= cutoff_time:
                    continue
                
                versions_to_delete.append(version)
            
            # Delete old versions
            deleted_count = 0
            for version in versions_to_delete:
                try:
                    self.client.delete_model_version(model_name, version.version)
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete version {version.version}: {e}")
            
            return deleted_count
            
        except Exception as e:
            print(f"Failed to cleanup old versions: {e}")
            return 0


class ABTestManager:
    """A/B testing manager for model versions"""
    
    def __init__(self, version_manager: ModelVersionManager):
        self.version_manager = version_manager
        self.active_tests: Dict[str, Dict[str, Any]] = {}
    
    def create_ab_test(
        self,
        test_name: str,
        model_name: str,
        baseline_version: str,
        candidate_version: str,
        traffic_split: float = 0.5,
        success_metrics: List[str] = None,
        test_duration_days: int = 7
    ) -> str:
        """Create a new A/B test between model versions"""
        
        test_id = str(uuid.uuid4())
        
        test_config = {
            "test_id": test_id,
            "test_name": test_name,
            "model_name": model_name,
            "baseline_version": baseline_version,
            "candidate_version": candidate_version,
            "traffic_split": traffic_split,
            "success_metrics": success_metrics or ["accuracy", "latency_ms", "user_satisfaction"],
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(days=test_duration_days),
            "status": "active",
            "results": {
                "baseline": {},
                "candidate": {},
                "statistical_significance": {},
                "winner": None
            }
        }
        
        self.active_tests[test_id] = test_config
        
        # Log test creation
        self.version_manager._log_model_event(
            model_name,
            candidate_version,
            "ab_test_started",
            {
                "test_id": test_id,
                "baseline_version": baseline_version,
                "traffic_split": traffic_split
            }
        )
        
        return test_id
    
    def record_test_result(
        self,
        test_id: str,
        version_type: str,  # "baseline" or "candidate"
        metrics: Dict[str, float]
    ):
        """Record test results for a specific version"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        
        # Update results
        if version_type not in test["results"]:
            test["results"][version_type] = {"metrics": [], "count": 0}
        
        test["results"][version_type]["metrics"].append(metrics)
        test["results"][version_type]["count"] += 1
    
    def analyze_test_results(
        self,
        test_id: str,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Analyze A/B test results and determine statistical significance"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        baseline_results = test["results"]["baseline"]
        candidate_results = test["results"]["candidate"]
        
        if not baseline_results.get("metrics") or not candidate_results.get("metrics"):
            return {"error": "Insufficient data for analysis"}
        
        analysis = {
            "test_id": test_id,
            "test_name": test["test_name"],
            "analysis_date": datetime.now().isoformat(),
            "metrics_analysis": {},
            "overall_winner": None,
            "confidence_level": confidence_level
        }
        
        # Analyze each metric
        for metric in test["success_metrics"]:
            baseline_values = [m.get(metric, 0) for m in baseline_results["metrics"]]
            candidate_values = [m.get(metric, 0) for m in candidate_results["metrics"]]
            
            if baseline_values and candidate_values:
                baseline_mean = np.mean(baseline_values)
                candidate_mean = np.mean(candidate_values)
                
                # Simple statistical test (in practice, use proper statistical tests)
                baseline_std = np.std(baseline_values)
                candidate_std = np.std(candidate_values)
                
                # Calculate effect size
                pooled_std = np.sqrt((baseline_std**2 + candidate_std**2) / 2)
                effect_size = (candidate_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
                
                # Determine if difference is significant (simplified)
                significant = abs(effect_size) > 0.2  # Cohen's d threshold
                
                analysis["metrics_analysis"][metric] = {
                    "baseline_mean": baseline_mean,
                    "candidate_mean": candidate_mean,
                    "difference": candidate_mean - baseline_mean,
                    "percent_change": ((candidate_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0,
                    "effect_size": effect_size,
                    "statistically_significant": significant,
                    "winner": "candidate" if candidate_mean > baseline_mean else "baseline"
                }
        
        # Determine overall winner
        candidate_wins = sum(1 for m in analysis["metrics_analysis"].values() if m["winner"] == "candidate")
        baseline_wins = sum(1 for m in analysis["metrics_analysis"].values() if m["winner"] == "baseline")
        
        if candidate_wins > baseline_wins:
            analysis["overall_winner"] = "candidate"
        elif baseline_wins > candidate_wins:
            analysis["overall_winner"] = "baseline"
        else:
            analysis["overall_winner"] = "tie"
        
        # Update test results
        test["results"]["analysis"] = analysis
        
        return analysis
    
    def conclude_test(
        self,
        test_id: str,
        auto_promote: bool = False
    ) -> Dict[str, Any]:
        """Conclude an A/B test and optionally promote the winner"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        
        # Analyze results
        analysis = self.analyze_test_results(test_id)
        
        # Mark test as concluded
        test["status"] = "concluded"
        test["conclusion_date"] = datetime.now()
        
        # Auto-promote winner if requested
        if auto_promote and analysis.get("overall_winner") == "candidate":
            success = self.version_manager.promote_model(
                test["model_name"],
                test["candidate_version"],
                ModelStage.PRODUCTION
            )
            
            analysis["auto_promoted"] = success
        
        # Log test conclusion
        self.version_manager._log_model_event(
            test["model_name"],
            test["candidate_version"],
            "ab_test_concluded",
            {
                "test_id": test_id,
                "winner": analysis.get("overall_winner"),
                "auto_promoted": analysis.get("auto_promoted", False)
            }
        )
        
        return analysis


# Example usage
if __name__ == "__main__":
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # Create model metadata
    metadata = ModelMetadata(
        model_id="ceo-agent-v1",
        model_name="ceo-pitch-generator",
        model_type=ModelType.LANGUAGE_MODEL,
        version="1.0.0",
        stage=ModelStage.DEVELOPMENT,
        created_at=datetime.now(),
        created_by="ai-team",
        description="CEO agent for pitch generation",
        accuracy=0.87,
        latency_ms=1200,
        cost_per_request=0.025,
        framework="openai-gpt4"
    )
    
    # Create model version
    version = version_manager.create_model_version(
        model_name="ceo-pitch-generator",
        model_type=ModelType.LANGUAGE_MODEL,
        model_artifact_path="s3://models/ceo-agent/v1",
        metadata=metadata
    )
    
    print(f"Created model version: {version}")
    
    # Initialize A/B testing
    ab_manager = ABTestManager(version_manager)
    
    # Create A/B test
    test_id = ab_manager.create_ab_test(
        test_name="CEO Agent Performance Test",
        model_name="ceo-pitch-generator",
        baseline_version="1.0.0",
        candidate_version="1.1.0",
        traffic_split=0.3,
        success_metrics=["accuracy", "latency_ms", "user_satisfaction"],
        test_duration_days=7
    )
    
    print(f"Created A/B test: {test_id}")