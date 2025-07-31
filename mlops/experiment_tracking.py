"""
MLOps Experiment Tracking for AI Agent Performance
Integrates with MLflow for comprehensive model and agent versioning
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.keras
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np


class AgentExperimentTracker:
    """Advanced experiment tracking for AI agents with MLflow integration"""
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "agentic-startup-studio",
        enable_auto_logging: bool = True
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = MlflowClient(tracking_uri=tracking_uri)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                tags={
                    "project": "agentic-startup-studio",
                    "team": "ai-ml",
                    "environment": os.getenv("ENVIRONMENT", "development")
                }
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Enable auto-logging
        if enable_auto_logging:
            self._setup_auto_logging()
    
    def _setup_auto_logging(self):
        """Setup automatic logging for various ML frameworks"""
        try:
            # Enable auto-logging for supported frameworks
            mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
            mlflow.pytorch.autolog(log_models=True)
            mlflow.keras.autolog(log_models=True)
            
            # OpenAI auto-logging (if available)
            try:
                import mlflow.openai
                mlflow.openai.autolog()
            except ImportError:
                pass
                
        except Exception as e:
            print(f"Warning: Could not setup auto-logging: {e}")
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """Context manager for MLflow runs with enhanced metadata"""
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"agent_run_{timestamp}"
        
        # Merge default tags with provided tags
        default_tags = {
            "mlflow.runName": run_name,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "agent_framework": "langgraph",
            "platform": "agentic-startup-studio",
            "run_timestamp": datetime.now().isoformat(),
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            "build_version": os.getenv("BUILD_VERSION", "unknown")
        }
        
        if tags:
            default_tags.update(tags)
        
        # Start MLflow run
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=default_tags,
            nested=nested
        ) as run:
            # Log system information
            self._log_system_info()
            yield run
    
    def _log_system_info(self):
        """Log system and environment information"""
        import platform
        import psutil
        
        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
        }
        
        for key, value in system_info.items():
            mlflow.log_param(f"system_{key}", value)
    
    def log_agent_performance(
        self,
        agent_type: str,
        metrics: Dict[str, Union[int, float]],
        parameters: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        model_info: Optional[Dict[str, str]] = None
    ):
        """Log agent performance metrics and metadata"""
        
        # Log agent parameters
        if parameters:
            for key, value in parameters.items():
                mlflow.log_param(f"agent_{key}", value)
        
        # Log performance metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"agent_{metric_name}", metric_value)
        
        # Log model information
        if model_info:
            for key, value in model_info.items():
                mlflow.log_param(f"model_{key}", value)
        
        # Log artifacts
        if artifacts:
            for artifact_name, artifact_path in artifacts.items():
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_name)
        
        # Log agent type as tag
        mlflow.set_tag("agent_type", agent_type)
    
    def log_agent_workflow(
        self,
        workflow_id: str,
        stages: List[Dict[str, Any]],
        total_duration: float,
        success_rate: float,
        error_details: Optional[List[str]] = None
    ):
        """Log multi-agent workflow performance"""
        
        # Log workflow-level metrics
        mlflow.log_metric("workflow_duration_seconds", total_duration)
        mlflow.log_metric("workflow_success_rate", success_rate)
        mlflow.log_metric("workflow_stage_count", len(stages))
        
        # Log workflow parameters
        mlflow.log_param("workflow_id", workflow_id)
        mlflow.log_param("workflow_stages", [stage.get("name", "unknown") for stage in stages])
        
        # Log individual stage metrics
        for i, stage in enumerate(stages):
            stage_prefix = f"stage_{i}_{stage.get('name', 'unknown')}"
            
            if "duration" in stage:
                mlflow.log_metric(f"{stage_prefix}_duration", stage["duration"])
            if "success" in stage:
                mlflow.log_metric(f"{stage_prefix}_success", 1.0 if stage["success"] else 0.0)
            if "tokens_used" in stage:
                mlflow.log_metric(f"{stage_prefix}_tokens", stage["tokens_used"])
            if "cost_usd" in stage:
                mlflow.log_metric(f"{stage_prefix}_cost_usd", stage["cost_usd"])
        
        # Log errors if any
        if error_details:
            error_summary = "\n".join(error_details)
            
            # Create temporary file for error log
            error_file = f"/tmp/workflow_errors_{workflow_id}.txt"
            with open(error_file, "w") as f:
                f.write(error_summary)
            
            mlflow.log_artifact(error_file, "workflow_errors")
            os.remove(error_file)  # Clean up
    
    def log_model_comparison(
        self,
        models: Dict[str, Dict[str, Any]],
        comparison_metrics: List[str],
        test_data_info: Optional[Dict[str, Any]] = None
    ):
        """Log model comparison results"""
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, model_data in models.items():
            row = {"model_name": model_name}
            row.update({metric: model_data.get("metrics", {}).get(metric, 0) 
                       for metric in comparison_metrics})
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison as artifact
        comparison_file = "/tmp/model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        mlflow.log_artifact(comparison_file, "model_comparisons")
        os.remove(comparison_file)
        
        # Log best model for each metric
        for metric in comparison_metrics:
            if metric in comparison_df.columns:
                best_model = comparison_df.loc[comparison_df[metric].idxmax(), "model_name"]
                best_value = comparison_df[metric].max()
                mlflow.log_metric(f"best_{metric}_model", hash(best_model) % 1000)  # Hash for numeric logging
                mlflow.log_metric(f"best_{metric}_value", best_value)
                mlflow.log_param(f"best_{metric}_model_name", best_model)
        
        # Log test data information
        if test_data_info:
            for key, value in test_data_info.items():
                mlflow.log_param(f"test_data_{key}", value)
    
    def register_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        stage: str = "Staging",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register a model in MLflow Model Registry"""
        
        # Create model version
        try:
            # Register model if it doesn't exist
            try:
                self.client.create_registered_model(
                    model_name,
                    tags=tags or {},
                    description=description
                )
            except mlflow.exceptions.MlflowException:
                # Model already exists
                pass
            
            # Create model version
            model_version_obj = self.client.create_model_version(
                model_name,
                source=mlflow.get_artifact_uri(),
                run_id=mlflow.active_run().info.run_id,
                tags=tags or {},
                description=description
            )
            
            # Transition to specified stage
            if stage != "None":
                self.client.transition_model_version_stage(
                    model_name,
                    model_version_obj.version,
                    stage
                )
            
            return model_version_obj.version
            
        except Exception as e:
            print(f"Error registering model: {e}")
            return None
    
    def get_model_performance_history(
        self,
        model_name: str,
        metrics: List[str],
        time_range_days: int = 30
    ) -> pd.DataFrame:
        """Get historical performance data for a model"""
        
        # Get all runs for the experiment
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"tags.model_name = '{model_name}'",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1000
        )
        
        # Extract performance data
        performance_data = []
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                "duration": (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None
            }
            
            # Add metrics
            for metric in metrics:
                metric_key = f"agent_{metric}"
                if metric_key in run.data.metrics:
                    run_data[metric] = run.data.metrics[metric_key]
            
            # Add parameters
            for param_key, param_value in run.data.params.items():
                if param_key.startswith("agent_") or param_key.startswith("model_"):
                    run_data[param_key] = param_value
            
            performance_data.append(run_data)
        
        return pd.DataFrame(performance_data)
    
    def create_performance_report(
        self,
        report_name: str,
        time_range_days: int = 7,
        include_charts: bool = True
    ) -> str:
        """Create a comprehensive performance report"""
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get recent runs
        end_time = int(time.time() * 1000)
        start_time = end_time - (time_range_days * 24 * 60 * 60 * 1000)
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"attribute.start_time >= {start_time}",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1000
        )
        
        # Analyze performance trends
        report_data = {
            "report_generated": datetime.now().isoformat(),
            "time_range_days": time_range_days,
            "total_runs": len(runs),
            "successful_runs": len([r for r in runs if r.info.status == "FINISHED"]),
            "failed_runs": len([r for r in runs if r.info.status == "FAILED"]),
            "avg_duration": np.mean([
                (r.info.end_time - r.info.start_time) / 1000 
                for r in runs if r.info.end_time
            ]) if runs else 0
        }
        
        # Agent type analysis
        agent_types = {}
        for run in runs:
            agent_type = run.data.tags.get("agent_type", "unknown")
            if agent_type not in agent_types:
                agent_types[agent_type] = {"count": 0, "success_rate": 0}
            agent_types[agent_type]["count"] += 1
            if run.info.status == "FINISHED":
                agent_types[agent_type]["success_rate"] += 1
        
        # Calculate success rates
        for agent_type in agent_types:
            if agent_types[agent_type]["count"] > 0:
                agent_types[agent_type]["success_rate"] = (
                    agent_types[agent_type]["success_rate"] / 
                    agent_types[agent_type]["count"]
                )
        
        report_data["agent_performance"] = agent_types
        
        # Save report
        report_file = f"/tmp/{report_name}_performance_report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        
        # Create visualizations if requested
        if include_charts and runs:
            self._create_performance_charts(runs, report_name)
        
        # Log report as artifact
        mlflow.log_artifact(report_file, "reports")
        os.remove(report_file)
        
        return report_file
    
    def _create_performance_charts(self, runs: List, report_name: str):
        """Create performance visualization charts"""
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Performance Report: {report_name}", fontsize=16)
        
        # Run status distribution
        statuses = [run.info.status for run in runs]
        status_counts = pd.Series(statuses).value_counts()
        axes[0, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title("Run Status Distribution")
        
        # Duration trends over time
        durations = []
        timestamps = []
        for run in runs:
            if run.info.end_time and run.info.start_time:
                duration = (run.info.end_time - run.info.start_time) / 1000
                durations.append(duration)
                timestamps.append(datetime.fromtimestamp(run.info.start_time / 1000))
        
        if durations:
            axes[0, 1].plot(timestamps, durations, marker='o', alpha=0.7)
            axes[0, 1].set_title("Run Duration Over Time")
            axes[0, 1].set_xlabel("Time")
            axes[0, 1].set_ylabel("Duration (seconds)")
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Agent type performance
        agent_types = [run.data.tags.get("agent_type", "unknown") for run in runs]
        agent_counts = pd.Series(agent_types).value_counts()
        axes[1, 0].bar(agent_counts.index, agent_counts.values)
        axes[1, 0].set_title("Runs by Agent Type")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Success rate by agent type
        agent_success = {}
        for run in runs:
            agent_type = run.data.tags.get("agent_type", "unknown")
            if agent_type not in agent_success:
                agent_success[agent_type] = {"total": 0, "success": 0}
            agent_success[agent_type]["total"] += 1
            if run.info.status == "FINISHED":
                agent_success[agent_type]["success"] += 1
        
        success_rates = {
            agent: data["success"] / data["total"] if data["total"] > 0 else 0
            for agent, data in agent_success.items()
        }
        
        if success_rates:
            axes[1, 1].bar(success_rates.keys(), success_rates.values())
            axes[1, 1].set_title("Success Rate by Agent Type")
            axes[1, 1].set_ylabel("Success Rate")
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_file = f"/tmp/{report_name}_performance_charts.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log chart as artifact
        mlflow.log_artifact(chart_file, "charts")
        os.remove(chart_file)


# Decorator for automatic experiment tracking
def track_agent_experiment(
    experiment_name: str = "agentic-startup-studio",
    agent_type: Optional[str] = None,
    auto_log_params: bool = True,
    auto_log_metrics: bool = True,
    log_artifacts: bool = True
):
    """Decorator to automatically track agent experiments"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = AgentExperimentTracker(experiment_name=experiment_name)
            
            # Extract agent type from function name or parameters
            inferred_agent_type = agent_type or func.__name__.replace("_", "-")
            
            # Create run name
            run_name = f"{inferred_agent_type}_{int(time.time())}"
            
            with tracker.start_run(
                run_name=run_name,
                tags={"agent_type": inferred_agent_type, "function": func.__name__}
            ):
                # Log function parameters if enabled
                if auto_log_params:
                    # Log kwargs as parameters
                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            mlflow.log_param(f"param_{key}", value)
                
                # Execute function and measure time
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    success = True
                    error_message = None
                except Exception as e:
                    execution_time = time.time() - start_time
                    success = False
                    error_message = str(e)
                    result = None
                
                # Log execution metrics if enabled
                if auto_log_metrics:
                    mlflow.log_metric("execution_time_seconds", execution_time)
                    mlflow.log_metric("success", 1.0 if success else 0.0)
                    
                    if error_message:
                        mlflow.log_param("error_message", error_message)
                
                # Log result artifacts if enabled and successful
                if log_artifacts and success and result:
                    if isinstance(result, dict):
                        # Save result as JSON artifact
                        result_file = f"/tmp/result_{int(time.time())}.json"
                        with open(result_file, "w") as f:
                            json.dump(result, f, indent=2, default=str)
                        mlflow.log_artifact(result_file, "results")
                        os.remove(result_file)
                
                if not success:
                    raise Exception(error_message)
                
                return result
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Initialize tracker
    tracker = AgentExperimentTracker()
    
    # Example: Track a CEO agent run
    with tracker.start_run(run_name="ceo_pitch_generation_test") as run:
        # Log agent parameters
        tracker.log_agent_performance(
            agent_type="ceo",
            metrics={
                "pitch_quality_score": 0.87,
                "generation_time_seconds": 12.5,
                "token_usage": 1250,
                "cost_usd": 0.025
            },
            parameters={
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                "target_audience": "series_a_vcs"
            },
            model_info={
                "provider": "openai",
                "model_version": "gpt-4-0613"
            }
        )
    
    # Example: Track a multi-agent workflow
    with tracker.start_run(run_name="full_startup_validation_workflow") as run:
        tracker.log_agent_workflow(
            workflow_id="wf_123456",
            stages=[
                {
                    "name": "idea_generation",
                    "duration": 8.2,
                    "success": True,
                    "tokens_used": 800,
                    "cost_usd": 0.016
                },
                {
                    "name": "technical_validation",
                    "duration": 15.7,
                    "success": True,
                    "tokens_used": 1500,
                    "cost_usd": 0.030
                },
                {
                    "name": "investor_scoring",
                    "duration": 6.3,
                    "success": True,
                    "tokens_used": 600,
                    "cost_usd": 0.012
                }
            ],
            total_duration=30.2,
            success_rate=1.0
        )
    
    # Generate performance report
    report_path = tracker.create_performance_report(
        "weekly_performance",
        time_range_days=7,
        include_charts=True
    )
    
    print(f"Performance report generated: {report_path}")