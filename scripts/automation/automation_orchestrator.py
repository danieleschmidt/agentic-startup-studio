#!/usr/bin/env python3
"""
Automation Orchestrator
Central coordination system for all automation scripts and tasks.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import schedule
import time
import logging


class AutomationOrchestrator:
    """Central orchestrator for all automation tasks."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.timestamp = datetime.now().isoformat()
        self.automation_log = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for automation activities."""
        log_dir = self.repo_path / ".github" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "automation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_daily_automation(self) -> Dict[str, Any]:
        """Run daily automation tasks."""
        self.logger.info("🚀 Starting daily automation cycle...")
        
        results = {
            "timestamp": self.timestamp,
            "cycle_type": "daily",
            "tasks_executed": [],
            "tasks_failed": [],
            "summary": {}
        }
        
        # Define daily tasks
        daily_tasks = [
            ("metrics_collection", self._run_metrics_collection),
            ("dependency_check", self._run_dependency_check),
            ("code_quality_scan", self._run_code_quality_scan),
            ("security_scan", self._run_security_scan)
        ]
        
        # Execute tasks
        for task_name, task_function in daily_tasks:
            try:
                self.logger.info(f"Executing task: {task_name}")
                task_result = task_function()
                results["tasks_executed"].append({
                    "name": task_name,
                    "status": "success",
                    "result": task_result,
                    "timestamp": datetime.now().isoformat()
                })
                self.logger.info(f"✅ Task completed: {task_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Task failed: {task_name} - {str(e)}")
                results["tasks_failed"].append({
                    "name": task_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Generate summary
        results["summary"] = {
            "total_tasks": len(daily_tasks),
            "successful_tasks": len(results["tasks_executed"]),
            "failed_tasks": len(results["tasks_failed"]),
            "success_rate": len(results["tasks_executed"]) / len(daily_tasks) * 100
        }
        
        self.logger.info(f"Daily automation completed: {results['summary']['success_rate']:.1f}% success rate")
        return results
    
    def run_weekly_automation(self) -> Dict[str, Any]:
        """Run weekly automation tasks."""
        self.logger.info("🗓️ Starting weekly automation cycle...")
        
        results = {
            "timestamp": self.timestamp,
            "cycle_type": "weekly",
            "tasks_executed": [],
            "tasks_failed": [],
            "summary": {}
        }
        
        # Define weekly tasks
        weekly_tasks = [
            ("repository_maintenance", self._run_repository_maintenance),
            ("comprehensive_reporting", self._run_comprehensive_reporting),
            ("dependency_updates", self._run_dependency_updates),
            ("performance_analysis", self._run_performance_analysis)
        ]
        
        # Execute tasks
        for task_name, task_function in weekly_tasks:
            try:
                self.logger.info(f"Executing weekly task: {task_name}")
                task_result = task_function()
                results["tasks_executed"].append({
                    "name": task_name,
                    "status": "success",
                    "result": task_result,
                    "timestamp": datetime.now().isoformat()
                })
                self.logger.info(f"✅ Weekly task completed: {task_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Weekly task failed: {task_name} - {str(e)}")
                results["tasks_failed"].append({
                    "name": task_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Generate summary
        results["summary"] = {
            "total_tasks": len(weekly_tasks),
            "successful_tasks": len(results["tasks_executed"]),
            "failed_tasks": len(results["tasks_failed"]),
            "success_rate": len(results["tasks_executed"]) / len(weekly_tasks) * 100
        }
        
        self.logger.info(f"Weekly automation completed: {results['summary']['success_rate']:.1f}% success rate")
        return results
    
    def _run_metrics_collection(self) -> Dict[str, Any]:
        """Run metrics collection."""
        script_path = self.repo_path / "scripts" / "automation" / "metrics_collector.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--output", ".github/latest-metrics.json"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def _run_dependency_check(self) -> Dict[str, Any]:
        """Run dependency check."""
        script_path = self.repo_path / "scripts" / "automation" / "dependency_updater.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--check", "--output", ".github/dependency-check.json"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def _run_code_quality_scan(self) -> Dict[str, Any]:
        """Run code quality scan."""
        script_path = self.repo_path / "scripts" / "automation" / "code_quality_monitor.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--output", ".github/code-quality-report.json"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def _run_security_scan(self) -> Dict[str, Any]:
        """Run basic security scan (part of repository maintenance)."""
        # Security scanning is integrated into repository maintenance
        return {"status": "integrated_with_maintenance", "message": "Security scan runs as part of repository maintenance"}
    
    def _run_repository_maintenance(self) -> Dict[str, Any]:
        """Run repository maintenance."""
        script_path = self.repo_path / "scripts" / "automation" / "repository_maintenance.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--output", ".github/maintenance-report.json"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def _run_comprehensive_reporting(self) -> Dict[str, Any]:
        """Run comprehensive reporting."""
        script_path = self.repo_path / "scripts" / "automation" / "automated_reporting.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--type", "all", "--output-dir", ".github/reports"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def _run_dependency_updates(self) -> Dict[str, Any]:
        """Run dependency updates (security only for safety)."""
        script_path = self.repo_path / "scripts" / "automation" / "dependency_updater.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path), "--update", "all", "--update-type", "security"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def _run_performance_analysis(self) -> Dict[str, Any]:
        """Run performance analysis."""
        # This would integrate with existing performance monitoring
        return {"status": "placeholder", "message": "Performance analysis to be integrated with existing monitoring"}
    
    def setup_scheduled_tasks(self):
        """Setup scheduled automation tasks."""
        self.logger.info("📅 Setting up scheduled automation tasks...")
        
        # Daily tasks at 2 AM
        schedule.every().day.at("02:00").do(self._scheduled_daily_run)
        
        # Weekly tasks on Sunday at 3 AM
        schedule.every().sunday.at("03:00").do(self._scheduled_weekly_run)
        
        # Metrics collection every 6 hours
        schedule.every(6).hours.do(self._scheduled_metrics_collection)
        
        self.logger.info("✅ Scheduled tasks configured")
    
    def _scheduled_daily_run(self):
        """Scheduled daily automation run."""
        try:
            results = self.run_daily_automation()
            self._save_automation_results(results, "daily")
        except Exception as e:
            self.logger.error(f"Scheduled daily run failed: {str(e)}")
    
    def _scheduled_weekly_run(self):
        """Scheduled weekly automation run."""
        try:
            results = self.run_weekly_automation()
            self._save_automation_results(results, "weekly")
        except Exception as e:
            self.logger.error(f"Scheduled weekly run failed: {str(e)}")
    
    def _scheduled_metrics_collection(self):
        """Scheduled metrics collection."""
        try:
            result = self._run_metrics_collection()
            self.logger.info("📊 Scheduled metrics collection completed")
        except Exception as e:
            self.logger.error(f"Scheduled metrics collection failed: {str(e)}")
    
    def _save_automation_results(self, results: Dict[str, Any], cycle_type: str):
        """Save automation results to file."""
        output_dir = self.repo_path / ".github" / "automation-results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{cycle_type}_automation_{timestamp}.json"
        
        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 Automation results saved: {filename}")
    
    def run_scheduler(self):
        """Run the automation scheduler."""
        self.logger.info("🤖 Starting automation scheduler...")
        self.setup_scheduled_tasks()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("⏹️ Automation scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Scheduler error: {str(e)}")
    
    def run_on_demand(self, task_type: str = "daily") -> Dict[str, Any]:
        """Run automation on demand."""
        if task_type == "daily":
            return self.run_daily_automation()
        elif task_type == "weekly":
            return self.run_weekly_automation()
        elif task_type == "metrics":
            return {"metrics": self._run_metrics_collection()}
        elif task_type == "maintenance":
            return {"maintenance": self._run_repository_maintenance()}
        elif task_type == "quality":
            return {"quality": self._run_code_quality_scan()}
        elif task_type == "reporting":
            return {"reporting": self._run_comprehensive_reporting()}
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_automation_status(self) -> Dict[str, Any]:
        """Get current automation status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "scheduler_running": False,  # Would need to track this properly
            "last_daily_run": "unknown",
            "last_weekly_run": "unknown",
            "upcoming_tasks": []
        }
        
        # Check for recent automation results
        results_dir = self.repo_path / ".github" / "automation-results"
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json"))
            if result_files:
                latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
                status["last_run_file"] = str(latest_file)
                status["last_run_time"] = datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
        
        # Get scheduled tasks info
        jobs = schedule.jobs
        status["scheduled_jobs_count"] = len(jobs)
        
        for job in jobs:
            status["upcoming_tasks"].append({
                "job": str(job.job_func),
                "next_run": str(job.next_run) if job.next_run else "Not scheduled"
            })
        
        return status


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automation Orchestrator")
    parser.add_argument("--mode", choices=["run", "schedule", "status"], default="run",
                       help="Operation mode")
    parser.add_argument("--task", choices=["daily", "weekly", "metrics", "maintenance", "quality", "reporting"],
                       default="daily", help="Task type to run (for run mode)")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    
    args = parser.parse_args()
    
    orchestrator = AutomationOrchestrator(args.repo_path)
    
    if args.mode == "schedule":
        # Run as daemon scheduler
        orchestrator.run_scheduler()
    
    elif args.mode == "status":
        # Show automation status
        status = orchestrator.get_automation_status()
        print(json.dumps(status, indent=2))
    
    else:
        # Run on demand
        try:
            results = orchestrator.run_on_demand(args.task)
            
            # Save results
            orchestrator._save_automation_results(results, args.task)
            
            # Print summary
            if "summary" in results:
                summary = results["summary"]
                print(f"\n🤖 Automation Complete: {args.task}")
                print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
                print(f"Tasks Executed: {summary.get('successful_tasks', 0)}")
                print(f"Tasks Failed: {summary.get('failed_tasks', 0)}")
            else:
                print(f"\n🤖 Task Complete: {args.task}")
                
        except Exception as e:
            orchestrator.logger.error(f"On-demand automation failed: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main()