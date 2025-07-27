#!/usr/bin/env python3
"""
Autonomous Backlog Management Engine
Implements the full WSJF-based backlog discovery, prioritization, and execution system.
"""

import yaml
import json
import subprocess
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class BacklogStatus(Enum):
    NEW = "NEW"
    REFINED = "REFINED" 
    READY = "READY"
    DOING = "DOING"
    PR = "PR"
    DONE = "DONE"
    BLOCKED = "BLOCKED"

class Priority(Enum):
    P0 = "P0"  # Critical
    P1 = "P1"  # High
    P2 = "P2"  # Important
    P3 = "P3"  # Enhancement
    P4 = "P4"  # Technical Debt

@dataclass
class BacklogItem:
    id: str
    title: str
    description: str
    type: str
    status: BacklogStatus
    priority: Priority
    business_value: int  # 1-13 scale
    time_criticality: int  # 1-13 scale
    risk_reduction: int  # 1-13 scale
    effort: int  # 1-13 scale
    wsjf_score: float
    acceptance_criteria: List[str]
    links: List[str]
    discovered_from: str
    tags: List[str]
    created_at: datetime.datetime
    aging_multiplier: float = 1.0

    def calculate_wsjf(self) -> float:
        """Calculate WSJF score with aging multiplier"""
        cost_of_delay = self.business_value + self.time_criticality + self.risk_reduction
        base_score = cost_of_delay / self.effort if self.effort > 0 else 0
        return base_score * self.aging_multiplier

    def apply_aging(self, max_multiplier: float = 2.0) -> None:
        """Apply aging multiplier based on item age"""
        age_days = (datetime.datetime.now() - self.created_at).days
        if age_days > 7:  # Start aging after a week
            self.aging_multiplier = min(max_multiplier, 1.0 + (age_days - 7) * 0.1)
            self.wsjf_score = self.calculate_wsjf()

class AutonomousBacklogEngine:
    """
    Main engine for autonomous backlog management following the charter:
    - Discover new work from TODOs, tests, issues
    - Score using WSJF methodology  
    - Execute in priority order with TDD
    - Generate metrics and reports
    """
    
    def __init__(self, repo_path: Path = Path("/root/repo")):
        self.repo_path = repo_path
        self.backlog_file = repo_path / "DOCS" / "backlog.yml" 
        self.status_dir = repo_path / "docs" / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.backlog: List[BacklogItem] = []
        
    def load_existing_backlog(self) -> None:
        """Load existing backlog from YAML file"""
        if not self.backlog_file.exists():
            return
            
        with open(self.backlog_file, 'r') as f:
            data = yaml.safe_load(f)
            
        for item_data in data.get('items', []):
            status = BacklogStatus(item_data.get('status', 'NEW'))
            priority = Priority(item_data.get('priority', 'P3'))
            
            item = BacklogItem(
                id=item_data['id'],
                title=item_data['title'],
                description=item_data['description'],
                type=item_data['type'],
                status=status,
                priority=priority,
                business_value=item_data['business_value'],
                time_criticality=item_data['time_criticality'], 
                risk_reduction=item_data['risk_reduction'],
                effort=item_data['effort'],
                wsjf_score=item_data['wsjf_score'],
                acceptance_criteria=item_data.get('acceptance_criteria', []),
                links=item_data.get('links', []),
                discovered_from=item_data.get('discovered_from', 'Manual'),
                tags=item_data.get('tags', []),
                created_at=datetime.datetime.now()  # Approximation
            )
            self.backlog.append(item)
    
    def discover_todos_and_fixmes(self) -> List[BacklogItem]:
        """Scan codebase for TODO/FIXME comments and convert to backlog items"""
        new_items = []
        
        # Use ripgrep to find TODO/FIXME patterns
        try:
            result = subprocess.run([
                'rg', '--type', 'py', '--type', 'js', '--type', 'md',
                '-i', '-n', 'TODO|FIXME|XXX|HACK',
                str(self.repo_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ':' in line and line.strip():
                        file_path, line_num, content = line.split(':', 2)
                        
                        # Skip documentation and template files
                        if any(skip in file_path.lower() for skip in ['docs/', 'templates/', 'adr_template']):
                            continue
                            
                        # Extract actionable TODOs
                        if any(keyword in content.upper() for keyword in ['TODO', 'FIXME', 'XXX', 'HACK']):
                            item_id = f"TODO-{abs(hash(line))}"[:10]
                            
                            item = BacklogItem(
                                id=item_id,
                                title=f"Address TODO in {Path(file_path).name}:{line_num}",
                                description=content.strip(),
                                type="Technical Debt",
                                status=BacklogStatus.NEW,
                                priority=Priority.P4,
                                business_value=2,
                                time_criticality=1,
                                risk_reduction=3,
                                effort=2,
                                wsjf_score=0,  # Will be calculated
                                acceptance_criteria=[f"Complete TODO at {file_path}:{line_num}"],
                                links=[f"{file_path}:{line_num}"],
                                discovered_from="Automated TODO scan",
                                tags=["technical-debt", "todo"],
                                created_at=datetime.datetime.now()
                            )
                            item.wsjf_score = item.calculate_wsjf()
                            new_items.append(item)
                            
        except FileNotFoundError:
            print("Warning: ripgrep not available, skipping TODO scan")
            
        return new_items
    
    def discover_failing_tests(self) -> List[BacklogItem]:
        """Discover failing or flaky tests that need attention"""
        new_items = []
        
        # This would typically run test suite and analyze results
        # For now, return empty list since tests aren't configured
        return new_items
    
    def discover_security_issues(self) -> List[BacklogItem]:
        """Scan for potential security vulnerabilities"""
        new_items = []
        
        # Check for common security anti-patterns
        security_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'exec\s*\(', "Dangerous exec() usage"),
            (r'eval\s*\(', "Dangerous eval() usage"),
        ]
        
        for pattern, description in security_patterns:
            try:
                result = subprocess.run([
                    'rg', '--type', 'py', '-n', pattern,
                    str(self.repo_path)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if ':' in line and line.strip():
                            file_path, line_num, content = line.split(':', 2)
                            
                            item_id = f"SEC-{abs(hash(line))}"[:10]
                            
                            item = BacklogItem(
                                id=item_id,
                                title=f"Security Issue: {description}",
                                description=f"Found potential security issue at {file_path}:{line_num}",
                                type="Security",
                                status=BacklogStatus.NEW,
                                priority=Priority.P0,
                                business_value=13,
                                time_criticality=13,
                                risk_reduction=13,
                                effort=3,
                                wsjf_score=0,
                                acceptance_criteria=[f"Fix security issue at {file_path}:{line_num}"],
                                links=[f"{file_path}:{line_num}"],
                                discovered_from="Security scan",
                                tags=["security", "critical"],
                                created_at=datetime.datetime.now()
                            )
                            item.wsjf_score = item.calculate_wsjf()
                            new_items.append(item)
                            
            except FileNotFoundError:
                continue
                
        return new_items
    
    def continuous_discovery(self) -> List[BacklogItem]:
        """Perform continuous discovery of new backlog items"""
        new_items = []
        
        # Discover from various sources
        new_items.extend(self.discover_todos_and_fixmes())
        new_items.extend(self.discover_failing_tests())
        new_items.extend(self.discover_security_issues())
        
        # Deduplicate based on ID
        existing_ids = {item.id for item in self.backlog}
        unique_items = [item for item in new_items if item.id not in existing_ids]
        
        return unique_items
    
    def score_and_sort_backlog(self) -> None:
        """Apply aging and re-sort backlog by WSJF score"""
        for item in self.backlog:
            item.apply_aging()
            
        # Sort by WSJF score (highest first), then by priority
        self.backlog.sort(key=lambda x: (-x.wsjf_score, x.priority.value))
    
    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get the highest priority item that's ready for execution"""
        for item in self.backlog:
            if item.status in [BacklogStatus.READY, BacklogStatus.NEW] and item.priority != Priority.P4:
                return item
        return None
    
    def generate_status_report(self) -> Dict:
        """Generate comprehensive status report"""
        now = datetime.datetime.now()
        
        status_counts = {}
        for status in BacklogStatus:
            status_counts[status.value] = len([i for i in self.backlog if i.status == status])
            
        priority_counts = {}
        for priority in Priority:
            priority_counts[priority.value] = len([i for i in self.backlog if i.priority == priority])
        
        report = {
            "timestamp": now.isoformat(),
            "total_items": len(self.backlog),
            "by_status": status_counts,
            "by_priority": priority_counts,
            "avg_wsjf_score": sum(item.wsjf_score for item in self.backlog) / len(self.backlog) if self.backlog else 0,
            "ready_items": len([i for i in self.backlog if i.status == BacklogStatus.READY]),
            "blocked_items": len([i for i in self.backlog if i.status == BacklogStatus.BLOCKED]),
            "completed_items": len([i for i in self.backlog if i.status == BacklogStatus.DONE]),
            "completion_rate": len([i for i in self.backlog if i.status == BacklogStatus.DONE]) / len(self.backlog) * 100 if self.backlog else 100
        }
        
        return report
    
    def save_status_report(self) -> Path:
        """Save status report to docs/status/"""
        report = self.generate_status_report()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # JSON report
        json_file = self.status_dir / f"status_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        return json_file
    
    def run_macro_execution_loop(self) -> None:
        """Main execution loop: discover, prioritize, execute"""
        print("🤖 Starting Autonomous Backlog Management Engine...")
        
        # Load existing backlog
        self.load_existing_backlog()
        print(f"📋 Loaded {len(self.backlog)} existing backlog items")
        
        # Continuous discovery
        new_items = self.continuous_discovery()
        if new_items:
            self.backlog.extend(new_items)
            print(f"🔍 Discovered {len(new_items)} new items")
        else:
            print("✅ No new actionable items discovered")
        
        # Score and sort
        self.score_and_sort_backlog()
        
        # Generate report
        report_file = self.save_status_report()
        print(f"📊 Status report saved to {report_file}")
        
        # Check for ready work
        next_item = self.get_next_ready_item()
        if next_item:
            print(f"🎯 Next ready item: {next_item.id} - {next_item.title}")
            print(f"   WSJF Score: {next_item.wsjf_score:.2f}")
            print(f"   Priority: {next_item.priority.value}")
        else:
            print("🎉 No ready items found - backlog execution complete!")
            
        # Display summary
        report = self.generate_status_report()
        print(f"\n📈 Backlog Summary:")
        print(f"   Total Items: {report['total_items']}")
        print(f"   Completion Rate: {report['completion_rate']:.1f}%")
        print(f"   Ready Items: {report['ready_items']}")
        print(f"   Blocked Items: {report['blocked_items']}")

if __name__ == "__main__":
    engine = AutonomousBacklogEngine()
    engine.run_macro_execution_loop()