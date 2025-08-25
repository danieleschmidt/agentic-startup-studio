"""
AI Self-Improvement Engine - Autonomous Code Generation and Enhancement
Implements self-modifying code capabilities with safety constraints.
"""

import asyncio
import ast
import inspect
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path
import hashlib

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImprovementType(str, Enum):
    """Types of improvements the AI can make"""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CODE_REFACTORING = "code_refactoring" 
    BUG_FIX = "bug_fix"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    DOCUMENTATION = "documentation"
    TEST_COVERAGE = "test_coverage"
    SECURITY_HARDENING = "security_hardening"


class SafetyLevel(str, Enum):
    """Safety levels for code modifications"""
    SAFE = "safe"  # Documentation, comments, non-functional improvements
    MODERATE = "moderate"  # Performance optimizations, refactoring
    RISKY = "risky"  # Logic changes, new features
    DANGEROUS = "dangerous"  # Core system modifications


@dataclass
class CodeAnalysis:
    """Analysis result for a code module"""
    file_path: str
    complexity_score: float
    performance_score: float
    maintainability_score: float
    test_coverage: float
    security_score: float
    improvement_opportunities: List[Dict[str, Any]]
    safety_level: SafetyLevel


@dataclass
class ImprovementSuggestion:
    """Suggested code improvement"""
    improvement_type: ImprovementType
    file_path: str
    line_number: int
    current_code: str
    improved_code: str
    confidence_score: float
    safety_level: SafetyLevel
    estimated_impact: float
    reasoning: str


@dataclass
class ImprovementResult:
    """Result of applying an improvement"""
    suggestion: ImprovementSuggestion
    applied: bool
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percentage: float
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AISelfImprovementEngine:
    """
    Advanced AI system for autonomous code improvement and optimization.
    
    Features:
    - Static code analysis and improvement detection
    - Safe code generation with multiple safety levels
    - Performance impact measurement
    - Rollback capabilities for failed improvements
    - Learning from improvement success/failure patterns
    """
    
    def __init__(self, max_safety_level: SafetyLevel = SafetyLevel.MODERATE):
        self.logger = logging.getLogger(__name__)
        self.max_safety_level = max_safety_level
        self.improvement_history: List[ImprovementResult] = []
        self.is_improving = False
        self._improvement_lock = threading.Lock()
        self.code_backups: Dict[str, str] = {}
        
        # Learning patterns from successful improvements
        self.success_patterns: Dict[str, float] = {}
        self.failure_patterns: Dict[str, float] = {}
        
    async def analyze_codebase(
        self, 
        target_directory: str = "pipeline/core"
    ) -> List[CodeAnalysis]:
        """
        Analyze codebase for improvement opportunities.
        
        Args:
            target_directory: Directory to analyze
            
        Returns:
            List of CodeAnalysis results
        """
        analyses = []
        target_path = Path(target_directory)
        
        if not target_path.exists():
            self.logger.warning(f"Target directory {target_directory} does not exist")
            return analyses
            
        python_files = list(target_path.rglob("*.py"))
        
        for file_path in python_files:
            if file_path.name.startswith("test_") or "/tests/" in str(file_path):
                continue  # Skip test files for safety
                
            try:
                analysis = await self._analyze_file(str(file_path))
                if analysis:
                    analyses.append(analysis)
            except Exception as e:
                self.logger.error(f"Failed to analyze {file_path}: {e}")
                
        self.logger.info(f"Analyzed {len(analyses)} files in {target_directory}")
        return analyses
    
    async def generate_improvements(
        self, 
        analyses: List[CodeAnalysis],
        max_suggestions: int = 10
    ) -> List[ImprovementSuggestion]:
        """
        Generate improvement suggestions based on code analysis.
        
        Args:
            analyses: List of code analysis results
            max_suggestions: Maximum number of suggestions to generate
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        for analysis in analyses:
            file_suggestions = await self._generate_file_improvements(analysis)
            suggestions.extend(file_suggestions)
            
        # Sort by impact and confidence, filter by safety level
        suggestions = [
            s for s in suggestions 
            if self._is_safety_level_allowed(s.safety_level)
        ]
        
        suggestions.sort(
            key=lambda s: s.estimated_impact * s.confidence_score, 
            reverse=True
        )
        
        return suggestions[:max_suggestions]
    
    async def apply_improvements(
        self, 
        suggestions: List[ImprovementSuggestion],
        test_before_apply: bool = True
    ) -> List[ImprovementResult]:
        """
        Apply improvement suggestions to the codebase.
        
        Args:
            suggestions: List of improvements to apply
            test_before_apply: Whether to test before applying changes
            
        Returns:
            List of improvement results
        """
        with self._improvement_lock:
            if self.is_improving:
                raise RuntimeError("Improvement process already in progress")
            self.is_improving = True
            
        results = []
        
        try:
            for suggestion in suggestions:
                if not self._is_safety_level_allowed(suggestion.safety_level):
                    continue
                    
                result = await self._apply_single_improvement(
                    suggestion, test_before_apply
                )
                results.append(result)
                
                # Learn from the result
                await self._learn_from_result(result)
                
                # Stop if we hit a failure
                if not result.applied:
                    self.logger.warning(
                        f"Stopping improvement process due to failure in {suggestion.file_path}"
                    )
                    break
                    
        finally:
            self.is_improving = False
            
        return results
    
    async def _analyze_file(self, file_path: str) -> Optional[CodeAnalysis]:
        """Analyze a single Python file for improvement opportunities"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST for static analysis
            tree = ast.parse(content)
            
            # Calculate various metrics
            complexity_score = self._calculate_complexity(tree, content)
            performance_score = self._calculate_performance_score(tree, content)
            maintainability_score = self._calculate_maintainability(tree, content)
            test_coverage = self._estimate_test_coverage(file_path)
            security_score = self._calculate_security_score(tree, content)
            
            # Identify improvement opportunities
            opportunities = self._identify_opportunities(tree, content)
            
            # Determine overall safety level
            safety_level = self._determine_file_safety_level(file_path, opportunities)
            
            return CodeAnalysis(
                file_path=file_path,
                complexity_score=complexity_score,
                performance_score=performance_score,
                maintainability_score=maintainability_score,
                test_coverage=test_coverage,
                security_score=security_score,
                improvement_opportunities=opportunities,
                safety_level=safety_level
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _calculate_complexity(self, tree: ast.AST, content: str) -> float:
        """Calculate code complexity score (0.0-1.0, lower is better)"""
        
        # Count various complexity indicators
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        nested_loops = self._count_nested_structures(tree)
        line_count = len(content.split('\n'))
        
        # Simple complexity calculation
        base_complexity = (function_count * 0.1 + class_count * 0.2 + nested_loops * 0.3) / line_count
        
        return min(base_complexity, 1.0)
    
    def _calculate_performance_score(self, tree: ast.AST, content: str) -> float:
        """Calculate performance score (0.0-1.0, higher is better)"""
        
        # Look for performance anti-patterns
        performance_issues = 0
        
        # Check for inefficient patterns
        if "for " in content and " in range(len(" in content:
            performance_issues += 1  # Use enumerate instead
            
        if ".append(" in content and "for " in content:
            performance_issues += 1  # Consider list comprehension
            
        if "import *" in content:
            performance_issues += 1  # Avoid star imports
            
        # More sophisticated analysis would go here
        
        base_score = 0.8
        penalty = min(performance_issues * 0.1, 0.5)
        
        return max(base_score - penalty, 0.0)
    
    def _calculate_maintainability(self, tree: ast.AST, content: str) -> float:
        """Calculate maintainability score (0.0-1.0, higher is better)"""
        
        lines = content.split('\n')
        
        # Check for good practices
        has_docstrings = '"""' in content or "'''" in content
        has_type_hints = "->" in content or ": " in content
        has_comments = any(line.strip().startswith('#') for line in lines)
        
        # Calculate function length average
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if functions:
            avg_function_length = sum(
                n.end_lineno - n.lineno for n in functions if hasattr(n, 'end_lineno')
            ) / len(functions)
            length_penalty = max((avg_function_length - 20) * 0.01, 0)
        else:
            length_penalty = 0
            
        base_score = 0.5
        if has_docstrings:
            base_score += 0.2
        if has_type_hints:
            base_score += 0.2
        if has_comments:
            base_score += 0.1
            
        return max(base_score - length_penalty, 0.0)
    
    def _calculate_security_score(self, tree: ast.AST, content: str) -> float:
        """Calculate security score (0.0-1.0, higher is better)"""
        
        security_issues = 0
        
        # Check for common security anti-patterns
        if "eval(" in content:
            security_issues += 2
        if "exec(" in content:
            security_issues += 2
        if "os.system(" in content:
            security_issues += 1
        if "subprocess.call(" in content and "shell=True" in content:
            security_issues += 1
        if "pickle.loads(" in content:
            security_issues += 1
            
        base_score = 0.9
        penalty = min(security_issues * 0.15, 0.8)
        
        return max(base_score - penalty, 0.1)
    
    def _estimate_test_coverage(self, file_path: str) -> float:
        """Estimate test coverage for the file"""
        
        # Look for corresponding test file
        path = Path(file_path)
        test_file = path.parent / f"test_{path.name}"
        
        if test_file.exists():
            return 0.8  # Assume good coverage if test file exists
        
        # Check in tests directory
        tests_dir = Path("tests")
        if tests_dir.exists():
            potential_test = tests_dir / path.parent.name / f"test_{path.name}"
            if potential_test.exists():
                return 0.8
                
        return 0.2  # Low coverage if no test file found
    
    def _count_nested_structures(self, tree: ast.AST) -> int:
        """Count nested control structures"""
        nested_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.If)):
                # Count nested structures within this node
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.For, ast.While, ast.If)):
                        nested_count += 1
                        
        return nested_count
    
    def _identify_opportunities(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities"""
        opportunities = []
        
        # Performance opportunities
        if " in range(len(" in content:
            opportunities.append({
                "type": ImprovementType.PERFORMANCE_OPTIMIZATION,
                "description": "Replace range(len()) pattern with enumerate",
                "impact": 0.3,
                "safety": SafetyLevel.SAFE
            })
            
        # Documentation opportunities
        if '"""' not in content and "'''" not in content:
            opportunities.append({
                "type": ImprovementType.DOCUMENTATION,
                "description": "Add module docstring",
                "impact": 0.2,
                "safety": SafetyLevel.SAFE
            })
            
        # Type hint opportunities
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        untyped_functions = [
            f for f in functions 
            if not f.returns and not any(
                arg.annotation for arg in f.args.args
            )
        ]
        
        if untyped_functions:
            opportunities.append({
                "type": ImprovementType.FEATURE_ENHANCEMENT,
                "description": "Add type hints to functions",
                "impact": 0.25,
                "safety": SafetyLevel.SAFE
            })
            
        return opportunities
    
    def _determine_file_safety_level(
        self, 
        file_path: str, 
        opportunities: List[Dict[str, Any]]
    ) -> SafetyLevel:
        """Determine the maximum safe modification level for a file"""
        
        # Core system files are more dangerous to modify
        if "core/" in file_path and any(
            critical in file_path.lower() 
            for critical in ["executor", "engine", "orchestrator"]
        ):
            return SafetyLevel.RISKY
            
        # Infrastructure files are moderately risky
        if "infrastructure/" in file_path:
            return SafetyLevel.MODERATE
            
        # Config and utility files are generally safe
        if any(
            safe_type in file_path.lower() 
            for safe_type in ["config", "utils", "helpers", "models"]
        ):
            return SafetyLevel.SAFE
            
        return SafetyLevel.MODERATE
    
    async def _generate_file_improvements(
        self, 
        analysis: CodeAnalysis
    ) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions for a single file"""
        
        suggestions = []
        
        for opportunity in analysis.improvement_opportunities:
            if not self._is_safety_level_allowed(opportunity["safety"]):
                continue
                
            # Generate specific improvement suggestion
            suggestion = await self._create_improvement_suggestion(
                analysis, opportunity
            )
            
            if suggestion:
                suggestions.append(suggestion)
                
        return suggestions
    
    async def _create_improvement_suggestion(
        self, 
        analysis: CodeAnalysis, 
        opportunity: Dict[str, Any]
    ) -> Optional[ImprovementSuggestion]:
        """Create a specific improvement suggestion"""
        
        try:
            with open(analysis.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Generate improvement based on type
            if opportunity["type"] == ImprovementType.DOCUMENTATION:
                return await self._generate_documentation_improvement(
                    analysis.file_path, content, opportunity
                )
            elif opportunity["type"] == ImprovementType.PERFORMANCE_OPTIMIZATION:
                return await self._generate_performance_improvement(
                    analysis.file_path, content, opportunity
                )
            elif opportunity["type"] == ImprovementType.FEATURE_ENHANCEMENT:
                return await self._generate_feature_improvement(
                    analysis.file_path, content, opportunity
                )
                
        except Exception as e:
            self.logger.error(f"Failed to create improvement suggestion: {e}")
            return None
            
        return None
    
    async def _generate_documentation_improvement(
        self, 
        file_path: str, 
        content: str, 
        opportunity: Dict[str, Any]
    ) -> Optional[ImprovementSuggestion]:
        """Generate documentation improvement suggestion"""
        
        lines = content.split('\n')
        
        # Add module docstring if missing
        if not content.lstrip().startswith('"""') and not content.lstrip().startswith("'''"):
            
            # Generate module docstring based on file analysis
            module_name = Path(file_path).stem.replace('_', ' ').title()
            docstring = f'"""\n{module_name} - Enhanced with AI improvements.\n\nThis module provides advanced functionality with optimized performance\nand comprehensive error handling.\n"""'
            
            # Find insertion point (after imports, before first class/function)
            insertion_line = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                    insertion_line = i
                    break
                    
            improved_lines = lines[:]
            improved_lines.insert(insertion_line, docstring)
            improved_lines.insert(insertion_line + 1, '')
            
            return ImprovementSuggestion(
                improvement_type=ImprovementType.DOCUMENTATION,
                file_path=file_path,
                line_number=insertion_line,
                current_code='\n'.join(lines[:insertion_line+3]),
                improved_code='\n'.join(improved_lines[insertion_line:insertion_line+3]),
                confidence_score=0.9,
                safety_level=SafetyLevel.SAFE,
                estimated_impact=0.2,
                reasoning="Added comprehensive module docstring for better documentation"
            )
            
        return None
    
    async def _generate_performance_improvement(
        self, 
        file_path: str, 
        content: str, 
        opportunity: Dict[str, Any]
    ) -> Optional[ImprovementSuggestion]:
        """Generate performance improvement suggestion"""
        
        lines = content.split('\n')
        
        # Look for range(len()) pattern
        for i, line in enumerate(lines):
            if " in range(len(" in line and "for " in line:
                # Extract the loop variable and iterable
                match = re.search(r'for\s+(\w+)\s+in\s+range\(len\((\w+)\)\)', line)
                if match:
                    loop_var, iterable = match.groups()
                    
                    # Generate improved version with enumerate
                    improved_line = line.replace(
                        f"for {loop_var} in range(len({iterable}))",
                        f"for {loop_var}, item in enumerate({iterable})"
                    )
                    
                    return ImprovementSuggestion(
                        improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
                        file_path=file_path,
                        line_number=i + 1,
                        current_code=line.strip(),
                        improved_code=improved_line.strip(),
                        confidence_score=0.85,
                        safety_level=SafetyLevel.SAFE,
                        estimated_impact=0.3,
                        reasoning="Replaced range(len()) with enumerate for better performance and readability"
                    )
                    
        return None
    
    async def _generate_feature_improvement(
        self, 
        file_path: str, 
        content: str, 
        opportunity: Dict[str, Any]
    ) -> Optional[ImprovementSuggestion]:
        """Generate feature enhancement suggestion"""
        
        # For now, just return None - feature improvements are more complex
        # and require deeper understanding of the code context
        return None
    
    def _is_safety_level_allowed(self, safety_level: SafetyLevel) -> bool:
        """Check if a safety level is allowed based on configuration"""
        
        safety_order = [SafetyLevel.SAFE, SafetyLevel.MODERATE, SafetyLevel.RISKY, SafetyLevel.DANGEROUS]
        
        return safety_order.index(safety_level) <= safety_order.index(self.max_safety_level)
    
    async def _apply_single_improvement(
        self, 
        suggestion: ImprovementSuggestion,
        test_before_apply: bool
    ) -> ImprovementResult:
        """Apply a single improvement suggestion"""
        
        start_time = time.time()
        
        try:
            # Backup original file
            with open(suggestion.file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                
            self.code_backups[suggestion.file_path] = original_content
            
            # Collect before metrics
            before_metrics = await self._collect_file_metrics(suggestion.file_path)
            
            # Apply the improvement
            lines = original_content.split('\n')
            if suggestion.line_number > 0:
                lines[suggestion.line_number - 1] = suggestion.improved_code
            else:
                # Handle insertions at the beginning
                lines.insert(0, suggestion.improved_code)
                
            improved_content = '\n'.join(lines)
            
            # Test the change if requested
            if test_before_apply:
                test_result = await self._test_code_change(suggestion.file_path, improved_content)
                if not test_result:
                    return ImprovementResult(
                        suggestion=suggestion,
                        applied=False,
                        before_metrics=before_metrics,
                        after_metrics=before_metrics,
                        improvement_percentage=0.0,
                        execution_time=time.time() - start_time
                    )
                    
            # Apply the change
            with open(suggestion.file_path, 'w', encoding='utf-8') as f:
                f.write(improved_content)
                
            # Collect after metrics
            after_metrics = await self._collect_file_metrics(suggestion.file_path)
            
            # Calculate improvement
            improvement_percentage = self._calculate_file_improvement(before_metrics, after_metrics)
            
            self.logger.info(
                f"Applied {suggestion.improvement_type.value} to {suggestion.file_path}: "
                f"{improvement_percentage:.1f}% improvement"
            )
            
            return ImprovementResult(
                suggestion=suggestion,
                applied=True,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement_percentage,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to apply improvement to {suggestion.file_path}: {e}")
            
            # Restore backup if exists
            if suggestion.file_path in self.code_backups:
                with open(suggestion.file_path, 'w', encoding='utf-8') as f:
                    f.write(self.code_backups[suggestion.file_path])
                    
            return ImprovementResult(
                suggestion=suggestion,
                applied=False,
                before_metrics={},
                after_metrics={},
                improvement_percentage=0.0,
                execution_time=time.time() - start_time
            )
    
    async def _collect_file_metrics(self, file_path: str) -> Dict[str, float]:
        """Collect metrics for a file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            return {
                "line_count": len(content.split('\n')),
                "function_count": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                "class_count": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                "complexity_score": self._calculate_complexity(tree, content),
                "maintainability_score": self._calculate_maintainability(tree, content)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for {file_path}: {e}")
            return {}
    
    def _calculate_file_improvement(
        self, 
        before: Dict[str, float], 
        after: Dict[str, float]
    ) -> float:
        """Calculate improvement percentage for file metrics"""
        
        if not before or not after:
            return 0.0
            
        improvements = []
        
        # Complexity improvement (lower is better)
        if "complexity_score" in before and "complexity_score" in after:
            complexity_improvement = (before["complexity_score"] - after["complexity_score"]) / max(before["complexity_score"], 0.001) * 100
            improvements.append(complexity_improvement)
            
        # Maintainability improvement (higher is better)
        if "maintainability_score" in before and "maintainability_score" in after:
            maintainability_improvement = (after["maintainability_score"] - before["maintainability_score"]) / max(before["maintainability_score"], 0.001) * 100
            improvements.append(maintainability_improvement)
            
        if improvements:
            return sum(improvements) / len(improvements)
        else:
            return 0.0
    
    async def _test_code_change(self, file_path: str, new_content: str) -> bool:
        """Test if a code change is syntactically valid"""
        
        try:
            # Basic syntax check
            ast.parse(new_content)
            
            # TODO: Add more sophisticated testing
            # - Import checks
            # - Basic runtime validation
            # - Unit test execution
            
            return True
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error in proposed change to {file_path}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error testing change to {file_path}: {e}")
            return False
    
    async def _learn_from_result(self, result: ImprovementResult):
        """Learn from the result of applying an improvement"""
        
        pattern_key = f"{result.suggestion.improvement_type.value}:{result.suggestion.safety_level.value}"
        
        if result.applied and result.improvement_percentage > 0:
            # Success pattern
            self.success_patterns[pattern_key] = self.success_patterns.get(pattern_key, 0.5) + 0.1
            self.success_patterns[pattern_key] = min(self.success_patterns[pattern_key], 0.95)
        else:
            # Failure pattern
            self.failure_patterns[pattern_key] = self.failure_patterns.get(pattern_key, 0.5) + 0.1
            self.failure_patterns[pattern_key] = min(self.failure_patterns[pattern_key], 0.95)
        
        self.logger.debug(
            f"Updated learning patterns: {pattern_key} -> "
            f"success: {self.success_patterns.get(pattern_key, 0.5):.2f}, "
            f"failure: {self.failure_patterns.get(pattern_key, 0.5):.2f}"
        )
    
    def rollback_improvements(self, file_paths: Optional[List[str]] = None):
        """Rollback improvements to specified files or all files"""
        
        target_files = file_paths or list(self.code_backups.keys())
        
        rolled_back = 0
        for file_path in target_files:
            if file_path in self.code_backups:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.code_backups[file_path])
                    rolled_back += 1
                    self.logger.info(f"Rolled back changes to {file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to rollback {file_path}: {e}")
                    
        self.logger.info(f"Rolled back {rolled_back} files")
        
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvements made"""
        
        if not self.improvement_history:
            return {"status": "no_improvements_performed"}
            
        applied_improvements = [r for r in self.improvement_history if r.applied]
        
        return {
            "total_improvements_attempted": len(self.improvement_history),
            "successful_improvements": len(applied_improvements),
            "success_rate": len(applied_improvements) / len(self.improvement_history),
            "average_improvement_percentage": sum(r.improvement_percentage for r in applied_improvements) / max(len(applied_improvements), 1),
            "total_execution_time": sum(r.execution_time for r in self.improvement_history),
            "improvement_types": {
                t.value: len([r for r in applied_improvements if r.suggestion.improvement_type == t])
                for t in ImprovementType
            },
            "files_modified": len(set(r.suggestion.file_path for r in applied_improvements)),
            "is_currently_improving": self.is_improving,
            "max_safety_level": self.max_safety_level.value
        }


# Global improvement engine instance
_improvement_engine: Optional[AISelfImprovementEngine] = None
_engine_lock = threading.Lock()


def get_ai_improvement_engine(max_safety_level: SafetyLevel = SafetyLevel.MODERATE) -> AISelfImprovementEngine:
    """Get global AI improvement engine instance"""
    global _improvement_engine
    
    if _improvement_engine is None:
        with _engine_lock:
            if _improvement_engine is None:
                _improvement_engine = AISelfImprovementEngine(max_safety_level)
                
    return _improvement_engine


async def autonomous_code_improvement(
    target_directory: str = "pipeline/core",
    max_improvements: int = 5,
    max_safety_level: SafetyLevel = SafetyLevel.SAFE
) -> List[ImprovementResult]:
    """
    Perform autonomous code improvement on the specified directory.
    
    Args:
        target_directory: Directory to improve
        max_improvements: Maximum number of improvements to apply
        max_safety_level: Maximum safety level for modifications
        
    Returns:
        List of improvement results
    """
    engine = get_ai_improvement_engine(max_safety_level)
    
    # Analyze codebase
    analyses = await engine.analyze_codebase(target_directory)
    
    # Generate improvement suggestions
    suggestions = await engine.generate_improvements(analyses, max_improvements)
    
    # Apply improvements
    results = await engine.apply_improvements(suggestions, test_before_apply=True)
    
    return results