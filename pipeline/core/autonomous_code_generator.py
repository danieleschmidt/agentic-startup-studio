"""
Autonomous Code Generator - Generation 3 Enhancement
Advanced autonomous code generation with self-improvement capabilities
"""

import asyncio
import json
import logging
import time
import ast
import textwrap
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from pydantic import BaseModel, Field
from opentelemetry import trace

from ..config.settings import get_settings
from ..telemetry import get_tracer

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class CodeLanguage(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    YAML = "yaml"
    JSON = "json"


class CodeType(str, Enum):
    """Types of code that can be generated"""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    API_ENDPOINT = "api_endpoint"
    DATABASE_SCHEMA = "database_schema"
    CONFIGURATION = "configuration"
    TEST = "test"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"


class QualityMetric(str, Enum):
    """Code quality metrics"""
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    TESTABILITY = "testability"
    DOCUMENTATION = "documentation"


@dataclass
class CodeRequirement:
    """Requirements for code generation"""
    requirement_id: str
    description: str
    language: CodeLanguage
    code_type: CodeType
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    quality_targets: Dict[QualityMetric, float] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-10, higher is more important
    
    def __post_init__(self):
        """Set default quality targets if not provided"""
        if not self.quality_targets:
            self.quality_targets = {
                QualityMetric.CORRECTNESS: 0.9,
                QualityMetric.PERFORMANCE: 0.7,
                QualityMetric.READABILITY: 0.8,
                QualityMetric.MAINTAINABILITY: 0.8,
                QualityMetric.SECURITY: 0.8,
                QualityMetric.TESTABILITY: 0.7,
                QualityMetric.DOCUMENTATION: 0.8
            }


@dataclass
class GeneratedCode:
    """Generated code with metadata"""
    code_id: str
    requirement_id: str
    code_content: str
    language: CodeLanguage
    code_type: CodeType
    quality_scores: Dict[QualityMetric, float] = field(default_factory=dict)
    generation_method: str = "autonomous"
    generation_time_ms: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_syntactically_valid(self) -> bool:
        """Check if generated code is syntactically valid"""
        try:
            if self.language == CodeLanguage.PYTHON:
                ast.parse(self.code_content)
                return True
            # For other languages, we'd use appropriate parsers
            return True
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score"""
        if not self.quality_scores:
            return 0.0
        
        # Weighted average with emphasis on correctness
        weights = {
            QualityMetric.CORRECTNESS: 0.3,
            QualityMetric.PERFORMANCE: 0.15,
            QualityMetric.READABILITY: 0.15,
            QualityMetric.MAINTAINABILITY: 0.15,
            QualityMetric.SECURITY: 0.15,
            QualityMetric.TESTABILITY: 0.05,
            QualityMetric.DOCUMENTATION: 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in self.quality_scores.items():
            weight = weights.get(metric, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class CodeTemplate:
    """Template system for code generation"""
    
    def __init__(self):
        self.templates: Dict[Tuple[CodeLanguage, CodeType], str] = {}
        self._initialize_templates()
    
    def _initialize_templates(self) -> None:
        """Initialize code templates for different languages and types"""
        
        # Python function template
        self.templates[(CodeLanguage.PYTHON, CodeType.FUNCTION)] = '''
def {function_name}({parameters}):
    """
    {description}
    
    Args:
        {args_documentation}
    
    Returns:
        {return_documentation}
    """
    {implementation}
    return {return_value}
'''

        # Python class template
        self.templates[(CodeLanguage.PYTHON, CodeType.CLASS)] = '''
class {class_name}:
    """
    {description}
    
    Attributes:
        {attributes_documentation}
    """
    
    def __init__(self{init_parameters}):
        """Initialize {class_name}."""
        {init_implementation}
    
    {methods}
'''

        # Python API endpoint template
        self.templates[(CodeLanguage.PYTHON, CodeType.API_ENDPOINT)] = '''
@router.{http_method}("{endpoint_path}")
async def {endpoint_name}({parameters}):
    """
    {description}
    
    Args:
        {args_documentation}
    
    Returns:
        {return_documentation}
    """
    try:
        {implementation}
        return {return_value}
    except Exception as e:
        logger.error(f"Error in {endpoint_name}: {{e}}")
        raise HTTPException(status_code=500, detail="Internal server error")
'''

        # Python test template
        self.templates[(CodeLanguage.PYTHON, CodeType.TEST)] = '''
import pytest
import asyncio
from unittest.mock import Mock, patch

{imports}

class Test{class_name}:
    """Test suite for {class_name}."""
    
    def setup_method(self):
        """Set up test fixtures."""
        {setup_code}
    
    {test_methods}
    
    def test_{function_name}_success(self):
        """Test successful execution of {function_name}."""
        {success_test}
    
    def test_{function_name}_error_handling(self):
        """Test error handling in {function_name}."""
        {error_test}
'''

        # SQL schema template
        self.templates[(CodeLanguage.SQL, CodeType.DATABASE_SCHEMA)] = '''
-- {description}
-- Created: {created_at}

{drop_statements}

{create_statements}

{index_statements}

{constraint_statements}

-- Sample data (optional)
{sample_data}
'''

        # Configuration templates
        self.templates[(CodeLanguage.YAML, CodeType.CONFIGURATION)] = '''
# {description}
# Generated: {created_at}

{configuration_content}
'''


class CodeAnalyzer:
    """Analyze and score generated code quality"""
    
    def __init__(self):
        self.analysis_cache: Dict[str, Dict[str, float]] = {}
        
    async def analyze_code_quality(
        self, 
        code: GeneratedCode,
        requirement: CodeRequirement
    ) -> Dict[QualityMetric, float]:
        """Analyze code quality across multiple metrics"""
        
        cache_key = f"{code.code_id}_{hash(code.code_content)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        quality_scores = {}
        
        # Analyze each quality metric
        quality_scores[QualityMetric.CORRECTNESS] = await self._analyze_correctness(code, requirement)
        quality_scores[QualityMetric.PERFORMANCE] = await self._analyze_performance(code)
        quality_scores[QualityMetric.READABILITY] = await self._analyze_readability(code)
        quality_scores[QualityMetric.MAINTAINABILITY] = await self._analyze_maintainability(code)
        quality_scores[QualityMetric.SECURITY] = await self._analyze_security(code)
        quality_scores[QualityMetric.TESTABILITY] = await self._analyze_testability(code)
        quality_scores[QualityMetric.DOCUMENTATION] = await self._analyze_documentation(code)
        
        # Cache results
        self.analysis_cache[cache_key] = quality_scores
        
        return quality_scores
    
    async def _analyze_correctness(self, code: GeneratedCode, requirement: CodeRequirement) -> float:
        """Analyze code correctness"""
        score = 0.5  # Base score
        
        # Syntax validation
        if code.is_syntactically_valid():
            score += 0.3
        
        # Check if code meets basic requirements
        if self._meets_basic_requirements(code, requirement):
            score += 0.2
        
        # Check for common errors
        if not self._has_common_errors(code):
            score += 0.1
        
        return min(1.0, score)
    
    async def _analyze_performance(self, code: GeneratedCode) -> float:
        """Analyze code performance characteristics"""
        score = 0.6  # Base score
        
        content = code.code_content.lower()
        
        # Check for performance best practices
        if 'async' in content and code.language == CodeLanguage.PYTHON:
            score += 0.1  # Async programming
        
        # Check for inefficient patterns
        if any(pattern in content for pattern in ['nested loop', 'o(n^2)', 'o(n*m)']):
            score -= 0.2
        
        # Check for optimization indicators
        if any(pattern in content for pattern in ['cache', 'memoize', 'index', 'batch']):
            score += 0.1
        
        # Check for proper resource management
        if 'with' in content and code.language == CodeLanguage.PYTHON:
            score += 0.1
        
        return max(0.1, min(1.0, score))
    
    async def _analyze_readability(self, code: GeneratedCode) -> float:
        """Analyze code readability"""
        score = 0.5  # Base score
        
        lines = code.code_content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Check line length (prefer shorter lines)
        avg_line_length = np.mean([len(line) for line in non_empty_lines]) if non_empty_lines else 0
        if avg_line_length < 80:
            score += 0.1
        elif avg_line_length > 120:
            score -= 0.1
        
        # Check for comments and documentation
        comment_ratio = sum(1 for line in lines if line.strip().startswith('#')) / len(lines) if lines else 0
        score += min(0.2, comment_ratio * 2)
        
        # Check for meaningful variable names
        if self._has_meaningful_names(code.code_content):
            score += 0.2
        
        # Check for proper indentation and formatting
        if self._is_well_formatted(code.code_content, code.language):
            score += 0.1
        
        return min(1.0, score)
    
    async def _analyze_maintainability(self, code: GeneratedCode) -> float:
        """Analyze code maintainability"""
        score = 0.5  # Base score
        
        # Check function/method complexity
        if self._has_low_complexity(code.code_content):
            score += 0.2
        
        # Check for modular design
        if self._is_modular(code.code_content, code.code_type):
            score += 0.2
        
        # Check for error handling
        if self._has_error_handling(code.code_content, code.language):
            score += 0.1
        
        # Check for configuration externalization
        if self._uses_configuration(code.code_content):
            score += 0.1
        
        return min(1.0, score)
    
    async def _analyze_security(self, code: GeneratedCode) -> float:
        """Analyze code security"""
        score = 0.7  # Base score (assume secure by default)
        
        content = code.code_content.lower()
        
        # Check for security vulnerabilities
        security_issues = [
            'eval(', 'exec(', 'system(', 'shell=true',
            'password', 'secret', 'token', 'api_key'
        ]
        
        for issue in security_issues:
            if issue in content:
                if issue in ['password', 'secret', 'token', 'api_key']:
                    # Check if it's properly handled (environment variables, etc.)
                    if 'os.environ' in content or 'getenv' in content:
                        score += 0.05  # Good practice
                    else:
                        score -= 0.15  # Hardcoded secrets
                else:
                    score -= 0.2  # Dangerous functions
        
        # Check for input validation
        if any(pattern in content for pattern in ['validate', 'sanitize', 'escape']):
            score += 0.1
        
        # Check for SQL injection protection
        if 'sql' in content and any(pattern in content for pattern in ['parameterized', 'prepared']):
            score += 0.1
        
        return max(0.1, min(1.0, score))
    
    async def _analyze_testability(self, code: GeneratedCode) -> float:
        """Analyze code testability"""
        score = 0.5  # Base score
        
        # Check for dependency injection
        if self._uses_dependency_injection(code.code_content):
            score += 0.2
        
        # Check for pure functions (no side effects)
        if self._has_pure_functions(code.code_content):
            score += 0.2
        
        # Check for test hooks
        if self._has_test_hooks(code.code_content):
            score += 0.1
        
        # Check for mockable dependencies
        if self._is_mockable(code.code_content):
            score += 0.1
        
        return min(1.0, score)
    
    async def _analyze_documentation(self, code: GeneratedCode) -> float:
        """Analyze code documentation quality"""
        score = 0.3  # Base score
        
        content = code.code_content
        
        # Check for docstrings (Python)
        if code.language == CodeLanguage.PYTHON:
            if '"""' in content or "'''" in content:
                score += 0.3
        
        # Check for inline comments
        lines = content.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        if comment_lines > 0:
            comment_ratio = comment_lines / len(lines)
            score += min(0.2, comment_ratio * 5)
        
        # Check for type hints (Python)
        if code.language == CodeLanguage.PYTHON and '->' in content:
            score += 0.1
        
        # Check for comprehensive documentation
        if len(content) > 500 and any(keyword in content.lower() for keyword in ['args:', 'returns:', 'raises:']):
            score += 0.1
        
        return min(1.0, score)
    
    def _meets_basic_requirements(self, code: GeneratedCode, requirement: CodeRequirement) -> bool:
        """Check if code meets basic requirements"""
        # This would be more sophisticated in a real implementation
        return len(code.code_content.strip()) > 10
    
    def _has_common_errors(self, code: GeneratedCode) -> bool:
        """Check for common coding errors"""
        content = code.code_content.lower()
        
        # Common Python errors
        if code.language == CodeLanguage.PYTHON:
            error_patterns = [
                'except:',  # Bare except
                'eval(',     # Eval usage
                'global ',   # Global variables
            ]
            return any(pattern in content for pattern in error_patterns)
        
        return False
    
    def _has_meaningful_names(self, content: str) -> bool:
        """Check for meaningful variable and function names"""
        # Simple heuristic: avoid single letter variables (except for loop counters)
        lines = content.split('\n')
        for line in lines:
            if '=' in line and not line.strip().startswith('#'):
                # Basic check for meaningful names
                if any(name in line for name in ['data', 'result', 'value', 'item', 'index']):
                    return True
        return True  # Default to true for now
    
    def _is_well_formatted(self, content: str, language: CodeLanguage) -> bool:
        """Check if code is well formatted"""
        if language == CodeLanguage.PYTHON:
            # Check for consistent indentation
            lines = [line for line in content.split('\n') if line.strip()]
            if not lines:
                return True
            
            # Simple indentation check
            indents = []
            for line in lines:
                if line.startswith('    ') or line.startswith('\t'):
                    indents.append(len(line) - len(line.lstrip()))
            
            # Should have some indentation consistency
            return len(set(indents)) <= 3 if indents else True
        
        return True  # Default for other languages
    
    def _has_low_complexity(self, content: str) -> bool:
        """Check for low cyclomatic complexity"""
        # Count decision points
        decision_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'and', 'or']
        complexity = sum(content.lower().count(keyword) for keyword in decision_keywords)
        return complexity < 10  # Arbitrary threshold
    
    def _is_modular(self, content: str, code_type: CodeType) -> bool:
        """Check for modular design"""
        if code_type == CodeType.FUNCTION:
            return len(content.split('\n')) < 50  # Functions should be reasonably sized
        elif code_type == CodeType.CLASS:
            return 'def ' in content  # Classes should have methods
        return True
    
    def _has_error_handling(self, content: str, language: CodeLanguage) -> bool:
        """Check for proper error handling"""
        if language == CodeLanguage.PYTHON:
            return 'try:' in content and 'except' in content
        return True  # Default for other languages
    
    def _uses_configuration(self, content: str) -> bool:
        """Check if code uses external configuration"""
        config_patterns = ['config', 'settings', 'env', 'environ']
        return any(pattern in content.lower() for pattern in config_patterns)
    
    def _uses_dependency_injection(self, content: str) -> bool:
        """Check for dependency injection patterns"""
        # Simple check for constructor parameters or function parameters
        return '(' in content and ')' in content and len(content.split('(')) > 2
    
    def _has_pure_functions(self, content: str) -> bool:
        """Check for pure functions (no side effects)"""
        # Simple heuristic: functions that return values
        return 'return ' in content
    
    def _has_test_hooks(self, content: str) -> bool:
        """Check for test-friendly hooks"""
        test_patterns = ['mock', 'stub', 'test', 'debug']
        return any(pattern in content.lower() for pattern in test_patterns)
    
    def _is_mockable(self, content: str) -> bool:
        """Check if code is mockable"""
        # Check for interface usage or dependency parameters
        return 'class ' in content or 'def ' in content


class CodeOptimizer:
    """Optimize generated code for performance and quality"""
    
    def __init__(self):
        self.optimization_patterns: Dict[CodeLanguage, List[Dict[str, str]]] = {}
        self._initialize_optimization_patterns()
    
    def _initialize_optimization_patterns(self) -> None:
        """Initialize code optimization patterns"""
        
        # Python optimization patterns
        self.optimization_patterns[CodeLanguage.PYTHON] = [
            {
                "pattern": "for i in range(len(items)):",
                "replacement": "for i, item in enumerate(items):",
                "reason": "Use enumerate for better readability and performance"
            },
            {
                "pattern": "if len(items) > 0:",
                "replacement": "if items:",
                "reason": "Direct boolean evaluation is more Pythonic"
            },
            {
                "pattern": "list(map(",
                "replacement": "[",
                "reason": "List comprehensions are often more readable and faster"
            },
            {
                "pattern": "string += item",
                "replacement": "items.append(item); string = ''.join(items)",
                "reason": "String concatenation in loops is inefficient"
            }
        ]
    
    async def optimize_code(self, code: GeneratedCode) -> Tuple[GeneratedCode, List[str]]:
        """Optimize generated code and return optimized version with suggestions"""
        
        optimized_content = code.code_content
        optimization_suggestions = []
        
        # Apply language-specific optimizations
        patterns = self.optimization_patterns.get(code.language, [])
        
        for pattern_info in patterns:
            pattern = pattern_info["pattern"]
            replacement = pattern_info["replacement"]
            reason = pattern_info["reason"]
            
            if pattern in optimized_content:
                optimized_content = optimized_content.replace(pattern, replacement)
                optimization_suggestions.append(f"Applied optimization: {reason}")
        
        # Performance optimizations
        performance_suggestions = await self._suggest_performance_optimizations(code)
        optimization_suggestions.extend(performance_suggestions)
        
        # Security optimizations
        security_suggestions = await self._suggest_security_optimizations(code)
        optimization_suggestions.extend(security_suggestions)
        
        # Create optimized code object
        optimized_code = GeneratedCode(
            code_id=f"{code.code_id}_optimized",
            requirement_id=code.requirement_id,
            code_content=optimized_content,
            language=code.language,
            code_type=code.code_type,
            generation_method="autonomous_optimized",
            optimization_suggestions=optimization_suggestions,
            dependencies=code.dependencies.copy()
        )
        
        return optimized_code, optimization_suggestions
    
    async def _suggest_performance_optimizations(self, code: GeneratedCode) -> List[str]:
        """Suggest performance optimizations"""
        suggestions = []
        content = code.code_content.lower()
        
        if code.language == CodeLanguage.PYTHON:
            # Check for potential performance issues
            if 'for' in content and 'in' in content and 'append' in content:
                suggestions.append("Consider using list comprehension instead of append in loops")
            
            if 'dict[' in content and 'key' in content:
                suggestions.append("Consider using dict.get() with default value for safer key access")
            
            if 'sql' in content and 'execute' in content:
                suggestions.append("Consider using connection pooling for database operations")
            
            if 'async' not in content and ('request' in content or 'api' in content):
                suggestions.append("Consider using async/await for I/O operations")
        
        return suggestions
    
    async def _suggest_security_optimizations(self, code: GeneratedCode) -> List[str]:
        """Suggest security optimizations"""
        suggestions = []
        content = code.code_content.lower()
        
        # Check for potential security issues
        if 'password' in content and 'input' in content:
            suggestions.append("Use secure input methods for passwords (e.g., getpass module)")
        
        if 'sql' in content and '+' in content:
            suggestions.append("Use parameterized queries to prevent SQL injection")
        
        if 'pickle' in content:
            suggestions.append("Consider alternatives to pickle for security (JSON, etc.)")
        
        if 'shell=true' in content:
            suggestions.append("Avoid shell=True in subprocess calls for security")
        
        return suggestions


class AutonomousCodeGenerator:
    """
    Main autonomous code generation system
    """
    
    def __init__(self):
        self.code_templates = CodeTemplate()
        self.code_analyzer = CodeAnalyzer()
        self.code_optimizer = CodeOptimizer()
        self.generation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        self.generated_code_cache: Dict[str, GeneratedCode] = {}
        self.improvement_patterns: Dict[str, List[str]] = defaultdict(list)
        self._learning_enabled = True
        
    async def generate_code(
        self, 
        requirement: CodeRequirement,
        optimization_level: int = 1
    ) -> GeneratedCode:
        """Generate code based on requirements"""
        
        start_time = time.time()
        
        with tracer.start_as_current_span("generate_code") as span:
            span.set_attributes({
                "language": requirement.language.value,
                "code_type": requirement.code_type.value,
                "requirement_id": requirement.requirement_id
            })
            
            try:
                # Generate initial code
                initial_code = await self._generate_initial_code(requirement)
                
                # Analyze quality
                quality_scores = await self.code_analyzer.analyze_code_quality(
                    initial_code, requirement
                )
                initial_code.quality_scores = quality_scores
                
                # Optimize if requested
                if optimization_level > 0:
                    optimized_code, suggestions = await self.code_optimizer.optimize_code(initial_code)
                    optimized_code.optimization_suggestions = suggestions
                    
                    # Re-analyze optimized code
                    optimized_quality = await self.code_analyzer.analyze_code_quality(
                        optimized_code, requirement
                    )
                    optimized_code.quality_scores = optimized_quality
                    
                    final_code = optimized_code
                else:
                    final_code = initial_code
                
                # Record generation time
                generation_time = (time.time() - start_time) * 1000
                final_code.generation_time_ms = generation_time
                
                # Store in cache
                self.generated_code_cache[final_code.code_id] = final_code
                
                # Record generation history
                self.generation_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "requirement_id": requirement.requirement_id,
                    "code_id": final_code.code_id,
                    "language": requirement.language.value,
                    "code_type": requirement.code_type.value,
                    "generation_time_ms": generation_time,
                    "overall_quality": final_code.get_overall_quality_score(),
                    "optimization_level": optimization_level,
                    "success": True
                })
                
                # Learn from generation
                if self._learning_enabled:
                    await self._learn_from_generation(requirement, final_code)
                
                logger.info(
                    f"Generated {requirement.code_type.value} code in {generation_time:.2f}ms "
                    f"with quality score {final_code.get_overall_quality_score():.3f}"
                )
                
                return final_code
                
            except Exception as e:
                generation_time = (time.time() - start_time) * 1000
                
                # Record failure
                self.generation_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "requirement_id": requirement.requirement_id,
                    "language": requirement.language.value,
                    "code_type": requirement.code_type.value,
                    "generation_time_ms": generation_time,
                    "error": str(e),
                    "success": False
                })
                
                logger.error(f"Code generation failed: {e}")
                raise
    
    async def _generate_initial_code(self, requirement: CodeRequirement) -> GeneratedCode:
        """Generate initial code based on requirement"""
        
        template_key = (requirement.language, requirement.code_type)
        template = self.code_templates.templates.get(template_key)
        
        if not template:
            # Fallback to generic generation
            return await self._generate_generic_code(requirement)
        
        # Extract parameters for template
        template_params = await self._extract_template_parameters(requirement)
        
        try:
            # Generate code from template
            generated_content = template.format(**template_params)
            
            # Clean up the generated content
            generated_content = textwrap.dedent(generated_content).strip()
            
            code = GeneratedCode(
                code_id=f"{requirement.requirement_id}_{int(time.time() * 1000)}",
                requirement_id=requirement.requirement_id,
                code_content=generated_content,
                language=requirement.language,
                code_type=requirement.code_type,
                dependencies=self._extract_dependencies(generated_content, requirement.language)
            )
            
            return code
            
        except KeyError as e:
            logger.warning(f"Template parameter missing: {e}")
            return await self._generate_generic_code(requirement)
    
    async def _extract_template_parameters(self, requirement: CodeRequirement) -> Dict[str, str]:
        """Extract parameters needed for code template"""
        
        params = requirement.parameters.copy()
        
        # Set default parameters based on code type
        if requirement.code_type == CodeType.FUNCTION:
            params.setdefault("function_name", self._generate_name_from_description(requirement.description))
            params.setdefault("parameters", self._generate_function_parameters(requirement))
            params.setdefault("description", requirement.description)
            params.setdefault("implementation", self._generate_function_implementation(requirement))
            params.setdefault("return_value", self._generate_return_value(requirement))
            params.setdefault("args_documentation", self._generate_args_docs(requirement))
            params.setdefault("return_documentation", self._generate_return_docs(requirement))
        
        elif requirement.code_type == CodeType.CLASS:
            params.setdefault("class_name", self._generate_class_name(requirement.description))
            params.setdefault("description", requirement.description)
            params.setdefault("init_parameters", self._generate_init_parameters(requirement))
            params.setdefault("init_implementation", self._generate_init_implementation(requirement))
            params.setdefault("methods", self._generate_class_methods(requirement))
            params.setdefault("attributes_documentation", self._generate_attributes_docs(requirement))
        
        elif requirement.code_type == CodeType.API_ENDPOINT:
            params.setdefault("http_method", requirement.parameters.get("method", "get"))
            params.setdefault("endpoint_path", requirement.parameters.get("path", "/api/endpoint"))
            params.setdefault("endpoint_name", self._generate_endpoint_name(requirement))
            params.setdefault("parameters", self._generate_endpoint_parameters(requirement))
            params.setdefault("description", requirement.description)
            params.setdefault("implementation", self._generate_endpoint_implementation(requirement))
            params.setdefault("return_value", self._generate_endpoint_return(requirement))
            params.setdefault("args_documentation", self._generate_endpoint_args_docs(requirement))
            params.setdefault("return_documentation", self._generate_endpoint_return_docs(requirement))
        
        # Convert all values to strings
        return {k: str(v) for k, v in params.items()}
    
    def _generate_name_from_description(self, description: str) -> str:
        """Generate a function/class name from description"""
        # Simple name generation from description
        words = description.lower().split()
        # Filter out common words
        filtered_words = [word for word in words if word not in ['a', 'an', 'the', 'and', 'or', 'but', 'for', 'to', 'of', 'in', 'on', 'at']]
        # Take first few meaningful words
        name_words = filtered_words[:3]
        # Convert to snake_case
        name = '_'.join(word.replace(',', '').replace('.', '') for word in name_words)
        return name if name else "generated_function"
    
    def _generate_function_parameters(self, requirement: CodeRequirement) -> str:
        """Generate function parameters"""
        params = requirement.parameters.get("parameters", [])
        if isinstance(params, list):
            return ", ".join(params)
        elif isinstance(params, dict):
            return ", ".join(f"{name}: {type_hint}" for name, type_hint in params.items())
        else:
            return "data: Dict[str, Any]"
    
    def _generate_function_implementation(self, requirement: CodeRequirement) -> str:
        """Generate function implementation"""
        # Basic implementation based on description
        if "calculate" in requirement.description.lower():
            return "    result = 0\n    # TODO: Implement calculation logic"
        elif "process" in requirement.description.lower():
            return "    # TODO: Implement processing logic\n    processed_data = data"
        elif "validate" in requirement.description.lower():
            return "    # TODO: Implement validation logic\n    if not data:\n        return False"
        else:
            return "    # TODO: Implement function logic\n    pass"
    
    def _generate_return_value(self, requirement: CodeRequirement) -> str:
        """Generate return value"""
        return_type = requirement.parameters.get("return_type", "result")
        if return_type == "bool":
            return "True"
        elif return_type == "int":
            return "0"
        elif return_type == "str":
            return "''"
        elif return_type == "dict":
            return "{}"
        elif return_type == "list":
            return "[]"
        else:
            return "result"
    
    def _generate_args_docs(self, requirement: CodeRequirement) -> str:
        """Generate arguments documentation"""
        params = requirement.parameters.get("parameters", {})
        if isinstance(params, dict):
            return "\n        ".join(f"{name}: {desc}" for name, desc in params.items())
        else:
            return "data: Input data for processing"
    
    def _generate_return_docs(self, requirement: CodeRequirement) -> str:
        """Generate return documentation"""
        return_type = requirement.parameters.get("return_type", "Any")
        return f"Returns the processed result ({return_type})"
    
    def _generate_class_name(self, description: str) -> str:
        """Generate class name from description"""
        name = self._generate_name_from_description(description)
        # Convert to PascalCase
        words = name.split('_')
        return ''.join(word.capitalize() for word in words)
    
    def _generate_init_parameters(self, requirement: CodeRequirement) -> str:
        """Generate __init__ parameters"""
        params = requirement.parameters.get("init_parameters", {})
        if params:
            param_strings = [f", {name}: {type_hint}" for name, type_hint in params.items()]
            return "".join(param_strings)
        else:
            return ""
    
    def _generate_init_implementation(self, requirement: CodeRequirement) -> str:
        """Generate __init__ implementation"""
        params = requirement.parameters.get("init_parameters", {})
        if params:
            assignments = [f"        self.{name} = {name}" for name in params.keys()]
            return "\n".join(assignments)
        else:
            return "        pass"
    
    def _generate_class_methods(self, requirement: CodeRequirement) -> str:
        """Generate class methods"""
        methods = requirement.parameters.get("methods", ["process"])
        method_code = []
        
        for method_name in methods:
            method_code.append(f"""
    def {method_name}(self, data: Any) -> Any:
        \"\"\"Process data using {method_name} method.\"\"\"
        # TODO: Implement {method_name} logic
        return data""")
        
        return "\n".join(method_code)
    
    def _generate_attributes_docs(self, requirement: CodeRequirement) -> str:
        """Generate attributes documentation"""
        params = requirement.parameters.get("init_parameters", {})
        if params:
            return "\n        ".join(f"{name}: {desc}" for name, desc in params.items())
        else:
            return "None"
    
    def _generate_endpoint_name(self, requirement: CodeRequirement) -> str:
        """Generate API endpoint name"""
        path = requirement.parameters.get("path", "/api/endpoint")
        # Convert path to function name
        name_parts = [part for part in path.split('/') if part and part != 'api']
        return '_'.join(name_parts) if name_parts else "api_endpoint"
    
    def _generate_endpoint_parameters(self, requirement: CodeRequirement) -> str:
        """Generate endpoint parameters"""
        params = ["request: Request"]
        
        # Add path parameters
        path_params = requirement.parameters.get("path_parameters", [])
        params.extend(f"{param}: str = Path(...)" for param in path_params)
        
        # Add query parameters
        query_params = requirement.parameters.get("query_parameters", [])
        params.extend(f"{param}: Optional[str] = Query(None)" for param in query_params)
        
        # Add body parameter if POST/PUT
        method = requirement.parameters.get("method", "get").lower()
        if method in ["post", "put", "patch"]:
            params.append("body: Dict[str, Any]")
        
        return ", ".join(params)
    
    def _generate_endpoint_implementation(self, requirement: CodeRequirement) -> str:
        """Generate endpoint implementation"""
        method = requirement.parameters.get("method", "get").lower()
        
        if method == "get":
            return "        data = await fetch_data()\n        return {'status': 'success', 'data': data}"
        elif method == "post":
            return "        result = await create_resource(body)\n        return {'status': 'created', 'id': result.id}"
        elif method == "put":
            return "        result = await update_resource(body)\n        return {'status': 'updated', 'data': result}"
        elif method == "delete":
            return "        await delete_resource()\n        return {'status': 'deleted'}"
        else:
            return "        return {'status': 'success', 'message': 'Operation completed'}"
    
    def _generate_endpoint_return(self, requirement: CodeRequirement) -> str:
        """Generate endpoint return value"""
        return "{'status': 'success'}"
    
    def _generate_endpoint_args_docs(self, requirement: CodeRequirement) -> str:
        """Generate endpoint arguments documentation"""
        return "request: HTTP request object"
    
    def _generate_endpoint_return_docs(self, requirement: CodeRequirement) -> str:
        """Generate endpoint return documentation"""
        return "JSON response with operation result"
    
    async def _generate_generic_code(self, requirement: CodeRequirement) -> GeneratedCode:
        """Generate generic code when no template is available"""
        
        if requirement.code_type == CodeType.FUNCTION:
            content = f'''
def {self._generate_name_from_description(requirement.description)}():
    """
    {requirement.description}
    """
    # TODO: Implement function logic
    pass
'''
        elif requirement.code_type == CodeType.CLASS:
            class_name = self._generate_class_name(requirement.description)
            content = f'''
class {class_name}:
    """
    {requirement.description}
    """
    
    def __init__(self):
        """Initialize {class_name}."""
        pass
    
    def process(self, data):
        """Process data."""
        return data
'''
        else:
            content = f'''
# {requirement.description}
# TODO: Implement {requirement.code_type.value}
'''
        
        return GeneratedCode(
            code_id=f"{requirement.requirement_id}_{int(time.time() * 1000)}",
            requirement_id=requirement.requirement_id,
            code_content=content.strip(),
            language=requirement.language,
            code_type=requirement.code_type
        )
    
    def _extract_dependencies(self, code_content: str, language: CodeLanguage) -> List[str]:
        """Extract dependencies from generated code"""
        dependencies = []
        
        if language == CodeLanguage.PYTHON:
            lines = code_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('import '):
                    module = line.replace('import ', '').split('.')[0]
                    dependencies.append(module)
                elif line.startswith('from '):
                    module = line.split()[1].split('.')[0]
                    dependencies.append(module)
        
        return list(set(dependencies))  # Remove duplicates
    
    async def _learn_from_generation(self, requirement: CodeRequirement, generated_code: GeneratedCode) -> None:
        """Learn from successful code generation"""
        
        # Record successful patterns
        quality_score = generated_code.get_overall_quality_score()
        
        if quality_score > 0.8:  # High quality code
            pattern_key = f"{requirement.language.value}_{requirement.code_type.value}"
            
            # Extract patterns that led to high quality
            if generated_code.optimization_suggestions:
                self.improvement_patterns[pattern_key].extend(generated_code.optimization_suggestions)
            
            # Learn parameter combinations that work well
            param_signature = str(sorted(requirement.parameters.items()))
            self.improvement_patterns[f"{pattern_key}_params"].append(param_signature)
    
    async def generate_test_code(self, generated_code: GeneratedCode) -> GeneratedCode:
        """Generate test code for the generated code"""
        
        if generated_code.language != CodeLanguage.PYTHON:
            raise ValueError("Test generation currently only supports Python")
        
        # Create test requirement
        test_requirement = CodeRequirement(
            requirement_id=f"test_{generated_code.requirement_id}",
            description=f"Tests for {generated_code.code_type.value}",
            language=CodeLanguage.PYTHON,
            code_type=CodeType.TEST,
            parameters={
                "target_code": generated_code.code_content,
                "class_name": self._extract_class_name(generated_code.code_content),
                "function_name": self._extract_function_name(generated_code.code_content)
            }
        )
        
        return await self.generate_code(test_requirement, optimization_level=0)
    
    def _extract_class_name(self, code_content: str) -> str:
        """Extract class name from code"""
        lines = code_content.split('\n')
        for line in lines:
            if line.strip().startswith('class '):
                return line.split()[1].split('(')[0].split(':')[0]
        return "TestTarget"
    
    def _extract_function_name(self, code_content: str) -> str:
        """Extract function name from code"""
        lines = code_content.split('\n')
        for line in lines:
            if line.strip().startswith('def ') and not line.strip().startswith('def __'):
                return line.split()[1].split('(')[0]
        return "test_function"
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics"""
        
        total_generations = len(self.generation_history)
        successful_generations = sum(1 for gen in self.generation_history if gen.get("success", False))
        
        success_rate = (successful_generations / total_generations * 100) if total_generations > 0 else 0
        
        # Language distribution
        language_dist = {}
        code_type_dist = {}
        
        for gen in self.generation_history:
            if gen.get("success", False):
                lang = gen.get("language", "unknown")
                code_type = gen.get("code_type", "unknown")
                
                language_dist[lang] = language_dist.get(lang, 0) + 1
                code_type_dist[code_type] = code_type_dist.get(code_type, 0) + 1
        
        # Performance metrics
        successful_gens = [gen for gen in self.generation_history if gen.get("success", False)]
        
        if successful_gens:
            avg_generation_time = np.mean([gen["generation_time_ms"] for gen in successful_gens])
            avg_quality = np.mean([gen.get("overall_quality", 0.5) for gen in successful_gens])
        else:
            avg_generation_time = 0
            avg_quality = 0
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_generations": total_generations,
            "successful_generations": successful_generations,
            "success_rate_percent": success_rate,
            "language_distribution": language_dist,
            "code_type_distribution": code_type_dist,
            "performance_metrics": {
                "avg_generation_time_ms": avg_generation_time,
                "avg_quality_score": avg_quality
            },
            "cached_code_count": len(self.generated_code_cache),
            "improvement_patterns_learned": len(self.improvement_patterns),
            "learning_enabled": self._learning_enabled
        }


# Global code generator instance
_code_generator: Optional[AutonomousCodeGenerator] = None


async def get_code_generator() -> AutonomousCodeGenerator:
    """Get or create the global autonomous code generator"""
    global _code_generator
    if _code_generator is None:
        _code_generator = AutonomousCodeGenerator()
    return _code_generator


async def generate_python_function(
    description: str,
    parameters: Dict[str, str] = None,
    return_type: str = "Any"
) -> GeneratedCode:
    """Convenience function to generate a Python function"""
    
    generator = await get_code_generator()
    
    requirement = CodeRequirement(
        requirement_id=f"python_function_{int(time.time() * 1000)}",
        description=description,
        language=CodeLanguage.PYTHON,
        code_type=CodeType.FUNCTION,
        parameters={
            "parameters": parameters or {},
            "return_type": return_type
        }
    )
    
    return await generator.generate_code(requirement)


async def generate_api_endpoint(
    description: str,
    method: str = "GET",
    path: str = "/api/endpoint",
    parameters: Dict[str, Any] = None
) -> GeneratedCode:
    """Convenience function to generate an API endpoint"""
    
    generator = await get_code_generator()
    
    requirement = CodeRequirement(
        requirement_id=f"api_endpoint_{int(time.time() * 1000)}",
        description=description,
        language=CodeLanguage.PYTHON,
        code_type=CodeType.API_ENDPOINT,
        parameters={
            "method": method.lower(),
            "path": path,
            **(parameters or {})
        }
    )
    
    return await generator.generate_code(requirement, optimization_level=1)