#!/usr/bin/env python3
"""
Generation 1 Validation Tests - MAKE IT WORK (Simple Implementation)
Tests core functionality without external dependencies.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class TestGeneration1Basic(unittest.TestCase):
    """Basic functionality tests for Generation 1"""
    
    def test_core_imports(self):
        """Test that core modules can be imported"""
        try:
            from pipeline.config.settings import get_settings
            from pipeline.models.idea import Idea
            self.assertTrue(True, "Core imports successful")
        except ImportError as e:
            self.fail(f"Core import failed: {e}")
    
    def test_settings_configuration(self):
        """Test settings can be loaded"""
        from pipeline.config.settings import get_settings
        settings = get_settings()
        self.assertIsNotNone(settings)
        self.assertTrue(hasattr(settings, 'ENVIRONMENT'))
    
    def test_idea_model_creation(self):
        """Test basic Idea model creation"""
        from pipeline.models.idea import Idea
        from datetime import datetime
        
        # Create a basic idea instance
        idea_data = {
            'title': 'Test Startup Idea',
            'description': 'A test idea for validation',
            'category': 'saas',
            'status': 'DRAFT',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        idea = Idea(**idea_data)
        self.assertEqual(idea.title, 'Test Startup Idea')
        self.assertEqual(idea.category, 'saas')
        self.assertEqual(idea.status, 'DRAFT')
    
    def test_project_structure(self):
        """Test that key project directories exist"""
        directories = [
            'pipeline',
            'pipeline/config',
            'pipeline/models', 
            'pipeline/cli',
            'pipeline/core',
            'tests'
        ]
        
        for directory in directories:
            dir_path = project_root / directory
            self.assertTrue(dir_path.exists(), f"Directory {directory} should exist")
    
    def test_autonomous_executor_import(self):
        """Test autonomous executor can be imported"""
        try:
            from pipeline.core.autonomous_executor import AutonomousExecutor, AutonomousTask
            self.assertTrue(True, "Autonomous executor imports successful")
        except ImportError as e:
            self.fail(f"Autonomous executor import failed: {e}")

if __name__ == '__main__':
    print("🧪 Running Generation 1 Basic Functionality Tests")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)
