#!/usr/bin/env python3
"""
Basic functionality test for Generation 1: Make it Work
"""

import sys
import os
from datetime import datetime
from uuid import uuid4

# Add project root to path
sys.path.insert(0, '/root/repo')

def test_basic_imports():
    """Test that core modules can be imported."""
    try:
        from pipeline.config.settings import get_settings
        from pipeline.models.idea import Idea, IdeaStatus, IdeaCategory, IdeaDraft
        print("✅ Core imports working")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_settings_configuration():
    """Test that settings can be loaded."""
    try:
        from pipeline.config.settings import get_settings
        settings = get_settings()
        print(f"✅ Settings loaded for environment: {settings.environment}")
        return True
    except Exception as e:
        print(f"❌ Settings error: {e}")
        return False

def test_idea_model_creation():
    """Test that idea models can be created."""
    try:
        from pipeline.models.idea import Idea, IdeaStatus, IdeaCategory, IdeaDraft
        
        # Create a draft idea
        draft = IdeaDraft(
            title="AI-Powered Code Review Assistant",
            description="Automated code review tool that provides intelligent feedback on pull requests using machine learning to identify patterns, bugs, and style issues.",
            category=IdeaCategory.AI_ML,
            problem_statement="Manual code review is time-consuming and inconsistent",
            solution_description="AI agent that analyzes code diffs and provides actionable feedback",
            target_market="Software development teams and DevOps engineers"
        )
        
        print("✅ IdeaDraft created successfully")
        
        # Create full idea
        idea = Idea(
            id=uuid4(),
            title=draft.title,
            description=draft.description,
            category=draft.category,
            status=IdeaStatus.DRAFT,
            problem_statement=draft.problem_statement,
            solution_description=draft.solution_description,
            target_market=draft.target_market,
            evidence_links=draft.evidence_links,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        print(f"✅ Idea model created: {idea.title}")
        return True
        
    except Exception as e:
        print(f"❌ Idea model error: {e}")
        return False

def test_api_imports():
    """Test API components can be imported."""
    try:
        from pipeline.api.gateway import app
        print("✅ API gateway import working")
        return True
    except Exception as e:
        print(f"❌ API import error: {e}")
        return False

def run_generation_1_tests():
    """Run all Generation 1 basic functionality tests."""
    print("🚀 Running Generation 1: Make it Work Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Settings Configuration", test_settings_configuration),
        ("Idea Model Creation", test_idea_model_creation),
        ("API Imports", test_api_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📊 Generation 1 Test Results:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n📈 Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Generation 1: Make it Work - COMPLETE!")
        return True
    else:
        print("⚠️  Generation 1: Some basic functionality issues remain")
        return False

if __name__ == "__main__":
    success = run_generation_1_tests()
    sys.exit(0 if success else 1)