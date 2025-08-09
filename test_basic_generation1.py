#!/usr/bin/env python3
"""
Generation 1 Basic Functionality Test
Tests that core functionality works as expected
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, '.')

async def test_generation1_basic():
    """Test basic functionality - MAKE IT WORK"""
    print("🧪 GENERATION 1 TEST: MAKE IT WORK")
    print("=" * 50)
    
    try:
        # Test 1: Core imports work
        print("✅ Test 1: Core Imports")
        from pipeline.main_pipeline import get_main_pipeline
        from pipeline.models.idea import Idea, IdeaStatus
        from core.search_tools import basic_web_search_tool, search_for_evidence
        print("   ✓ All core imports successful")
        
        # Test 2: Basic search functionality
        print("\n✅ Test 2: Search Tools")
        urls = basic_web_search_tool("AI startup validation", 3)
        print(f"   ✓ Basic web search returned {len(urls)} URLs")
        
        # Test async search
        evidence = await search_for_evidence("startup validation", 2)
        print(f"   ✓ Evidence search returned {len(evidence)} items")
        
        # Test 3: Data models work
        print("\n✅ Test 3: Data Models")
        test_idea = Idea(
            title="Test AI Startup",
            description="AI-powered customer support automation",
            category="ai_ml",
            status=IdeaStatus.DRAFT
        )
        print(f"   ✓ Idea model created: {test_idea.title}")
        print(f"   ✓ Status: {test_idea.status}")
        
        # Test 4: Pipeline initialization
        print("\n✅ Test 4: Pipeline Setup")
        pipeline = get_main_pipeline()
        print("   ✓ Main pipeline initialized")
        
        print("\n" + "=" * 50)
        print("🎉 GENERATION 1 BASIC FUNCTIONALITY: ✅ WORKING")
        print("✓ Core imports functional")
        print("✓ Search tools operational") 
        print("✓ Data models valid")
        print("✓ Pipeline accessible")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Generation 1 test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_generation1_basic())
    sys.exit(0 if success else 1)