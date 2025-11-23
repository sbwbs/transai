#!/usr/bin/env python3
"""
Production Import Test for Phase 2 Package Reorganization

This script tests that all Phase 2 components can be imported and used
in a production-like scenario, simulating how external scripts would
import and use the reorganized package.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path (as production scripts would)
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_production_imports():
    """Test production-style imports and basic functionality"""
    
    print("🧪 Testing Phase 2 Production Import Patterns")
    print("=" * 50)
    
    # Test 1: Main package imports
    print("\n1. Testing main package imports...")
    try:
        from src import (
            EnhancedTranslationService,
            EnhancedTranslationRequest,
            OperationMode,
            ContextBuisample_clientr,
            TokenOptimizer,
            ValkeyManager,
            SessionManager
        )
        print("✅ Main package imports successful")
    except ImportError as e:
        print(f"❌ Main package import failed: {e}")
        return False
    
    # Test 2: Direct module imports
    print("\n2. Testing direct module imports...")
    try:
        from src.enhanced_translation_service import EnhancedTranslationService
        from src.context_buisample_clientr import ContextBuisample_clientr, ContextRequest
        from src.token_optimizer import TokenOptimizer, ContextComponent
        from src.memory.valkey_manager import ValkeyManager
        from src.memory.session_manager import SessionManager
        from src.model_adapters import BaseModelAdapter, OpenAIAdapter
        print("✅ Direct module imports successful")
    except ImportError as e:
        print(f"❌ Direct module import failed: {e}")
        return False
    
    # Test 3: Basic object instantiation (without external dependencies)
    print("\n3. Testing basic object instantiation...")
    try:
        # Test TokenOptimizer (no external deps)
        optimizer = TokenOptimizer()
        print(f"✅ TokenOptimizer instantiated: {type(optimizer)}")
        
        # Test OperationMode enum
        mode = OperationMode.PHASE2_SMART_CONTEXT
        print(f"✅ OperationMode enum works: {mode}")
        
        # Test EnhancedTranslationRequest dataclass
        request = EnhancedTranslationRequest(
            korean_text="테스트",
            model_name="test-model",
            segment_id="seg-001",
            doc_id="doc-001"
        )
        print(f"✅ EnhancedTranslationRequest created: {request.korean_text}")
        
    except Exception as e:
        print(f"❌ Object instantiation failed: {e}")
        return False
    
    # Test 4: Import completeness check
    print("\n4. Testing import completeness...")
    try:
        # Check that all expected classes are available
        expected_classes = [
            'EnhancedTranslationService',
            'ContextBuisample_clientr', 
            'TokenOptimizer',
            'ValkeyManager',
            'SessionManager',
            'BaseModelAdapter',
            'OpenAIAdapter'
        ]
        
        import src
        for class_name in expected_classes:
            if hasattr(src, class_name):
                print(f"✅ {class_name} available in package")
            else:
                print(f"❌ {class_name} missing from package")
                return False
                
    except Exception as e:
        print(f"❌ Import completeness check failed: {e}")
        return False
    
    return True

def test_production_usage_pattern():
    """Test how production scripts would typically use the package"""
    
    print("\n" + "=" * 50)
    print("🚀 Testing Production Usage Patterns")
    print("=" * 50)
    
    try:
        # Typical production import pattern
        from src import (
            EnhancedTranslationService,
            EnhancedTranslationRequest, 
            OperationMode
        )
        
        # Create a translation request (typical usage)
        request = EnhancedTranslationRequest(
            korean_text="의료기기 번역 테스트",
            model_name="gpt-4o",
            segment_id="test-001", 
            doc_id="test-doc-001",
            operation_mode=OperationMode.PHASE2_SMART_CONTEXT,
            enable_session_tracking=True
        )
        
        print("✅ Production usage pattern works:")
        print(f"   - Korean text: {request.korean_text}")
        print(f"   - Operation mode: {request.operation_mode}")
        print(f"   - Session tracking: {request.enable_session_tracking}")
        
        return True
        
    except Exception as e:
        print(f"❌ Production usage pattern failed: {e}")
        return False

def main():
    """Run all production import tests"""
    
    print("🎯 Phase 2 Package Reorganization - Production Import Test")
    print("Testing import structure fixes and package organization...")
    
    success1 = test_production_imports()
    success2 = test_production_usage_pattern()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Phase 2 package reorganization successful")
        print("✅ Import issues resolved")
        print("✅ Production-ready package structure")
        print("\n📁 Reorganized structure:")
        print("   /phase2/src/")
        print("   ├── __init__.py (main package exports)")
        print("   ├── enhanced_translation_service.py (absolute imports)")
        print("   ├── context_buisample_clientr.py (absolute imports)")
        print("   ├── token_optimizer.py")
        print("   ├── memory/")
        print("   │   ├── __init__.py")
        print("   │   ├── valkey_manager.py (with mock fallback)")
        print("   │   ├── session_manager.py")
        print("   │   └── cached_glossary_search.py")
        print("   └── model_adapters/")
        print("       ├── __init__.py")
        print("       ├── base_adapter.py")
        print("       └── openai_adapter.py")
        
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("Package reorganization needs additional work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)