#!/usr/bin/env python3
"""
Phase 2 Integration Validation Script

Quick validation script to verify that all Phase 2 components are properly integrated
and working together. This script performs basic functionality checks without requiring
extensive setup or API keys.

Usage:
    python validate_integration.py

Requirements:
    - Valkey/Redis running on localhost:6379 (optional)
    - OpenAI API key for full validation (optional)
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_imports():
    """Check if all required modules can be imported"""
    print("🔍 Checking imports...")
    
    try:
        # Core Phase 2 components
        from enhanced_translation_service import EnhancedTranslationService, OperationMode
        from document_processor import DocumentProcessor, BatchConfiguration
        from context_buisample_clientr import ContextBuisample_clientr, ContextRequest
        from glossary_search import GlossarySearchEngine
        from memory.valkey_manager import ValkeyManager
        from memory.cached_glossary_search import CachedGlossarySearch
        from memory.session_manager import SessionManager
        from model_adapters.openai_adapter import OpenAIAdapter
        from token_optimizer import TokenOptimizer
        
        print("✅ All core components imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_glossary_search():
    """Test glossary search engine functionality"""
    print("\n🔍 Testing Glossary Search Engine...")
    
    try:
        from glossary_search import GlossarySearchEngine, create_sample_glossary
        
        # Initialize with sample data
        engine = GlossarySearchEngine()
        sample_terms = create_sample_glossary()
        engine.add_terms(sample_terms)
        
        # Test search
        results = engine.search("임상시험", max_results=3)
        
        if results:
            print(f"✅ Glossary search working: found {len(results)} results")
            for result in results[:2]:
                print(f"   - {result.term.korean} → {result.term.english} (score: {result.relevance_score:.2f})")
            return True
        else:
            print("⚠️  Glossary search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Glossary search test failed: {e}")
        return False

def check_context_buisample_clientr():
    """Test context buisample_clientr without external dependencies"""
    print("\n🔍 Testing Context Buisample_clientr...")
    
    try:
        from token_optimizer import TokenOptimizer, create_source_component, create_glossary_component
        
        # Test token optimizer
        optimizer = TokenOptimizer(target_token_limit=500)
        
        # Create test components
        source_comp = create_source_component("테스트 한국어 텍스트입니다.", "test_seg")
        glossary_comp = create_glossary_component(["임상시험 → clinical trial"])
        
        components = [source_comp, glossary_comp]
        
        # Test optimization
        result = optimizer.optimize_context(components, 500)
        
        if result.optimized_context:
            print(f"✅ Context buisample_clientr working: {result.total_tokens} tokens generated")
            print(f"   Components included: {len(result.components_included)}")
            return True
        else:
            print("⚠️  Context buisample_clientr produced empty result")
            return False
            
    except Exception as e:
        print(f"❌ Context buisample_clientr test failed: {e}")
        return False

def check_valkey_connection():
    """Test Valkey/Redis connection (optional)"""
    print("\n🔍 Testing Valkey Connection...")
    
    try:
        from memory.valkey_manager import ValkeyManager
        
        # Try to connect to Valkey
        valkey_manager = ValkeyManager(host="localhost", port=6379, db=15)  # Use test DB
        
        # Test basic operations
        test_key = "integration_test"
        valkey_manager.valkey_client.set(test_key, "test_value", ex=10)
        result = valkey_manager.valkey_client.get(test_key)
        
        if result == b"test_value":
            print("✅ Valkey connection working")
            valkey_manager.valkey_client.delete(test_key)
            valkey_manager.close()
            return True
        else:
            print("⚠️  Valkey connection established but basic operations failed")
            return False
            
    except Exception as e:
        print(f"⚠️  Valkey connection failed: {e}")
        print("   This is optional - Phase 2 can work without Valkey but with reduced functionality")
        return False

def check_enhanced_translation_service():
    """Test enhanced translation service initialization"""
    print("\n🔍 Testing Enhanced Translation Service...")
    
    try:
        from enhanced_translation_service import EnhancedTranslationService, OperationMode
        
        # Initialize without external dependencies
        service = EnhancedTranslationService(
            enable_valkey=False,  # Disable for validation
            fallback_to_phase1=True
        )
        
        # Check available models
        available_models = service.get_available_models()
        
        # Check health
        health = service.health_check()
        
        print(f"✅ Enhanced translation service initialized")
        print(f"   Available models: {len(available_models)}")
        print(f"   Health status: {health.get('status', 'unknown')}")
        print(f"   Phase 2 available: {service.phase2_available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced translation service test failed: {e}")
        return False

def check_document_processor():
    """Test document processor initialization"""
    print("\n🔍 Testing Document Processor...")
    
    try:
        from enhanced_translation_service import EnhancedTranslationService
        from document_processor import DocumentProcessor, BatchConfiguration
        
        # Initialize with minimal setup
        service = EnhancedTranslationService(enable_valkey=False)
        processor = DocumentProcessor(
            translation_service=service,
            enable_checkpointing=False
        )
        
        # Test configuration
        batch_config = BatchConfiguration(batch_size=5)
        
        print("✅ Document processor initialized")
        print(f"   Checkpointing: {processor.enable_checkpointing}")
        print(f"   Default batch size: {batch_config.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False

def check_file_structure():
    """Verify Phase 2 file structure"""
    print("\n🔍 Checking Phase 2 file structure...")
    
    required_files = [
        "enhanced_translation_service.py",
        "document_processor.py", 
        "context_buisample_clientr.py",
        "glossary_search.py",
        "token_optimizer.py",
        "memory/valkey_manager.py",
        "memory/cached_glossary_search.py",
        "memory/session_manager.py",
        "model_adapters/openai_adapter.py",
        "model_adapters/base_adapter.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if not missing_files:
        print("✅ All required files present")
        return True
    else:
        print(f"❌ Missing files: {missing_files}")
        return False

def main():
    """Run complete integration validation"""
    print("=" * 80)
    print("🔬 PHASE 2 INTEGRATION VALIDATION")
    print("=" * 80)
    
    tests = [
        ("File Structure", check_file_structure),
        ("Imports", check_imports),
        ("Glossary Search", check_glossary_search),
        ("Context Buisample_clientr", check_context_buisample_clientr),
        ("Valkey Connection", check_valkey_connection),
        ("Enhanced Translation Service", check_enhanced_translation_service),
        ("Document Processor", check_document_processor)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 VALIDATION SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED - Phase 2 integration is ready!")
        print("   You can proceed with running the full integration test.")
    elif passed >= total - 1:  # Allow for Valkey being optional
        print("\n✅ CORE VALIDATIONS PASSED - Phase 2 integration is mostly ready!")
        print("   Valkey connection may be optional for basic functionality.")
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED - Please check the errors above.")
        print("   Ensure all dependencies are installed and configured correctly.")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. For full testing: python test_phase2_integration.py")
    print("2. For demo: python phase2_production_demo.py")
    print("3. Documentation: see INTEGRATION_GUIDE.md")
    print("=" * 80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)