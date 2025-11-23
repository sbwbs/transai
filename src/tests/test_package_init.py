#!/usr/bin/env python3
"""
Test script to validate that the main package __init__.py works correctly
"""

import sys
from pathlib import Path

# Add the parent directory to Python path for testing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_package_imports():
    """Test importing from the main package __init__.py"""
    
    print("Testing main package imports from __init__.py...")
    
    try:
        print("✓ Testing package-level imports...")
        from src import (
            EnhancedTranslationService,
            EnhancedTranslationRequest,
            OperationMode,
            ContextBuisample_clientr,
            TokenOptimizer,
            ValkeyManager,
            SessionManager,
            CachedGlossarySearch,
            BaseModelAdapter,
            OpenAIAdapter,
            EnhancedDataLoader
        )
        print("✓ All main package imports successful!")
        
        print(f"✓ EnhancedTranslationService: {EnhancedTranslationService}")
        print(f"✓ OperationMode: {OperationMode}")
        print(f"✓ ContextBuisample_clientr: {ContextBuisample_clientr}")
        print(f"✓ TokenOptimizer: {TokenOptimizer}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Package import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_package_imports()
    if success:
        print("\n🎉 Package reorganization successful! All imports working correctly.")
    else:
        print("\n❌ Package reorganization needs more work.")
        sys.exit(1)