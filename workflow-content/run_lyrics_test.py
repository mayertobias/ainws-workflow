#!/usr/bin/env python3
"""
Runner script for lyrics API tests
Ensures proper environment setup and imports
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if we're in the right directory and have dependencies"""
    current_dir = Path.cwd()
    
    # Check if we're in workflow-content directory
    if current_dir.name != 'workflow-content':
        workflow_content_dir = current_dir / 'workflow-content'
        if workflow_content_dir.exists():
            print(f"📂 Changing to workflow-content directory: {workflow_content_dir}")
            os.chdir(workflow_content_dir)
        else:
            print("❌ Please run this script from the workflow-content directory")
            return False
    
    # Check if app directory exists
    if not (Path.cwd() / 'app').exists():
        print("❌ app directory not found. Make sure you're in workflow-content directory")
        return False
    
    # Check if virtual environment is active
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected. Make sure to activate your venv:")
        print("   source cenv/bin/activate  # or your venv name")
        return False
    
    print("✅ Environment checks passed")
    return True

def install_missing_dependencies():
    """Install any missing test dependencies"""
    try:
        import fastapi
        from fastapi.testclient import TestClient
        print("✅ FastAPI and TestClient available")
    except ImportError:
        print("📦 Installing missing test dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi[all]", "httpx"])

def run_test():
    """Run the lyrics API test"""
    test_file = Path.cwd() / 'tests' / 'test_lyrics_api.py'
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"🚀 Running lyrics API test: {test_file}")
    print("=" * 60)
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("\n✅ Test completed successfully!")
            return True
        else:
            print(f"\n❌ Test failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    """Main runner function"""
    print("🎵 Lyrics API Test Runner")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Install dependencies
    install_missing_dependencies()
    
    # Run test
    success = run_test()
    
    if success:
        print("\n🎉 All done! Check the tests/ directory for saved JSON responses.")
    else:
        print("\n❌ Test run failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 