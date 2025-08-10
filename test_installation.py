#!/usr/bin/env python3
"""
Test script to verify AI Movie Maker installation
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required packages can be imported"""
    logger.info("Testing package imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        logger.info(f"✓ TorchVision {torchvision.__version__}")
        
        import diffusers
        logger.info(f"✓ Diffusers {diffusers.__version__}")
        
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__}")
        
        import gradio
        logger.info(f"✓ Gradio {gradio.__version__}")
        
        import opencv_python
        logger.info(f"✓ OpenCV {opencv_python.__version__}")
        
        import numpy
        logger.info(f"✓ NumPy {numpy.__version__}")
        
        import PIL
        logger.info(f"✓ Pillow {PIL.__version__}")
        
        import imageio
        logger.info(f"✓ ImageIO {imageio.__version__}")
        
        import trimesh
        logger.info(f"✓ Trimesh {trimesh.__version__}")
        
        import open3d
        logger.info(f"✓ Open3D {open3d.__version__}")
        
        import scipy
        logger.info(f"✓ SciPy {scipy.__version__}")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_gpu():
    """Test GPU availability"""
    logger.info("Testing GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"✓ GPU available: {gpu_name}")
            logger.info(f"✓ GPU count: {gpu_count}")
            logger.info(f"✓ GPU memory: {gpu_memory:.1f} GB")
            
            return True
        else:
            logger.warning("⚠ No GPU available, will use CPU (slower)")
            return False
            
    except Exception as e:
        logger.error(f"✗ GPU test failed: {e}")
        return False

def test_models():
    """Test model loading capabilities"""
    logger.info("Testing model loading...")
    
    try:
        from diffusers import WanPipeline
        
        # Test if we can access Wan2.1 models
        logger.info("✓ Wan2.1 pipeline available")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False

def test_components():
    """Test our custom components"""
    logger.info("Testing custom components...")
    
    try:
        # Test IP-Adapter integration
        from ip_adapter_integration import IPAdapterPlus, CharacterConsistencyEnhancer
        logger.info("✓ IP-Adapter integration available")
        
        # Test ThreeStudio integration
        from threestudio_integration import ThreeStudioManager, PhysicsObject
        logger.info("✓ ThreeStudio integration available")
        
        # Test self-improvement engine
        from self_improvement_engine import SelfImprovementEngine, CodeLlamaAnalyzer
        logger.info("✓ Self-improvement engine available")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Component test failed: {e}")
        return False

def test_main_app():
    """Test main application"""
    logger.info("Testing main application...")
    
    try:
        from ai_movie_maker import AIMovieMaker, create_gradio_interface
        logger.info("✓ Main application available")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Main app test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    logger.info("Testing file structure...")
    
    required_files = [
        "ai_movie_maker.py",
        "ip_adapter_integration.py",
        "threestudio_integration.py",
        "self_improvement_engine.py",
        "requirements.txt",
        "setup.py",
        "README_AI_MOVIE_MAKER.md"
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✓ {file}")
        else:
            logger.error(f"✗ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def run_all_tests():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("AI Movie Maker Installation Test")
    logger.info("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Package Imports", test_imports),
        ("GPU Availability", test_gpu),
        ("Model Loading", test_models),
        ("Custom Components", test_components),
        ("Main Application", test_main_app)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Installation is successful.")
        logger.info("\nTo start the AI Movie Maker:")
        logger.info("python ai_movie_maker.py")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Install missing dependencies: pip install -r requirements.txt")
        logger.info("2. Check GPU drivers and CUDA installation")
        logger.info("3. Ensure all files are present in the directory")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)