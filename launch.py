#!/usr/bin/env python3
"""
AI Movie Maker Launcher
Simple launcher script with error handling and user guidance
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'torch', 'diffusers', 'transformers', 'gradio', 
        'opencv-python', 'numpy', 'PIL', 'imageio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"\nMissing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install -r requirements.txt")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'ai_movie_maker.py',
        'ip_adapter_integration.py',
        'threestudio_integration.py',
        'self_improvement_engine.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✓ {file}")
        else:
            logger.error(f"✗ {file} - Missing")
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"\nMissing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            logger.warning("⚠ No GPU detected - will use CPU (slower)")
            return True
    except Exception as e:
        logger.warning(f"⚠ GPU check failed: {e}")
        return True

def show_welcome():
    """Show welcome message"""
    print("=" * 60)
    print("🎬 AI Movie Maker with Consistent Characters")
    print("=" * 60)
    print()
    print("Features:")
    print("• Character consistency using IP-Adapter")
    print("• Text-to-video generation with Wan2.1")
    print("• Physics world rendering with ThreeStudio")
    print("• Self-improvement using CodeLlama")
    print("• Duration: 1s to 10m")
    print("• Download options for videos")
    print()
    print("System Requirements:")
    print("• GPU: RTX 3060+ (8GB+ VRAM)")
    print("• RAM: 16GB+")
    print("• Storage: 50GB+ free space")
    print()

def show_usage_tips():
    """Show usage tips"""
    print("Usage Tips:")
    print("• Use 5-10 character reference images for best consistency")
    print("• Start with shorter durations (5-10 seconds) for testing")
    print("• Enable physics for realistic movement")
    print("• Adjust character strength based on your needs")
    print("• Monitor GPU memory usage during generation")
    print()

def launch_app():
    """Launch the AI Movie Maker application"""
    try:
        logger.info("Starting AI Movie Maker...")
        
        # Import and run the main application
        from ai_movie_maker import main
        
        # Run the application
        main()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.info("Try running: python ai_movie_maker.py --help")
        return False
    
    return True

def main():
    """Main launcher function"""
    show_welcome()
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Files", check_files),
        ("GPU", check_gpu)
    ]
    
    logger.info("Running system checks...")
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        if not check_func():
            logger.error(f"❌ {check_name} check failed")
            logger.info("\nPlease fix the issues above and try again.")
            return False
    
    logger.info("\n✅ All checks passed!")
    show_usage_tips()
    
    # Ask user if they want to continue
    try:
        response = input("Press Enter to start the AI Movie Maker (or Ctrl+C to exit): ")
    except KeyboardInterrupt:
        logger.info("Launch cancelled by user")
        return True
    
    # Launch the application
    return launch_app()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)