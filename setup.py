#!/usr/bin/env python3
"""
Setup script for AI Movie Maker
Installs dependencies and prepares the system for immediate use
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

def setup_ai_movie_maker():
    """Complete setup process for AI Movie Maker"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🎬 Setting up AI Movie Maker...")
    logger.info("=" * 50)
    
    # Step 1: Install dependencies
    logger.info("📦 Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        logger.info("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False
    
    # Step 2: Create necessary directories
    logger.info("📁 Creating directories...")
    directories = ["models", "outputs", "temp", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    # Step 3: Create default configuration
    logger.info("⚙️ Creating configuration...")
    try:
        subprocess.run([
            sys.executable, "launch_ai_movie_maker.py", "--create-config"
        ], check=True)
        logger.info("✅ Configuration created")
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Configuration creation failed: {e}")
    
    # Step 4: Test system components
    logger.info("🧪 Testing system components...")
    try:
        subprocess.run([
            sys.executable, "launch_ai_movie_maker.py", "--test-systems", "--skip-checks"
        ], check=True)
        logger.info("✅ System tests completed")
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Some system tests failed: {e}")
    
    # Step 5: Make main script executable
    logger.info("🔧 Setting up launcher...")
    try:
        if sys.platform != "win32":
            os.chmod("launch_ai_movie_maker.py", 0o755)
            os.chmod("ai_movie_maker.py", 0o755)
        logger.info("✅ Scripts made executable")
    except Exception as e:
        logger.warning(f"⚠️ Could not set executable permissions: {e}")
    
    logger.info("=" * 50)
    logger.info("🎉 AI Movie Maker setup completed!")
    logger.info("")
    logger.info("📖 Next steps:")
    logger.info("1. Place Wan2.1 model checkpoints in ./models directory")
    logger.info("2. Run: python launch_ai_movie_maker.py --ckpt_dir ./models")
    logger.info("3. Open browser to http://localhost:7860")
    logger.info("")
    logger.info("💡 For help: python launch_ai_movie_maker.py --help")
    
    return True

if __name__ == "__main__":
    success = setup_ai_movie_maker()
    sys.exit(0 if success else 1)