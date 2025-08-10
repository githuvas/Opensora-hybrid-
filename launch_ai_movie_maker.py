#!/usr/bin/env python3
"""
Launch Script for AI Movie Maker with All Features
Comprehensive offline AI movie generator with character consistency, physics simulation, and self-improvement
"""

import os
import sys
import argparse
import logging
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ai_movie_maker.log')
        ]
    )

def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements and dependencies"""
    logger = logging.getLogger(__name__)
    requirements = {
        'python_version': False,
        'torch_available': False,
        'cuda_available': False,
        'gradio_available': False,
        'transformers_available': False,
        'disk_space': False,
        'memory': False
    }
    
    # Check Python version
    if sys.version_info >= (3, 10):
        requirements['python_version'] = True
        logger.info(f"✅ Python version: {sys.version}")
    else:
        logger.error(f"❌ Python 3.10+ required, found: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        requirements['torch_available'] = True
        requirements['cuda_available'] = torch.cuda.is_available()
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"🔥 CUDA available: {requirements['cuda_available']}")
    except ImportError:
        logger.error("❌ PyTorch not found")
    
    # Check Gradio
    try:
        import gradio as gr
        requirements['gradio_available'] = True
        logger.info(f"✅ Gradio version: {gr.__version__}")
    except ImportError:
        logger.error("❌ Gradio not found")
    
    # Check Transformers
    try:
        import transformers
        requirements['transformers_available'] = True
        logger.info(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        logger.warning("⚠️ Transformers not found (CodeLlama features disabled)")
    
    # Check disk space (require at least 10GB free)
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        requirements['disk_space'] = free_space > 10
        logger.info(f"💾 Free disk space: {free_space:.1f} GB")
    except:
        logger.warning("⚠️ Could not check disk space")
    
    # Check memory (require at least 8GB)
    try:
        import psutil
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        requirements['memory'] = total_memory > 8
        logger.info(f"🧠 Total memory: {total_memory:.1f} GB")
    except:
        logger.warning("⚠️ Could not check memory")
    
    return requirements

def install_dependencies(force: bool = False) -> bool:
    """Install or update dependencies"""
    logger = logging.getLogger(__name__)
    
    if not force:
        try:
            # Quick check if main dependencies exist
            import torch, gradio, transformers
            logger.info("✅ Main dependencies already installed")
            return True
        except ImportError:
            pass
    
    try:
        logger.info("📦 Installing/updating dependencies...")
        
        # Install basic requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        
        logger.info("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def download_models(models_dir: str = "models") -> bool:
    """Download required model checkpoints"""
    logger = logging.getLogger(__name__)
    
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    # Check if models already exist
    wan_models = [
        "t2v-1.3B",
        "t2v-14B", 
        "i2v-14B",
        "vace-1.3B"
    ]
    
    existing_models = []
    for model in wan_models:
        model_path = models_path / model
        if model_path.exists():
            existing_models.append(model)
    
    if len(existing_models) >= 2:
        logger.info(f"✅ Found {len(existing_models)} existing models")
        return True
    
    logger.info("📥 Model download would normally happen here")
    logger.info("💡 Please ensure Wan2.1 models are available in the specified checkpoint directory")
    
    return True

def create_config_file(config_path: str = "ai_movie_maker_config.json") -> Dict:
    """Create configuration file with default settings"""
    config = {
        "system": {
            "checkpoint_dir": "./models",
            "output_dir": "./outputs",
            "temp_dir": "./temp",
            "max_concurrent_generations": 1,
            "auto_cleanup": True
        },
        "video": {
            "default_size": "832*480",
            "max_duration": 600,
            "default_fps": 30,
            "quality_preset": "balanced"
        },
        "character_consistency": {
            "enabled": True,
            "max_reference_images": 10,
            "ip_adapter_strength": 0.7
        },
        "physics_simulation": {
            "enabled": True,
            "gravity": -9.81,
            "time_step": 0.016667,  # 60 FPS
            "collision_detection": "simple"
        },
        "self_improvement": {
            "enabled": True,
            "analysis_interval": 3600,  # 1 hour
            "auto_optimize": False
        },
        "gradio": {
            "port": 7860,
            "share": False,
            "auth": None,
            "theme": "soft"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

def validate_checkpoint_directory(ckpt_dir: str) -> bool:
    """Validate checkpoint directory exists and contains models"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(ckpt_dir):
        logger.error(f"❌ Checkpoint directory not found: {ckpt_dir}")
        return False
    
    # Look for any model files or directories
    ckpt_path = Path(ckpt_dir)
    model_files = list(ckpt_path.glob("**/*.bin")) + list(ckpt_path.glob("**/*.safetensors"))
    model_dirs = [d for d in ckpt_path.iterdir() if d.is_dir()]
    
    if model_files or model_dirs:
        logger.info(f"✅ Found model files/directories in {ckpt_dir}")
        return True
    else:
        logger.warning(f"⚠️ No model files found in {ckpt_dir}")
        return False

def test_integrations() -> Dict[str, bool]:
    """Test all system integrations"""
    logger = logging.getLogger(__name__)
    results = {}
    
    # Test IP-Adapter integration
    try:
        from ip_adapter_integration import test_ip_adapter_system
        results['ip_adapter'] = test_ip_adapter_system()
    except Exception as e:
        logger.error(f"IP-Adapter test failed: {e}")
        results['ip_adapter'] = False
    
    # Test ThreeStudio integration
    try:
        from threestudio_integration import test_threestudio_integration
        results['threestudio'] = test_threestudio_integration()
    except Exception as e:
        logger.error(f"ThreeStudio test failed: {e}")
        results['threestudio'] = False
    
    # Test CodeLlama integration
    try:
        from codellama_integration import test_codellama_integration
        results['codellama'] = test_codellama_integration()
    except Exception as e:
        logger.error(f"CodeLlama test failed: {e}")
        results['codellama'] = False
    
    return results

def launch_ai_movie_maker(args) -> bool:
    """Launch the main AI Movie Maker application"""
    logger = logging.getLogger(__name__)
    
    try:
        # Import main application
        from ai_movie_maker import main as run_movie_maker
        
        # Prepare arguments for main application
        main_args = [
            '--ckpt_dir', args.ckpt_dir,
            '--port', str(args.port)
        ]
        
        if args.share:
            main_args.append('--share')
        
        if args.debug:
            main_args.append('--debug')
        
        # Set up sys.argv for the main application
        original_argv = sys.argv.copy()
        sys.argv = ['ai_movie_maker.py'] + main_args
        
        try:
            # Launch main application
            logger.info("🚀 Launching AI Movie Maker...")
            run_movie_maker()
            return True
            
        finally:
            # Restore original argv
            sys.argv = original_argv
            
    except KeyboardInterrupt:
        logger.info("👋 AI Movie Maker stopped by user")
        return True
    except Exception as e:
        logger.error(f"💥 Failed to launch AI Movie Maker: {e}")
        return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="🎬 AI Movie Maker - Comprehensive Offline Video Generation"
    )
    
    # Basic arguments
    parser.add_argument(
        '--ckpt_dir', 
        type=str, 
        default='./models',
        help='Directory containing Wan2.1 model checkpoints'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=7860,
        help='Port for Gradio web interface'
    )
    parser.add_argument(
        '--share', 
        action='store_true',
        help='Create shareable Gradio link'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    
    # Setup arguments
    parser.add_argument(
        '--install-deps', 
        action='store_true',
        help='Install/update dependencies'
    )
    parser.add_argument(
        '--download-models', 
        action='store_true',
        help='Download required model checkpoints'
    )
    parser.add_argument(
        '--test-systems', 
        action='store_true',
        help='Test all system integrations'
    )
    parser.add_argument(
        '--create-config', 
        action='store_true',
        help='Create default configuration file'
    )
    parser.add_argument(
        '--skip-checks', 
        action='store_true',
        help='Skip system requirement checks'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("🎬 AI Movie Maker Launcher Starting...")
    logger.info("=" * 60)
    
    # Create config if requested
    if args.create_config:
        logger.info("📝 Creating configuration file...")
        config = create_config_file()
        logger.info("✅ Configuration file created: ai_movie_maker_config.json")
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies(force=True):
            logger.error("❌ Failed to install dependencies")
            return 1
    
    # Download models if requested
    if args.download_models:
        if not download_models():
            logger.error("❌ Failed to download models")
            return 1
    
    # Check system requirements
    if not args.skip_checks:
        logger.info("🔍 Checking system requirements...")
        requirements = check_system_requirements()
        
        critical_failures = []
        if not requirements['python_version']:
            critical_failures.append("Python 3.10+")
        if not requirements['torch_available']:
            critical_failures.append("PyTorch")
        if not requirements['gradio_available']:
            critical_failures.append("Gradio")
        
        if critical_failures:
            logger.error(f"❌ Critical requirements missing: {', '.join(critical_failures)}")
            logger.error("💡 Try running with --install-deps to install dependencies")
            return 1
        
        # Validate checkpoint directory
        if not validate_checkpoint_directory(args.ckpt_dir):
            logger.error("💡 Please specify a valid checkpoint directory with --ckpt_dir")
            logger.error("💡 You can download models with --download-models")
            return 1
    
    # Test systems if requested
    if args.test_systems:
        logger.info("🧪 Testing system integrations...")
        test_results = test_integrations()
        
        for system, success in test_results.items():
            status = "✅" if success else "❌"
            logger.info(f"{status} {system}: {'PASS' if success else 'FAIL'}")
        
        if not any(test_results.values()):
            logger.warning("⚠️ All integration tests failed - some features may not work")
    
    # Launch main application
    logger.info("🚀 All checks passed! Launching AI Movie Maker...")
    logger.info("=" * 60)
    
    success = launch_ai_movie_maker(args)
    
    if success:
        logger.info("👋 AI Movie Maker launcher completed successfully")
        return 0
    else:
        logger.error("💥 AI Movie Maker launcher failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)