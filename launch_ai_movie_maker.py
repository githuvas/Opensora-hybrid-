#!/usr/bin/env python3
"""
AI Movie Maker Launcher
Simple script to launch the AI Movie Maker with proper setup
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'torch', 'gradio', 'numpy', 'PIL', 'cv2', 'moviepy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_models():
    """Check if required models are available"""
    config_path = Path("config.json")
    
    if not config_path.exists():
        print("⚠️  Configuration file not found")
        print("Creating default configuration...")
        create_default_config()
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    missing_models = []
    
    # Check Wan model
    wan_path = config.get('models', {}).get('wan_checkpoint', '')
    if not wan_path or not Path(wan_path).exists():
        missing_models.append('Wan2.1 model')
    
    # Check IP-Adapter model
    ip_path = config.get('models', {}).get('ip_adapter_checkpoint', '')
    if not ip_path or not Path(ip_path).exists():
        missing_models.append('IP-Adapter model')
    
    if missing_models:
        print("⚠️  Missing model files:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nPlease download the required models and update config.json")
        return False
    
    print("✅ All required models are available")
    return True

def create_default_config():
    """Create default configuration file"""
    default_config = {
        "models": {
            "wan_checkpoint": "models/wan2.1-t2v-1.3b",
            "ip_adapter_checkpoint": "models/ip-adapter-plus-face_sd15.bin",
            "controlnet_checkpoint": "models/controlnet",
            "code_llama_path": "models/codellama-7b-instruct.gguf"
        },
        "generation": {
            "default_steps": 50,
            "default_guidance_scale": 7.5,
            "default_shift_scale": 5.0,
            "max_duration": 600,
            "min_duration": 1,
            "default_fps": 24
        },
        "physics": {
            "enabled": True,
            "gravity": 981,
            "friction": 0.7,
            "elasticity": 0.8
        },
        "character_consistency": {
            "ip_adapter_scale": 1.0,
            "face_consistency": True,
            "max_characters": 10,
            "max_reference_images": 5
        },
        "audio": {
            "enabled": True,
            "sample_rate": 22050,
            "voice_generation": True,
            "background_music": True
        },
        "output": {
            "formats": ["mp4", "webm", "gif"],
            "output_directory": "output",
            "temp_directory": "temp",
            "characters_directory": "characters"
        },
        "system": {
            "gpu_enabled": True,
            "parallel_processing": True,
            "memory_limit": 16
        }
    }
    
    with open("config.json", 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print("✅ Created default configuration file: config.json")
    print("Please edit config.json with your model paths")

def create_directories():
    """Create necessary directories"""
    directories = ["output", "temp", "characters", "models"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Created necessary directories")

def download_models():
    """Download required models (placeholder)"""
    print("📥 Model download functionality")
    print("Please download the following models manually:")
    print("1. Wan2.1-T2V-1.3B: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B")
    print("2. IP-Adapter: https://huggingface.co/h94/IP-Adapter")
    print("3. Code Llama (optional): https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF")

def launch_interface(port=7860, share=False, debug=False):
    """Launch the AI Movie Maker interface"""
    print(f"🚀 Launching AI Movie Maker on port {port}")
    
    try:
        # Import and launch the interface
        from gradio_ai_movie_maker import main
        
        # Set environment variables
        os.environ['GRADIO_SERVER_PORT'] = str(port)
        os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
        
        if share:
            os.environ['GRADIO_SHARE'] = 'True'
        
        if debug:
            os.environ['GRADIO_DEBUG'] = 'True'
        
        # Launch the interface
        main()
        
    except ImportError as e:
        print(f"❌ Failed to import interface: {e}")
        print("Please ensure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Failed to launch interface: {e}")
        return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="AI Movie Maker Launcher")
    parser.add_argument("--port", type=int, default=7860, help="Port for the interface")
    parser.add_argument("--share", action="store_true", help="Share the interface publicly")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and models")
    parser.add_argument("--setup", action="store_true", help="Setup the environment")
    parser.add_argument("--download-models", action="store_true", help="Download required models")
    
    args = parser.parse_args()
    
    print("🎬 AI Movie Maker Launcher")
    print("=" * 50)
    
    # Setup mode
    if args.setup:
        print("🔧 Setting up AI Movie Maker...")
        create_directories()
        create_default_config()
        print("\n✅ Setup complete!")
        print("Next steps:")
        print("1. Download required models")
        print("2. Update config.json with model paths")
        print("3. Run: python launch_ai_movie_maker.py")
        return
    
    # Download models mode
    if args.download_models:
        download_models()
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Check models
    if not check_models():
        print("\n⚠️  Some models are missing")
        print("You can still run the interface, but some features may not work")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Check-only mode
    if args.check_only:
        print("\n✅ Environment check complete!")
        return
    
    # Launch interface
    print("\n🚀 Starting AI Movie Maker...")
    launch_interface(args.port, args.share, args.debug)

if __name__ == "__main__":
    main()