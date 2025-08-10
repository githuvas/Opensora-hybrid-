# 🚀 AI Movie Maker Installation Guide

This guide will help you install and set up the AI Movie Maker with Consistent Characters.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux, Windows 10/11, or macOS
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better recommended)
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 50GB+ free space
- **CUDA**: 11.8 or higher (for GPU acceleration)

### Software Requirements
- Python 3.8+
- pip (Python package installer)
- Git (for cloning the repository)

## 🔧 Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/ai-movie-maker.git
cd ai-movie-maker
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv ai_movie_maker_env

# Activate virtual environment
# On Linux/macOS:
source ai_movie_maker_env/bin/activate
# On Windows:
ai_movie_maker_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Install the Package
```bash
pip install -e .
```

### Step 5: Download Models (Optional)
The models will be downloaded automatically on first use, but you can pre-download them:

```bash
# Create models directory
mkdir -p models

# Download Wan2.1 models (will be done automatically)
# Download IP-Adapter models (will be done automatically)
```

## 🧪 Testing the Installation

### Run the Test Script
```bash
python3 test_installation.py
```

This will check:
- ✅ File structure
- ✅ Package imports
- ✅ GPU availability
- ✅ Model loading
- ✅ Custom components
- ✅ Main application

### Expected Output
```
==================================================
AI Movie Maker Installation Test
==================================================

--- File Structure ---
✓ ai_movie_maker.py
✓ ip_adapter_integration.py
✓ threestudio_integration.py
✓ self_improvement_engine.py
✓ requirements.txt
✓ setup.py
✓ README_AI_MOVIE_MAKER.md

--- Package Imports ---
✓ PyTorch 2.4.0
✓ TorchVision 0.19.0
✓ Diffusers 0.31.0
✓ Transformers 4.49.0
✓ Gradio 5.0.0
✓ OpenCV 4.9.0.80
✓ NumPy 1.24.0
✓ Pillow 10.0.0
✓ ImageIO 2.31.0
✓ Trimesh 4.0.0
✓ Open3D 0.17.0
✓ SciPy 1.11.0

--- GPU Availability ---
✓ GPU: NVIDIA GeForce RTX 4090 (24.0 GB)

--- Model Loading ---
✓ Wan2.1 pipeline available

--- Custom Components ---
✓ IP-Adapter integration available
✓ ThreeStudio integration available
✓ Self-improvement engine available

--- Main Application ---
✓ Main application available

==================================================
TEST SUMMARY
==================================================
File Structure: ✓ PASS
Package Imports: ✓ PASS
GPU Availability: ✓ PASS
Model Loading: ✓ PASS
Custom Components: ✓ PASS
Main Application: ✓ PASS

Overall: 6/6 tests passed
🎉 All tests passed! Installation is successful.
```

## 🚀 Starting the Application

### Method 1: Using the Launcher (Recommended)
```bash
python3 launch.py
```

### Method 2: Direct Launch
```bash
python3 ai_movie_maker.py
```

### Method 3: With Custom Parameters
```bash
python3 ai_movie_maker.py --port 8080 --host 0.0.0.0 --share
```

## 🌐 Accessing the Interface

Once started, open your web browser and navigate to:
- **Local**: http://localhost:7860
- **Network**: http://your-ip:7860
- **Public**: (if using --share flag)

## 🎯 First Steps

1. **Upload Character Images**: Add 1-10 reference images of the same person
2. **Enter Description**: Write a detailed scene description
3. **Set Duration**: Choose between 1-600 seconds
4. **Enable Physics**: Toggle physics simulation
5. **Adjust Strength**: Set character consistency strength (0.1-1.0)
6. **Generate**: Click "Generate Movie" and wait for completion

## 🔧 Configuration

### Environment Variables
```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set model cache directory
export HF_HOME=/path/to/model/cache

# Set temporary directory
export TMPDIR=/path/to/temp
```

### Configuration File
Create `config.json` in the project directory:
```json
{
    "models": {
        "t2v_model": "Wan-Video/Wan2.1-T2V-1.3B",
        "i2v_model": "Wan-Video/Wan2.1-I2V-14B",
        "ip_adapter_model": "ip-adapter-plus-face_sd15.safetensors"
    },
    "performance": {
        "use_fp16": true,
        "enable_attention_slicing": true,
        "enable_vae_slicing": true,
        "memory_efficient_attention": true
    },
    "output": {
        "default_resolution": [1920, 1080],
        "default_fps": 30,
        "default_duration": 10
    }
}
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory Error
```bash
# Solution: Reduce batch size or use smaller model
export CUDA_VISIBLE_DEVICES=0
python3 ai_movie_maker.py --low-memory
```

#### 2. Missing Dependencies
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

#### 3. CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Model Download Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface
python3 ai_movie_maker.py
```

### Performance Optimization

#### For RTX 3060 (8GB VRAM)
```python
# Use 1.3B model
config = {
    "t2v_model": "Wan-Video/Wan2.1-T2V-1.3B",
    "use_fp16": True,
    "enable_attention_slicing": True,
    "max_duration": 30
}
```

#### For RTX 4090 (24GB VRAM)
```python
# Use 14B model for higher quality
config = {
    "t2v_model": "Wan-Video/Wan2.1-T2V-14B",
    "use_fp16": True,
    "max_duration": 600
}
```

## 📚 Additional Resources

- [Wan2.1 Documentation](https://github.com/Wan-Video/Wan2.1)
- [IP-Adapter Guide](https://github.com/tencent-ailab/IP-Adapter)
- [ThreeStudio Documentation](https://github.com/threestudio-project/threestudio)
- [CodeLlama Models](https://github.com/meta-llama/codellama)

## 🤝 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the console
2. **Run tests**: Execute `python3 test_installation.py`
3. **Check system**: Ensure your system meets requirements
4. **Search issues**: Check existing GitHub issues
5. **Create issue**: Provide detailed error information

## 🎉 Success!

Once you see the Gradio interface, you're ready to create amazing AI movies with consistent characters!

---

**Happy Movie Making! 🎬**