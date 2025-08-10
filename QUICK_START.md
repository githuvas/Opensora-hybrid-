# 🚀 Quick Start Guide - AI Movie Maker

## Instant Setup (3 Steps)

### 📋 Prerequisites
- **Python 3.10+** ✅ (Available in this environment)
- **GPU with CUDA** (Recommended for best performance)
- **8GB+ RAM** (16GB+ recommended)

### 🔧 Step 1: Setup Environment

**Option A: Virtual Environment (Recommended)**
```bash
# Create virtual environment
python3 -m venv ai_movie_maker_env

# Activate environment
source ai_movie_maker_env/bin/activate  # Linux/Mac
# or
ai_movie_maker_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Option B: System Install (Advanced)**
```bash
# Override system package management (use with caution)
pip install -r requirements.txt --break-system-packages
```

### 📦 Step 2: Prepare Models
```bash
# Create models directory
mkdir -p models

# Place your Wan2.1 model checkpoints in ./models/
# Download from: https://github.com/Wan-Video/Wan2.1
```

### 🎬 Step 3: Launch AI Movie Maker
```bash
# Create configuration
python3 launch_ai_movie_maker.py --create-config

# Launch the application
python3 launch_ai_movie_maker.py --ckpt_dir ./models
```

Then open your browser to: **http://localhost:7860**

## ✨ First Video Generation

1. **Enter prompt**: "A cat wearing sunglasses dancing on the beach"
2. **Set duration**: 5 seconds
3. **Click "Generate Movie"**
4. **Wait for magic!** ✨

## 🎯 Key Features Available

### ✅ **Ready to Use**
- Text-to-Video Generation (Wan2.1)
- Duration Control (1s to 10 minutes)
- Modern Gradio Interface
- Video Download

### ✅ **Character Consistency**
- Upload 1-10 reference images
- IP-Adapter integration
- Consistent characters across frames

### ✅ **Physics Simulation**
- Real-time physics world
- Object interactions
- 3D scene generation

### ✅ **Self-Improvement**
- CodeLlama integration
- Automatic optimization
- Performance monitoring

## 🛠️ Advanced Usage

### Custom Configuration
Edit `ai_movie_maker_config.json`:
```json
{
  "video": {
    "default_size": "832*480",
    "max_duration": 600
  },
  "character_consistency": {
    "enabled": true,
    "max_reference_images": 10
  }
}
```

### Command Line Options
```bash
# Debug mode
python3 launch_ai_movie_maker.py --debug --ckpt_dir ./models

# Custom port
python3 launch_ai_movie_maker.py --port 8080 --ckpt_dir ./models

# Share publicly (creates shareable link)
python3 launch_ai_movie_maker.py --share --ckpt_dir ./models

# Test all systems
python3 launch_ai_movie_maker.py --test-systems
```

## 🔍 Troubleshooting

### Common Issues

**1. "Command not found: python"**
```bash
# Use python3 instead
python3 launch_ai_movie_maker.py --ckpt_dir ./models
```

**2. "Externally managed environment"**
```bash
# Create virtual environment first
python3 -m venv ai_movie_maker_env
source ai_movie_maker_env/bin/activate
pip install -r requirements.txt
```

**3. "CUDA out of memory"**
- Reduce video duration
- Use smaller resolution (832×480)
- Close other GPU applications

**4. "Model not found"**
- Verify models are in ./models directory
- Check Wan2.1 model format compatibility

### System Check
```bash
# Verify your system
python3 launch_ai_movie_maker.py --test-systems
```

## 📊 Performance Tips

### GPU Memory Usage
| Model | VRAM Required |
|-------|---------------|
| T2V-1.3B | 8.2 GB |
| T2V-14B | 16.5 GB |
| I2V-14B | 16.8 GB |

### Generation Times (RTX 4090)
| Duration | Time |
|----------|------|
| 5 seconds | ~4 minutes |
| 30 seconds | ~15 minutes |
| 60 seconds | ~35 minutes |

## 🎨 Example Prompts

### Basic
```
"A robot dancing in a futuristic city"
```

### With Physics
```
"Red balls falling and bouncing in a room" + Enable Physics
```

### Character Consistency
```
"A knight fighting a dragon" + Upload character images
```

### Long Form
```
"A day in the life of a space explorer" + Duration: 60s
```

## 📚 What's Included

### Core Files
- `ai_movie_maker.py` - Main application
- `launch_ai_movie_maker.py` - Advanced launcher
- `ip_adapter_integration.py` - Character consistency
- `threestudio_integration.py` - Physics simulation
- `codellama_integration.py` - Self-improvement

### Supporting Files
- `requirements.txt` - Python dependencies
- `setup.py` - Automated setup script
- `README_AI_MOVIE_MAKER.md` - Complete documentation

## 🆘 Need Help?

1. **Check logs**: Look at `ai_movie_maker.log`
2. **Debug mode**: Add `--debug` flag
3. **System test**: Run `--test-systems`
4. **Read docs**: See `README_AI_MOVIE_MAKER.md`

## 🎯 Next Steps

1. **Generate your first video** with the basic setup
2. **Explore character consistency** by uploading reference images
3. **Try physics simulation** for dynamic scenes
4. **Experiment with longer videos** (up to 10 minutes)
5. **Read the full documentation** for advanced features

---

**🎬 Start Creating Amazing AI Videos in Minutes! 🎬**