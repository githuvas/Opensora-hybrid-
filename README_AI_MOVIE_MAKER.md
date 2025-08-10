# 🎬 AI Movie Maker with Consistent Characters

An advanced offline AI movie maker that creates videos with consistent characters using state-of-the-art AI models including Wan2.1, IP-Adapter, and ThreeStudio physics simulation.

## ✨ Features

### 🎭 Character Consistency
- **IP-Adapter Integration**: Maintain character appearance across video frames
- **Multiple Reference Images**: Support for 1-10 character reference images
- **Face Detection & Enhancement**: Automatic face region extraction for better consistency
- **Adaptive Strength Control**: Adjustable character consistency strength (0.1-1.0)

### 🎥 Video Generation
- **Wan2.1 Text-to-Video**: High-quality video generation using Wan2.1 models
- **Duration Control**: Generate videos from 1 second to 10 minutes
- **Multiple Resolutions**: Support for various output resolutions (up to 4K)
- **Frame Rate Control**: Configurable FPS (15-60 fps)

### 🌍 Physics World Rendering
- **ThreeStudio Integration**: Realistic 3D physics simulation
- **Physics Objects**: Add realistic physics to scenes
- **Character Animation**: Physics-based character movement
- **Scene Management**: 3D scene composition and rendering

### 🤖 Self-Improvement Engine
- **CodeLlama Integration**: AI-powered workflow optimization
- **Performance Analysis**: Automatic bottleneck detection
- **Code Quality Assessment**: Continuous code improvement suggestions
- **Workflow Optimization**: Real-time optimization recommendations

### 🎨 Advanced Features
- **Download Options**: Direct video download functionality
- **Batch Processing**: Generate multiple videos simultaneously
- **Custom Prompts**: Rich text input for scene description
- **Gradio Interface**: User-friendly web interface

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-repo/ai-movie-maker.git
cd ai-movie-maker
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install the package**:
```bash
pip install -e .
```

### Basic Usage

1. **Start the application**:
```bash
python ai_movie_maker.py
```

2. **Open your browser** and navigate to `http://localhost:7860`

3. **Create your first movie**:
   - Enter a text description of your scene
   - Upload 1-10 character reference images
   - Set duration (1-600 seconds)
   - Enable physics simulation (optional)
   - Click "Generate Movie"

## 📖 Detailed Usage

### Character Consistency

The AI Movie Maker uses IP-Adapter to maintain character consistency across video frames:

```python
from ai_movie_maker import AIMovieMaker

# Initialize the movie maker
movie_maker = AIMovieMaker()

# Create movie with character consistency
result = movie_maker.create_movie(
    text_prompt="A superhero flying through a futuristic city",
    character_images=[image1, image2, image3],  # 1-10 reference images
    duration=10,  # seconds
    physics_enabled=True,
    character_strength=0.8  # Consistency strength
)
```

### Physics Simulation

Enable realistic physics for your scenes:

```python
from threestudio_integration import ThreeStudioManager

# Create physics scene
scene = ThreeStudioManager()

# Add character with physics
scene.add_character(
    character_id="hero",
    mesh_path="path/to/character.obj",
    position=[0, 1, 0]
)

# Add physics objects
scene.add_object(
    object_id="cube",
    mesh_path="path/to/cube.obj",
    position=[2, 0, 0],
    mass=5.0
)

# Simulate physics
frames = scene.simulate_physics(duration=10, fps=30)
```

### Self-Improvement

The system continuously improves itself using CodeLlama:

```python
from self_improvement_engine import SelfImprovementEngine

# Initialize improvement engine
improvement_engine = SelfImprovementEngine()

# Analyze and improve workflow
workflow_data = {
    "prompt": "Your movie description",
    "duration": 10,
    "physics_enabled": True,
    "character_count": 3
}

result = improvement_engine.analyze_and_improve(workflow_data)
print(f"Optimization potential: {result['summary']['estimated_improvement']}%")
```

## 🔧 Configuration

### Model Configuration

The system supports multiple model configurations:

```python
# Wan2.1 Models
T2V_MODELS = {
    "1.3B": "Wan-Video/Wan2.1-T2V-1.3B",  # Fast generation
    "14B": "Wan-Video/Wan2.1-T2V-14B"     # High quality
}

# IP-Adapter Models
IP_ADAPTER_MODELS = {
    "plus_face": "ip-adapter-plus-face_sd15.safetensors",
    "plus": "ip-adapter-plus_sd15.safetensors"
}
```

### Performance Settings

Optimize for your hardware:

```python
# GPU Optimization
config = {
    "use_fp16": True,
    "enable_attention_slicing": True,
    "enable_vae_slicing": True,
    "memory_efficient_attention": True
}

# Quality vs Speed
quality_settings = {
    "fast": {"num_inference_steps": 25, "guidance_scale": 7.5},
    "balanced": {"num_inference_steps": 50, "guidance_scale": 7.5},
    "high_quality": {"num_inference_steps": 100, "guidance_scale": 8.0}
}
```

## 📊 Performance

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (8GB) | RTX 4090 (24GB) |
| RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ |
| Python | 3.8+ | 3.10+ |

### Generation Times

| Duration | Resolution | GPU | Time |
|----------|------------|-----|------|
| 5s | 1080p | RTX 3060 | ~4 min |
| 10s | 1080p | RTX 4090 | ~6 min |
| 30s | 4K | RTX 4090 | ~45 min |

## 🎯 Examples

### Example 1: Superhero Scene
```
Prompt: "A superhero flying through a futuristic city at sunset, dramatic lighting, cinematic camera angles"
Duration: 10 seconds
Character Images: 5 reference images of the same person
Physics: Enabled
Result: High-quality video with consistent character appearance
```

### Example 2: Fantasy Scene
```
Prompt: "A wizard casting magical spells in a mystical forest, glowing particles, atmospheric lighting"
Duration: 8 seconds
Character Images: 3 reference images
Physics: Disabled (magical effects)
Result: Fantasy scene with consistent wizard character
```

### Example 3: Sci-Fi Scene
```
Prompt: "A robot dancing in a neon-lit cyberpunk street with rain, reflections, moody atmosphere"
Duration: 6 seconds
Character Images: 2 reference images
Physics: Enabled for realistic movement
Result: Cyberpunk scene with physics-based animation
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch size
   - Enable memory optimization
   - Use smaller model (1.3B instead of 14B)

2. **Character Inconsistency**:
   - Use more reference images (5-10)
   - Increase character strength
   - Ensure reference images are high quality

3. **Slow Generation**:
   - Use faster model (1.3B)
   - Reduce inference steps
   - Enable GPU optimizations

### Performance Tips

- Use SSD storage for faster I/O
- Close other GPU applications
- Monitor GPU memory usage
- Use appropriate model size for your hardware

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-repo/ai-movie-maker.git
cd ai-movie-maker

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Wan2.1](https://github.com/Wan-Video/Wan2.1) for video generation models
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) for character consistency
- [ThreeStudio](https://github.com/threestudio-project/threestudio) for 3D rendering
- [CodeLlama](https://github.com/meta-llama/codellama) for self-improvement

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/ai-movie-maker/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/ai-movie-maker/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/ai-movie-maker/wiki)

## 🔄 Changelog

### v1.0.0 (2024-01-XX)
- Initial release
- Wan2.1 integration
- IP-Adapter character consistency
- ThreeStudio physics simulation
- CodeLlama self-improvement
- Gradio web interface

---

**Made with ❤️ by the AI Movie Maker Team**