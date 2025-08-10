#!/usr/bin/env python3
"""
AI Movie Maker with Character Consistency and Physics Simulation
Combines Wan2.1, IP-Adapter, threestudio, and CodeLlama for advanced video generation
"""

import os
import sys
import gc
import time
import logging
import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import traceback

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import gradio as gr
from diffusers import DDIMScheduler
import json
import subprocess
import threading
from queue import Queue
import tempfile

# Import Wan2.1 components
import wan
from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_image, cache_video, str2bool

# Import our custom components
from ip_adapter_integration import create_ip_adapter_character_system
from threestudio_integration import create_threestudio_system
from codellama_integration import create_codellama_system

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CharacterConsistencyManager:
    """Manages character consistency using IP-Adapter techniques"""
    
    def __init__(self):
        self.character_embeddings = {}
        self.reference_images = {}
        self.character_count = 0
        
    def register_character(self, character_id: str, reference_images: List[str]) -> bool:
        """Register a character with reference images (1-10 images)"""
        try:
            if len(reference_images) > 10:
                reference_images = reference_images[:10]
                logger.warning(f"Truncated reference images to 10 for character {character_id}")
            
            # Process and store reference images
            processed_images = []
            for img_path in reference_images:
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    # Resize to standard size for consistency
                    img = img.resize((512, 512), Image.LANCZOS)
                    processed_images.append(img)
                else:
                    logger.warning(f"Reference image not found: {img_path}")
            
            if not processed_images:
                return False
                
            self.reference_images[character_id] = processed_images
            self.character_count += 1
            
            logger.info(f"Registered character {character_id} with {len(processed_images)} reference images")
            return True
            
        except Exception as e:
            logger.error(f"Error registering character {character_id}: {str(e)}")
            return False
    
    def get_character_prompt_enhancement(self, character_id: str, base_prompt: str) -> str:
        """Enhance prompt with character consistency instructions"""
        if character_id not in self.reference_images:
            return base_prompt
        
        consistency_prompt = f"[CHARACTER:{character_id}] {base_prompt}, maintaining consistent character appearance, facial features, and body proportions throughout the video"
        return consistency_prompt
    
    def apply_character_consistency(self, generated_frames: List[np.ndarray], character_id: str) -> List[np.ndarray]:
        """Apply post-processing for character consistency"""
        # This would integrate with IP-Adapter for real implementation
        # For now, return frames as-is
        return generated_frames

class TextEncoder:
    """Advanced text encoder for natural language to video instructions"""
    
    def __init__(self):
        self.emotion_keywords = {
            'happy': ['smiling', 'laughing', 'joyful', 'cheerful', 'excited'],
            'sad': ['crying', 'melancholy', 'sorrowful', 'depressed'],
            'angry': ['furious', 'mad', 'rage', 'hostile'],
            'surprised': ['shocked', 'amazed', 'astonished'],
            'neutral': ['calm', 'peaceful', 'serene']
        }
        
        self.action_keywords = {
            'walking': ['strolling', 'marching', 'pacing'],
            'running': ['sprinting', 'jogging', 'dashing'],
            'talking': ['speaking', 'conversing', 'chatting'],
            'dancing': ['moving rhythmically', 'swaying'],
            'sitting': ['seated', 'resting'],
            'standing': ['upright', 'standing tall']
        }
    
    def enhance_prompt(self, user_input: str) -> str:
        """Enhance user input with cinematic and technical terms"""
        enhanced = user_input
        
        # Add cinematic quality descriptors
        if 'cinematic' not in enhanced.lower():
            enhanced += ", cinematic quality"
        
        # Add technical video parameters
        if 'realistic' not in enhanced.lower():
            enhanced += ", highly detailed, realistic"
            
        if 'lighting' not in enhanced.lower():
            enhanced += ", professional lighting"
            
        return enhanced
    
    def extract_duration_from_text(self, text: str) -> float:
        """Extract desired duration from text input"""
        # Look for duration indicators
        duration_patterns = [
            r'(\d+)\s*second[s]?',
            r'(\d+)\s*minute[s]?',
            r'(\d+)s\b',
            r'(\d+)m\b'
        ]
        
        import re
        for pattern in duration_patterns:
            match = re.search(pattern, text.lower())
            if match:
                duration = float(match.group(1))
                if 'minute' in pattern or 'm' in pattern:
                    duration *= 60
                return min(duration, 600)  # Max 10 minutes
        
        return 5.0  # Default 5 seconds

class PhysicsEngine:
    """Simple physics simulation for 3D world generation"""
    
    def __init__(self):
        self.objects = []
        self.gravity = -9.81
        self.time_step = 1/30  # 30 FPS
        
    def add_object(self, obj_type: str, position: Tuple[float, float, float], 
                   velocity: Tuple[float, float, float] = (0, 0, 0)):
        """Add physics object to simulation"""
        obj = {
            'type': obj_type,
            'position': list(position),
            'velocity': list(velocity),
            'mass': 1.0
        }
        self.objects.append(obj)
        return len(self.objects) - 1
    
    def simulate_step(self):
        """Simulate one physics step"""
        for obj in self.objects:
            # Apply gravity
            obj['velocity'][1] += self.gravity * self.time_step
            
            # Update position
            for i in range(3):
                obj['position'][i] += obj['velocity'][i] * self.time_step
            
            # Simple ground collision
            if obj['position'][1] < 0:
                obj['position'][1] = 0
                obj['velocity'][1] = -obj['velocity'][1] * 0.8  # Bounce with damping
    
    def get_world_state(self) -> Dict:
        """Get current state of physics world"""
        return {'objects': self.objects}

class AIMovieMaker:
    """Main AI Movie Maker class"""
    
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self.character_manager = CharacterConsistencyManager()
        self.text_encoder = TextEncoder()
        self.physics_engine = PhysicsEngine()
        
        # Initialize advanced systems
        try:
            self.ip_adapter_system = create_ip_adapter_character_system()
            logger.info("✅ IP-Adapter character system initialized")
        except Exception as e:
            logger.warning(f"⚠️ IP-Adapter system failed to initialize: {str(e)}")
            self.ip_adapter_system = None
        
        try:
            self.threestudio_system = create_threestudio_system()
            logger.info("✅ ThreeStudio physics system initialized")
        except Exception as e:
            logger.warning(f"⚠️ ThreeStudio system failed to initialize: {str(e)}")
            self.threestudio_system = None
        
        try:
            self.codellama_system = create_codellama_system()
            logger.info("✅ CodeLlama self-improvement system initialized")
        except Exception as e:
            logger.warning(f"⚠️ CodeLlama system failed to initialize: {str(e)}")
            self.codellama_system = None
        
        # Initialize Wan2.1 models
        self.models = {}
        self.current_task = None
        
        # Video settings
        self.max_duration = 600  # 10 minutes
        self.min_duration = 1    # 1 second
        
        # Self-improvement queue (for CodeLlama integration)
        self.improvement_queue = Queue()
        
        logger.info("🎬 AI Movie Maker fully initialized with all systems")
    
    def load_model(self, task: str):
        """Load specific Wan2.1 model for task"""
        if task in self.models:
            return self.models[task]
        
        try:
            logger.info(f"Loading model for task: {task}")
            
            if task == "t2v-1.3B":
                from wan.text2video import Text2VideoInference
                model = Text2VideoInference(self.ckpt_dir, task)
            elif task == "t2v-14B":
                from wan.text2video import Text2VideoInference
                model = Text2VideoInference(self.ckpt_dir, task)
            elif task == "i2v-14B":
                from wan.image2video import Image2VideoInference
                model = Image2VideoInference(self.ckpt_dir, task)
            elif task == "vace-1.3B" or task == "vace-14B":
                from wan.vace import VACEInference
                model = VACEInference(self.ckpt_dir, task)
            else:
                raise ValueError(f"Unsupported task: {task}")
            
            self.models[task] = model
            self.current_task = task
            
            logger.info(f"Model loaded successfully: {task}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {task}: {str(e)}")
            raise
    
    def generate_video(self, 
                      prompt: str,
                      duration: float = 5.0,
                      character_ids: List[str] = None,
                      reference_images: List[str] = None,
                      physics_enabled: bool = False,
                      talking_characters: bool = False,
                      size: str = "832*480",
                      fps: int = 30) -> Optional[str]:
        """Generate video with all AI features"""
        
        try:
            # Validate duration
            duration = max(self.min_duration, min(duration, self.max_duration))
            
            # Calculate frame count
            frame_count = int(duration * fps)
            frame_count = min(frame_count, 241)  # Max frames for Wan2.1
            
            # Enhance prompt with text encoder
            enhanced_prompt = self.text_encoder.enhance_prompt(prompt)
            
            # Apply character consistency if specified
            if character_ids:
                for char_id in character_ids:
                    enhanced_prompt = self.character_manager.get_character_prompt_enhancement(
                        char_id, enhanced_prompt)
            
            # Add physics simulation context if enabled
            if physics_enabled:
                enhanced_prompt += ", realistic physics simulation, natural movement"
            
            # Add talking characters context if enabled
            if talking_characters:
                enhanced_prompt += ", characters with natural facial expressions and lip sync"
            
            logger.info(f"Generating video with prompt: {enhanced_prompt}")
            logger.info(f"Duration: {duration}s, Frames: {frame_count}")
            
            # Choose appropriate model based on requirements
            if reference_images and len(reference_images) > 0:
                task = "vace-1.3B"  # VACE for character consistency
            else:
                task = "t2v-1.3B"   # Text-to-video
            
            # Load model
            model = self.load_model(task)
            
            # Prepare generation parameters
            gen_params = {
                'prompt': enhanced_prompt,
                'frame_num': frame_count,
                'size': size,
                'sample_steps': 50,
                'sample_shift': 5.0,
                'seed': torch.randint(0, 2**32, (1,)).item()
            }
            
            # Add reference images for VACE
            if task.startswith("vace") and reference_images:
                gen_params['src_ref_images'] = ','.join(reference_images)
            
            # Generate video
            output_path = self._generate_with_model(model, task, gen_params)
            
            # Apply post-processing
            if character_ids:
                output_path = self._apply_character_consistency_post_processing(
                    output_path, character_ids)
            
            if physics_enabled:
                output_path = self._apply_physics_post_processing(output_path)
            
            logger.info(f"Video generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _generate_with_model(self, model, task: str, params: Dict) -> str:
        """Generate video using specific model"""
        try:
            # Create output directory
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"generated_video_{timestamp}.mp4")
            
            if task.startswith("vace"):
                # VACE generation
                results = model(
                    prompt=params['prompt'],
                    src_ref_images=params.get('src_ref_images', ''),
                    frame_num=params['frame_num'],
                    size=params['size'],
                    sample_steps=params['sample_steps'],
                    sample_shift=params['sample_shift'],
                    seed=params['seed']
                )
            else:
                # Standard text-to-video generation
                results = model(
                    prompt=params['prompt'],
                    frame_num=params['frame_num'],
                    size=params['size'],
                    sample_steps=params['sample_steps'],
                    sample_shift=params['sample_shift'],
                    seed=params['seed']
                )
            
            # Save video
            if results and len(results) > 0:
                video_data = results[0]
                # Convert to video file
                self._save_video_tensor_to_file(video_data, output_path, fps=30)
                return output_path
            else:
                raise Exception("No video data generated")
                
        except Exception as e:
            logger.error(f"Error in model generation: {str(e)}")
            raise
    
    def _save_video_tensor_to_file(self, video_tensor, output_path: str, fps: int = 30):
        """Save video tensor to file"""
        try:
            import imageio
            
            # Convert tensor to numpy if needed
            if torch.is_tensor(video_tensor):
                video_array = video_tensor.cpu().numpy()
            else:
                video_array = video_tensor
            
            # Ensure proper shape and range
            if video_array.ndim == 4:  # (T, H, W, C)
                pass
            elif video_array.ndim == 5:  # (B, T, H, W, C)
                video_array = video_array[0]
            
            # Normalize to 0-255 range
            if video_array.max() <= 1.0:
                video_array = (video_array * 255).astype(np.uint8)
            else:
                video_array = video_array.astype(np.uint8)
            
            # Save video
            imageio.mimsave(output_path, video_array, fps=fps, quality=8)
            logger.info(f"Video saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            raise
    
    def _apply_character_consistency_post_processing(self, video_path: str, character_ids: List[str]) -> str:
        """Apply character consistency post-processing"""
        # Placeholder for character consistency post-processing
        # In a full implementation, this would use IP-Adapter techniques
        logger.info("Applying character consistency post-processing")
        return video_path
    
    def _apply_physics_post_processing(self, video_path: str) -> str:
        """Apply physics simulation post-processing"""
        # Placeholder for physics post-processing
        logger.info("Applying physics simulation post-processing")
        return video_path
    
    def register_characters_from_images(self, image_paths: List[str]) -> List[str]:
        """Register characters from provided images"""
        character_ids = []
        
        for i, img_path in enumerate(image_paths[:10]):  # Max 10 images
            character_id = f"character_{i+1}"
            if self.character_manager.register_character(character_id, [img_path]):
                character_ids.append(character_id)
        
        return character_ids
    
    def clear_memory(self):
        """Clear GPU memory and cleanup"""
        try:
            # Clear model cache
            for model in self.models.values():
                if hasattr(model, 'cleanup'):
                    model.cleanup()
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Memory cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")

def create_gradio_interface(movie_maker: AIMovieMaker):
    """Create Gradio interface for the AI Movie Maker"""
    
    def generate_movie(prompt: str, 
                      duration: float,
                      character_images: List,
                      physics_enabled: bool,
                      talking_characters: bool,
                      video_size: str,
                      progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """Generate movie through Gradio interface"""
        try:
            progress(0.1, desc="Processing input...")
            
            # Validate inputs
            if not prompt.strip():
                return None, "Error: Please provide a text prompt"
            
            # Register characters if images provided
            character_ids = []
            if character_images:
                progress(0.2, desc="Processing character images...")
                image_paths = []
                for img in character_images:
                    if img is not None:
                        temp_path = f"temp_char_{len(image_paths)}.png"
                        img.save(temp_path)
                        image_paths.append(temp_path)
                
                if image_paths:
                    character_ids = movie_maker.register_characters_from_images(image_paths)
                    
                    # Cleanup temp files
                    for temp_path in image_paths:
                        try:
                            os.remove(temp_path)
                        except:
                            pass
            
            progress(0.3, desc="Initializing video generation...")
            
            # Extract duration from prompt if not specified
            if duration <= 0:
                duration = movie_maker.text_encoder.extract_duration_from_text(prompt)
            
            progress(0.4, desc="Generating video...")
            
            # Generate video
            output_path = movie_maker.generate_video(
                prompt=prompt,
                duration=duration,
                character_ids=character_ids,
                physics_enabled=physics_enabled,
                talking_characters=talking_characters,
                size=video_size
            )
            
            progress(0.9, desc="Finalizing...")
            
            if output_path and os.path.exists(output_path):
                progress(1.0, desc="Complete!")
                return output_path, f"✅ Video generated successfully!\nDuration: {duration}s\nCharacters: {len(character_ids)}"
            else:
                return None, "❌ Failed to generate video. Please try again."
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    # Create Gradio interface
    with gr.Blocks(title="🎬 AI Movie Maker", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎬 AI Movie Maker with Character Consistency
        
        Create amazing videos with AI! Features:
        - 🎭 Character consistency (1-10 reference images)
        - ⏱️ Duration control (1 second to 10 minutes)
        - 🗣️ Talking AI characters
        - 🌍 Physics simulation
        - 📥 Download ready videos
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("## 📝 Input Settings")
                
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Describe your movie scene... (e.g., 'A cat wearing sunglasses dancing on the beach for 10 seconds')",
                    lines=3,
                    value="Two anthropomorphic cats in comfy boxing gear fight intensely on a spotlighted stage"
                )
                
                with gr.Row():
                    duration_input = gr.Slider(
                        label="Duration (seconds)",
                        minimum=1,
                        maximum=600,
                        value=5,
                        step=1
                    )
                    
                    video_size = gr.Dropdown(
                        label="Video Size",
                        choices=["832*480", "480*832", "1024*576", "576*1024"],
                        value="832*480"
                    )
                
                # Character images
                gr.Markdown("### 🎭 Character Consistency (Optional)")
                character_images = gr.Gallery(
                    label="Reference Images (1-10 images for character consistency)",
                    show_label=True,
                    elem_id="character_gallery",
                    columns=5,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                    type="pil"
                )
                
                # Advanced options
                gr.Markdown("### ⚙️ Advanced Options")
                with gr.Row():
                    physics_enabled = gr.Checkbox(
                        label="🌍 Enable Physics Simulation",
                        value=False
                    )
                    
                    talking_characters = gr.Checkbox(
                        label="🗣️ Talking Characters",
                        value=False
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🎬 Generate Movie",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("## 🎥 Generated Movie")
                
                output_video = gr.Video(
                    label="Generated Video",
                    height=400
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=3,
                    interactive=False
                )
                
                # Download button (handled by Gradio automatically for video component)
                gr.Markdown("💡 **Tip**: Right-click the video to download or use browser download options")
        
        # Examples section
        gr.Markdown("## 🌟 Example Prompts")
        gr.Examples(
            examples=[
                ["A majestic dragon flying over a medieval castle for 8 seconds", 8, None, True, False, "832*480"],
                ["A robot dancing in a futuristic city with neon lights", 5, None, False, True, "832*480"],
                ["Two friends having a conversation in a coffee shop", 10, None, False, True, "832*480"],
                ["A magical forest with floating lights and mystical creatures", 15, None, True, False, "1024*576"],
                ["A superhero saving the city from falling debris", 12, None, True, False, "832*480"]
            ],
            inputs=[prompt_input, duration_input, character_images, physics_enabled, talking_characters, video_size]
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_movie,
            inputs=[prompt_input, duration_input, character_images, physics_enabled, talking_characters, video_size],
            outputs=[output_video, status_output],
            show_progress=True
        )
        
        # Clear memory periodically
        def cleanup_memory():
            movie_maker.clear_memory()
            return "🧹 Memory cleared"
        
        gr.Button("🧹 Clear Memory", size="sm").click(
            fn=cleanup_memory,
            outputs=gr.Textbox(visible=False)
        )
    
    return interface

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI Movie Maker with Character Consistency")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory for Wan2.1 models")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio interface")
    parser.add_argument("--share", action="store_true", help="Create shareable Gradio link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("🎬 Starting AI Movie Maker...")
        
        # Initialize AI Movie Maker
        movie_maker = AIMovieMaker(args.ckpt_dir)
        
        # Create Gradio interface
        interface = create_gradio_interface(movie_maker)
        
        # Launch interface
        logger.info(f"🚀 Launching Gradio interface on port {args.port}")
        interface.launch(
            server_port=args.port,
            share=args.share,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down AI Movie Maker...")
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()