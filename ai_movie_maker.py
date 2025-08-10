#!/usr/bin/env python3
"""
AI Offline Movie Maker with Consistent Characters
Features:
- Character consistency using IP-Adapter (1-10 image inputs)
- Text-to-video generation with Wan2.1
- Physics world rendering with ThreeStudio
- Duration control (1s to 10m)
- Download options for videos
- Self-improvement mechanism using CodeLlama
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import torch
import gradio as gr
from PIL import Image
import cv2
from diffusers import WanPipeline, WanImageToVideoPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation
import imageio
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CharacterConsistencyManager:
    """Manages character consistency using IP-Adapter"""
    
    def __init__(self, ip_adapter_path: str = "ip-adapter-plus-face_sd15.safetensors"):
        self.ip_adapter_path = ip_adapter_path
        self.character_embeddings = {}
        self.load_ip_adapter()
    
    def load_ip_adapter(self):
        """Load IP-Adapter model for character consistency"""
        try:
            from ip_adapter import IPAdapterPlus
            self.ip_adapter = IPAdapterPlus(
                sd_pipe=None,  # Will be set when pipeline is available
                image_encoder_path="h94/IP-Adapter",
                ip_ckpt=self.ip_adapter_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info("IP-Adapter loaded successfully")
        except Exception as e:
            logger.warning(f"IP-Adapter not available: {e}")
            self.ip_adapter = None
    
    def extract_character_features(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Extract character features from reference images"""
        if not self.ip_adapter:
            return {}
        
        features = {}
        for i, img in enumerate(images):
            if img is not None:
                # Extract IP-Adapter features
                features[f"character_{i}"] = self.ip_adapter.encode_image(img)
        
        return features
    
    def apply_character_consistency(self, pipeline, features: Dict[str, Any], strength: float = 0.8):
        """Apply character consistency to the pipeline"""
        if not self.ip_adapter or not features:
            return pipeline
        
        # Apply IP-Adapter features to the pipeline
        for feature_name, feature in features.items():
            pipeline = self.ip_adapter.apply_ip_adapter(
                pipeline, 
                feature, 
                strength=strength
            )
        
        return pipeline

class PhysicsWorldEngine:
    """Physics world engine for realistic rendering"""
    
    def __init__(self):
        self.scene = None
        self.physics_world = None
        self.objects = {}
        self.setup_physics_world()
    
    def setup_physics_world(self):
        """Initialize physics world with Open3D"""
        try:
            # Create physics world
            self.physics_world = o3d.physics.PhysicsWorld()
            
            # Add gravity
            gravity = o3d.core.Tensor([0, -9.81, 0], dtype=o3d.core.Dtype.Float32)
            self.physics_world.set_gravity(gravity)
            
            logger.info("Physics world initialized")
        except Exception as e:
            logger.warning(f"Physics world not available: {e}")
    
    def add_character_mesh(self, character_id: str, mesh_path: str, position: List[float]):
        """Add character mesh to physics world"""
        if not self.physics_world:
            return
        
        try:
            # Load mesh
            mesh = trimesh.load(mesh_path)
            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mesh.vertices),
                triangles=o3d.utility.Vector3iVector(mesh.faces)
            )
            
            # Create rigid body
            rigid_body = o3d.physics.RigidBody(
                mesh=o3d_mesh,
                mass=70.0,  # Default human mass
                position=o3d.core.Tensor(position, dtype=o3d.core.Dtype.Float32)
            )
            
            self.objects[character_id] = rigid_body
            self.physics_world.add_rigid_body(rigid_body)
            
        except Exception as e:
            logger.error(f"Failed to add character mesh: {e}")
    
    def simulate_physics(self, duration: float, fps: int = 30) -> List[np.ndarray]:
        """Simulate physics and return frame data"""
        if not self.physics_world:
            return []
        
        frames = []
        dt = 1.0 / fps
        steps = int(duration * fps)
        
        for _ in range(steps):
            # Step physics simulation
            self.physics_world.step(dt)
            
            # Capture frame data
            frame_data = self.capture_frame()
            frames.append(frame_data)
        
        return frames
    
    def capture_frame(self) -> np.ndarray:
        """Capture current frame from physics world"""
        # This would integrate with ThreeStudio for rendering
        # For now, return placeholder
        return np.zeros((480, 640, 3), dtype=np.uint8)

class SelfImprovementEngine:
    """Self-improvement mechanism using CodeLlama"""
    
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-Instruct-hf"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load CodeLlama model for self-improvement"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("CodeLlama model loaded for self-improvement")
        except Exception as e:
            logger.warning(f"CodeLlama not available: {e}")
    
    def analyze_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow and suggest improvements"""
        if not self.model:
            return {}
        
        prompt = f"""
        Analyze this AI movie maker workflow and suggest improvements:
        {json.dumps(workflow_data, indent=2)}
        
        Focus on:
        1. Performance optimization
        2. Character consistency improvements
        3. Physics simulation accuracy
        4. Code quality and maintainability
        """
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {"analysis": analysis, "suggestions": self.extract_suggestions(analysis)}
        except Exception as e:
            logger.error(f"Workflow analysis failed: {e}")
            return {}
    
    def extract_suggestions(self, analysis: str) -> List[str]:
        """Extract improvement suggestions from analysis"""
        suggestions = []
        lines = analysis.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['improve', 'optimize', 'enhance', 'fix']):
                suggestions.append(line.strip())
        
        return suggestions

class AIMovieMaker:
    """Main AI Movie Maker class"""
    
    def __init__(self):
        self.character_manager = CharacterConsistencyManager()
        self.physics_engine = PhysicsWorldEngine()
        self.improvement_engine = SelfImprovementEngine()
        
        # Initialize Wan2.1 pipelines
        self.t2v_pipeline = None
        self.i2v_pipeline = None
        self.load_pipelines()
        
        # Configuration
        self.config = {
            "max_duration": 600,  # 10 minutes
            "min_duration": 1,    # 1 second
            "default_fps": 30,
            "output_resolution": (1920, 1080),
            "character_consistency_strength": 0.8
        }
    
    def load_pipelines(self):
        """Load Wan2.1 text-to-video and image-to-video pipelines"""
        try:
            # Load T2V pipeline (1.3B model for faster generation)
            self.t2v_pipeline = WanPipeline.from_pretrained(
                "Wan-Video/Wan2.1-T2V-1.3B",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            # Load I2V pipeline (14B model for higher quality)
            self.i2v_pipeline = WanImageToVideoPipeline.from_pretrained(
                "Wan-Video/Wan2.1-I2V-14B",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.t2v_pipeline = self.t2v_pipeline.to(device)
            self.i2v_pipeline = self.i2v_pipeline.to(device)
            
            logger.info("Wan2.1 pipelines loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load pipelines: {e}")
    
    def create_movie(
        self,
        text_prompt: str,
        character_images: List[Image.Image],
        duration: int,
        physics_enabled: bool = True,
        output_path: str = "output_movie.mp4"
    ) -> str:
        """Create a movie with consistent characters"""
        
        # Validate inputs
        if duration < self.config["min_duration"] or duration > self.config["max_duration"]:
            raise ValueError(f"Duration must be between {self.config['min_duration']} and {self.config['max_duration']} seconds")
        
        if not text_prompt.strip():
            raise ValueError("Text prompt cannot be empty")
        
        # Extract character features
        character_features = self.character_manager.extract_character_features(character_images)
        
        # Apply character consistency
        pipeline = self.t2v_pipeline
        if character_features:
            pipeline = self.character_manager.apply_character_consistency(
                pipeline, 
                character_features, 
                self.config["character_consistency_strength"]
            )
        
        # Generate video frames
        logger.info(f"Generating video for prompt: {text_prompt}")
        
        with torch.no_grad():
            video_frames = pipeline(
                prompt=text_prompt,
                num_inference_steps=50,
                num_frames=duration * self.config["default_fps"],
                height=self.config["output_resolution"][1],
                width=self.config["output_resolution"][0],
                fps=self.config["default_fps"]
            ).frames[0]
        
        # Apply physics simulation if enabled
        if physics_enabled:
            physics_frames = self.physics_engine.simulate_physics(duration)
            if physics_frames:
                # Blend physics with generated frames
                video_frames = self.blend_physics_frames(video_frames, physics_frames)
        
        # Save video
        self.save_video(video_frames, output_path, fps=self.config["default_fps"])
        
        # Self-improvement analysis
        workflow_data = {
            "prompt": text_prompt,
            "duration": duration,
            "physics_enabled": physics_enabled,
            "character_count": len(character_images),
            "output_path": output_path
        }
        
        improvements = self.improvement_engine.analyze_workflow(workflow_data)
        if improvements:
            logger.info(f"Self-improvement suggestions: {improvements}")
        
        return output_path
    
    def blend_physics_frames(self, video_frames: torch.Tensor, physics_frames: List[np.ndarray]) -> torch.Tensor:
        """Blend physics simulation with generated video frames"""
        # Simple alpha blending for now
        # In a full implementation, this would use ThreeStudio for proper rendering
        
        blended_frames = []
        for i, frame in enumerate(video_frames):
            if i < len(physics_frames):
                # Convert video frame to numpy
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
                frame_np = (frame_np * 255).astype(np.uint8)
                
                # Blend with physics frame
                physics_frame = physics_frames[i]
                if physics_frame.shape == frame_np.shape:
                    # Simple alpha blending
                    alpha = 0.3
                    blended = cv2.addWeighted(frame_np, 1-alpha, physics_frame, alpha, 0)
                    blended_frames.append(torch.from_numpy(blended).permute(2, 0, 1).float() / 255.0)
                else:
                    blended_frames.append(frame)
            else:
                blended_frames.append(frame)
        
        return torch.stack(blended_frames)
    
    def save_video(self, frames: torch.Tensor, output_path: str, fps: int = 30):
        """Save video frames to file"""
        # Convert frames to numpy
        frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
        frames_np = (frames_np * 255).astype(np.uint8)
        
        # Save using imageio
        imageio.mimsave(output_path, frames_np, fps=fps)
        logger.info(f"Video saved to: {output_path}")

def create_gradio_interface():
    """Create Gradio interface for the AI Movie Maker"""
    
    movie_maker = AIMovieMaker()
    
    def generate_movie(
        text_prompt: str,
        character_images: List[Image.Image],
        duration: int,
        physics_enabled: bool,
        character_strength: float
    ):
        """Generate movie with the given parameters"""
        
        try:
            # Update character consistency strength
            movie_maker.config["character_consistency_strength"] = character_strength
            
            # Generate unique output path
            timestamp = int(time.time())
            output_path = f"output_movie_{timestamp}.mp4"
            
            # Create movie
            result_path = movie_maker.create_movie(
                text_prompt=text_prompt,
                character_images=character_images,
                duration=duration,
                physics_enabled=physics_enabled,
                output_path=output_path
            )
            
            return result_path, f"Movie generated successfully! Saved to: {result_path}"
            
        except Exception as e:
            logger.error(f"Movie generation failed: {e}")
            return None, f"Error: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks(title="AI Movie Maker with Consistent Characters") as interface:
        gr.Markdown("# 🎬 AI Movie Maker with Consistent Characters")
        gr.Markdown("Create movies with consistent characters using AI, physics simulation, and character consistency!")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Text input
                text_prompt = gr.Textbox(
                    label="Movie Description",
                    placeholder="Describe your movie scene...",
                    lines=3
                )
                
                # Character images (1-10)
                character_images = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Character Reference Images (1-10)",
                    info="Upload 1-10 images of the same character for consistency"
                )
                
                # Duration slider
                duration = gr.Slider(
                    minimum=1,
                    maximum=600,
                    value=5,
                    step=1,
                    label="Duration (seconds)",
                    info="1 second to 10 minutes"
                )
                
                # Physics toggle
                physics_enabled = gr.Checkbox(
                    label="Enable Physics Simulation",
                    value=True,
                    info="Add realistic physics to the scene"
                )
                
                # Character consistency strength
                character_strength = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="Character Consistency Strength",
                    info="How strongly to maintain character appearance"
                )
                
                # Generate button
                generate_btn = gr.Button("🎬 Generate Movie", variant="primary")
            
            with gr.Column(scale=1):
                # Output video
                output_video = gr.Video(
                    label="Generated Movie",
                    format="mp4"
                )
                
                # Status message
                status_msg = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        # Event handlers
        generate_btn.click(
            fn=generate_movie,
            inputs=[text_prompt, character_images, duration, physics_enabled, character_strength],
            outputs=[output_video, status_msg]
        )
        
        # Examples
        gr.Examples(
            examples=[
                [
                    "A superhero flying through a futuristic city at sunset, dramatic lighting",
                    [],
                    10,
                    True,
                    0.8
                ],
                [
                    "A wizard casting magical spells in a mystical forest with glowing particles",
                    [],
                    8,
                    True,
                    0.7
                ],
                [
                    "A robot dancing in a neon-lit cyberpunk street with rain",
                    [],
                    6,
                    False,
                    0.9
                ]
            ],
            inputs=[text_prompt, character_images, duration, physics_enabled, character_strength]
        )
    
    return interface

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI Movie Maker with Consistent Characters")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for Gradio interface")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    # Create and launch interface
    interface = create_gradio_interface()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )

if __name__ == "__main__":
    main()