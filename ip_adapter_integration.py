#!/usr/bin/env python3
"""
IP-Adapter Integration for Character Consistency
Provides seamless integration with IP-Adapter for maintaining character appearance
across video frames in AI movie generation.
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import logging

logger = logging.getLogger(__name__)

class IPAdapterPlus:
    """Enhanced IP-Adapter for character consistency"""
    
    def __init__(
        self,
        sd_pipe,
        image_encoder_path: str = "h94/IP-Adapter",
        ip_ckpt: str = "ip-adapter-plus-face_sd15.safetensors",
        device: str = "cuda"
    ):
        self.sd_pipe = sd_pipe
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        
        # Initialize components
        self.image_encoder = None
        self.image_proj = None
        self.ip_layers = None
        
        self.load_ip_adapter()
    
    def load_ip_adapter(self):
        """Load IP-Adapter model and components"""
        try:
            # Load image encoder
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.image_encoder_path,
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Load image processor
            self.image_processor = CLIPImageProcessor.from_pretrained(self.image_encoder_path)
            
            # Load IP-Adapter weights
            self.load_ip_adapter_weights()
            
            logger.info("IP-Adapter loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load IP-Adapter: {e}")
            raise
    
    def load_ip_adapter_weights(self):
        """Load IP-Adapter weights from checkpoint"""
        try:
            from safetensors import safe_open
            
            with safe_open(self.ip_ckpt, framework="pt", device=self.device) as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
            
            # Extract image projection weights
            image_proj_state_dict = {}
            ip_layers_state_dict = {}
            
            for key, value in state_dict.items():
                if key.startswith("image_proj."):
                    image_proj_state_dict[key.replace("image_proj.", "")] = value
                elif key.startswith("ip_adapter."):
                    ip_layers_state_dict[key.replace("ip_adapter.", "")] = value
            
            # Create image projection layer
            self.image_proj = self.create_image_projection(image_proj_state_dict)
            
            # Store IP layers weights
            self.ip_layers = ip_layers_state_dict
            
            logger.info("IP-Adapter weights loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load IP-Adapter weights: {e}")
            raise
    
    def create_image_projection(self, state_dict: Dict[str, torch.Tensor]):
        """Create image projection layer from state dict"""
        # This is a simplified version - in practice, you'd need to match the exact architecture
        input_dim = state_dict.get("proj_in.weight", torch.randn(768, 768)).shape[1]
        output_dim = state_dict.get("proj_out.weight", torch.randn(768, 768)).shape[0]
        
        class ImageProjection(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.proj_in = torch.nn.Linear(input_dim, output_dim)
                self.proj_out = torch.nn.Linear(output_dim, output_dim)
                self.norm = torch.nn.LayerNorm(output_dim)
                
            def forward(self, x):
                x = self.proj_in(x)
                x = self.norm(x)
                x = self.proj_out(x)
                return x
        
        projection = ImageProjection(input_dim, output_dim).to(self.device)
        
        # Load weights
        for name, param in projection.named_parameters():
            if name in state_dict:
                param.data = state_dict[name]
        
        return projection
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to IP-Adapter features"""
        try:
            # Preprocess image
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Process image
            inputs = self.image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Encode with CLIP vision model
            with torch.no_grad():
                image_embeds = self.image_encoder(**inputs).image_embeds
            
            # Project to IP-Adapter space
            ip_image_embeds = self.image_proj(image_embeds)
            
            return ip_image_embeds
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None
    
    def apply_ip_adapter(
        self, 
        pipeline, 
        image_embeds: torch.Tensor, 
        strength: float = 0.8,
        start_at: float = 0.0,
        end_at: float = 1.0
    ):
        """Apply IP-Adapter to the pipeline"""
        if image_embeds is None:
            return pipeline
        
        # Store IP-Adapter features in pipeline
        pipeline.ip_adapter_image_embeds = image_embeds
        pipeline.ip_adapter_strength = strength
        pipeline.ip_adapter_start_at = start_at
        pipeline.ip_adapter_end_at = end_at
        
        # Patch UNet to use IP-Adapter
        self.patch_unet(pipeline.unet)
        
        return pipeline
    
    def patch_unet(self, unet: UNet2DConditionModel):
        """Patch UNet to inject IP-Adapter features"""
        original_forward = unet.forward
        
        def patched_forward(sample, timestep, encoder_hidden_states, **kwargs):
            # Get IP-Adapter features from pipeline
            pipeline = kwargs.get('pipeline', None)
            if pipeline and hasattr(pipeline, 'ip_adapter_image_embeds'):
                ip_embeds = pipeline.ip_adapter_image_embeds
                strength = getattr(pipeline, 'ip_adapter_strength', 0.8)
                start_at = getattr(pipeline, 'ip_adapter_start_at', 0.0)
                end_at = getattr(pipeline, 'ip_adapter_end_at', 1.0)
                
                # Calculate current timestep progress
                timestep_progress = timestep.float() / 1000.0  # Normalize to 0-1
                
                # Apply IP-Adapter if within range
                if start_at <= timestep_progress <= end_at:
                    # Inject IP-Adapter features into cross-attention
                    encoder_hidden_states = self.inject_ip_features(
                        encoder_hidden_states, 
                        ip_embeds, 
                        strength
                    )
            
            return original_forward(sample, timestep, encoder_hidden_states, **kwargs)
        
        unet.forward = patched_forward
    
    def inject_ip_features(
        self, 
        encoder_hidden_states: torch.Tensor, 
        ip_embeds: torch.Tensor, 
        strength: float
    ) -> torch.Tensor:
        """Inject IP-Adapter features into encoder hidden states"""
        # Concatenate IP-Adapter features with text features
        if ip_embeds.dim() == 2:
            ip_embeds = ip_embeds.unsqueeze(1)  # Add sequence dimension
        
        # Blend features based on strength
        blended_embeds = torch.cat([
            encoder_hidden_states,
            ip_embeds * strength
        ], dim=1)
        
        return blended_embeds

class CharacterConsistencyEnhancer:
    """Enhanced character consistency with multiple reference images"""
    
    def __init__(self, ip_adapter: IPAdapterPlus):
        self.ip_adapter = ip_adapter
        self.character_embeddings = {}
    
    def add_character_references(
        self, 
        character_id: str, 
        images: List[Image.Image],
        weights: Optional[List[float]] = None
    ):
        """Add multiple reference images for a character"""
        if not images:
            return
        
        # Default equal weights if not provided
        if weights is None:
            weights = [1.0 / len(images)] * len(images)
        
        # Encode all images
        embeddings = []
        for img, weight in zip(images, weights):
            if img is not None:
                embedding = self.ip_adapter.encode_image(img)
                if embedding is not None:
                    embeddings.append(embedding * weight)
        
        if embeddings:
            # Average embeddings
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            self.character_embeddings[character_id] = avg_embedding
    
    def get_character_embedding(self, character_id: str) -> Optional[torch.Tensor]:
        """Get character embedding by ID"""
        return self.character_embeddings.get(character_id)
    
    def apply_character_consistency(
        self, 
        pipeline, 
        character_ids: List[str],
        strength: float = 0.8
    ):
        """Apply character consistency to pipeline"""
        if not character_ids:
            return pipeline
        
        # Combine all character embeddings
        all_embeddings = []
        for char_id in character_ids:
            embedding = self.get_character_embedding(char_id)
            if embedding is not None:
                all_embeddings.append(embedding)
        
        if all_embeddings:
            # Average all character embeddings
            combined_embedding = torch.stack(all_embeddings).mean(dim=0)
            
            # Apply to pipeline
            pipeline = self.ip_adapter.apply_ip_adapter(
                pipeline, 
                combined_embedding, 
                strength=strength
            )
        
        return pipeline

class IPAdapterOptimizer:
    """Optimization utilities for IP-Adapter"""
    
    @staticmethod
    def optimize_image_for_consistency(image: Image.Image) -> Image.Image:
        """Optimize image for better character consistency"""
        # Resize to optimal size for IP-Adapter
        optimal_size = (224, 224)
        if image.size != optimal_size:
            image = image.resize(optimal_size, Image.LANCZOS)
        
        # Enhance contrast slightly
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    @staticmethod
    def extract_face_region(image: Image.Image) -> Optional[Image.Image]:
        """Extract face region for better character consistency"""
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Load face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detect faces
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Add padding
                padding = int(min(w, h) * 0.2)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_cv.shape[1] - x, w + 2 * padding)
                h = min(img_cv.shape[0] - y, h + 2 * padding)
                
                # Crop face region
                face_img = img_cv[y:y+h, x:x+w]
                
                # Convert back to PIL
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                
                return face_pil
            
        except Exception as e:
            logger.warning(f"Face extraction failed: {e}")
        
        return None
    
    @staticmethod
    def create_character_variations(
        base_image: Image.Image, 
        num_variations: int = 3
    ) -> List[Image.Image]:
        """Create variations of a character image for better consistency"""
        variations = [base_image]
        
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Brightness variations
            enhancer = ImageEnhance.Brightness(base_image)
            variations.append(enhancer.enhance(1.1))
            variations.append(enhancer.enhance(0.9))
            
            # Slight blur for robustness
            variations.append(base_image.filter(ImageFilter.GaussianBlur(radius=0.5)))
            
            # Return requested number of variations
            return variations[:num_variations]
            
        except Exception as e:
            logger.warning(f"Failed to create variations: {e}")
            return [base_image]

# Utility functions
def load_ip_adapter_model(
    model_path: str = "ip-adapter-plus-face_sd15.safetensors",
    device: str = "cuda"
) -> Optional[IPAdapterPlus]:
    """Load IP-Adapter model with error handling"""
    try:
        ip_adapter = IPAdapterPlus(
            sd_pipe=None,  # Will be set later
            ip_ckpt=model_path,
            device=device
        )
        return ip_adapter
    except Exception as e:
        logger.error(f"Failed to load IP-Adapter model: {e}")
        return None

def validate_character_images(images: List[Image.Image]) -> List[Image.Image]:
    """Validate and preprocess character images"""
    valid_images = []
    
    for img in images:
        if img is not None:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Optimize for consistency
            img = IPAdapterOptimizer.optimize_image_for_consistency(img)
            valid_images.append(img)
    
    return valid_images

def create_character_embedding(
    images: List[Image.Image],
    ip_adapter: IPAdapterPlus,
    use_face_extraction: bool = True
) -> Optional[torch.Tensor]:
    """Create character embedding from multiple images"""
    if not images or not ip_adapter:
        return None
    
    embeddings = []
    
    for img in images:
        # Extract face if requested
        if use_face_extraction:
            face_img = IPAdapterOptimizer.extract_face_region(img)
            if face_img:
                img = face_img
        
        # Encode image
        embedding = ip_adapter.encode_image(img)
        if embedding is not None:
            embeddings.append(embedding)
    
    if embeddings:
        # Average all embeddings
        return torch.stack(embeddings).mean(dim=0)
    
    return None