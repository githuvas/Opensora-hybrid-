#!/usr/bin/env python3
"""
IP-Adapter Integration for AI Movie Maker
Enhanced character consistency using IP-Adapter models
"""

import os
import sys
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple, Union
import cv2

warnings.filterwarnings('ignore')

try:
    from diffusers import (
        StableDiffusionPipeline,
        DDIMScheduler,
        DPMSolverMultistepScheduler
    )
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Diffusers not available for IP-Adapter")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available")

class IPAdapterManager:
    """Manages IP-Adapter models for character consistency"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.ip_adapter = None
        self.image_encoder = None
        self.image_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.character_embeddings = {}
        self.face_embeddings = {}
        
        self.setup_ip_adapter()
    
    def setup_ip_adapter(self):
        """Setup IP-Adapter model"""
        if not DIFFUSERS_AVAILABLE:
            print("Diffusers not available, skipping IP-Adapter setup")
            return
        
        try:
            # Load IP-Adapter model
            if self.model_path and os.path.exists(self.model_path):
                # Load custom IP-Adapter
                self.ip_adapter = torch.load(self.model_path, map_location=self.device)
                print(f"✓ Loaded custom IP-Adapter from {self.model_path}")
            else:
                # Try to load from HuggingFace
                try:
                    from ip_adapter import IPAdapter
                    self.ip_adapter = IPAdapter.from_pretrained(
                        "h94/IP-Adapter",
                        subfolder="models",
                        device_map=self.device
                    )
                    print("✓ Loaded IP-Adapter from HuggingFace")
                except Exception as e:
                    print(f"✗ Failed to load IP-Adapter: {e}")
            
            # Setup image encoder
            if CLIP_AVAILABLE:
                self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    "h94/IP-Adapter",
                    subfolder="models/image_encoder",
                    device_map=self.device
                )
                self.image_processor = CLIPImageProcessor.from_pretrained(
                    "h94/IP-Adapter",
                    subfolder="models/image_encoder"
                )
                print("✓ Image encoder loaded")
        
        except Exception as e:
            print(f"✗ Failed to setup IP-Adapter: {e}")
    
    def extract_character_embedding(self, image_path: str, character_name: str) -> Optional[torch.Tensor]:
        """Extract character embedding from reference image"""
        if not self.image_encoder or not self.image_processor:
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_inputs = self.image_processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                image_embeds = self.image_encoder(**image_inputs).image_embeds
                image_embeds = F.normalize(image_embeds, dim=-1)
            
            # Store embedding
            self.character_embeddings[character_name] = image_embeds
            
            print(f"✓ Extracted embedding for character: {character_name}")
            return image_embeds
        
        except Exception as e:
            print(f"✗ Failed to extract embedding for {character_name}: {e}")
            return None
    
    def extract_face_embedding(self, image_path: str, face_id: str) -> Optional[torch.Tensor]:
        """Extract face-specific embedding"""
        if not self.image_encoder or not self.image_processor:
            return None
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Detect and crop face (simplified)
            # In a full implementation, use face detection library
            face_crop = self.crop_face_region(image)
            if face_crop is None:
                return None
            
            # Process face crop
            image_inputs = self.image_processor(
                images=face_crop,
                return_tensors="pt"
            ).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                face_embeds = self.image_encoder(**image_inputs).image_embeds
                face_embeds = F.normalize(face_embeds, dim=-1)
            
            # Store embedding
            self.face_embeddings[face_id] = face_embeds
            
            print(f"✓ Extracted face embedding for: {face_id}")
            return face_embeds
        
        except Exception as e:
            print(f"✗ Failed to extract face embedding for {face_id}: {e}")
            return None
    
    def crop_face_region(self, image: Image.Image) -> Optional[Image.Image]:
        """Crop face region from image (simplified implementation)"""
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Crop and return
            face_crop = image.crop((x, y, x + w, y + h))
            return face_crop
        
        return None
    
    def apply_character_conditioning(self, prompt: str, character_name: str, 
                                   strength: float = 1.0) -> Tuple[str, Optional[torch.Tensor]]:
        """Apply character conditioning to prompt"""
        if character_name not in self.character_embeddings:
            return prompt, None
        
        # Get character embedding
        character_embedding = self.character_embeddings[character_name]
        
        # Enhance prompt with character-specific terms
        enhanced_prompt = f"{prompt}, consistent character appearance, detailed face, high quality"
        
        return enhanced_prompt, character_embedding * strength
    
    def apply_face_conditioning(self, prompt: str, face_id: str, 
                              strength: float = 1.0) -> Tuple[str, Optional[torch.Tensor]]:
        """Apply face-specific conditioning"""
        if face_id not in self.face_embeddings:
            return prompt, None
        
        # Get face embedding
        face_embedding = self.face_embeddings[face_id]
        
        # Enhance prompt with face-specific terms
        enhanced_prompt = f"{prompt}, detailed facial features, consistent face, high quality portrait"
        
        return enhanced_prompt, face_embedding * strength
    
    def get_character_consistency_loss(self, generated_image: torch.Tensor, 
                                     character_name: str) -> torch.Tensor:
        """Calculate consistency loss between generated image and character reference"""
        if character_name not in self.character_embeddings:
            return torch.tensor(0.0, device=self.device)
        
        try:
            # Extract embedding from generated image
            if self.image_encoder and self.image_processor:
                # Process generated image
                gen_image_pil = Image.fromarray(
                    (generated_image.cpu().numpy() * 255).astype(np.uint8)
                )
                gen_inputs = self.image_processor(
                    images=gen_image_pil,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    gen_embeds = self.image_encoder(**gen_inputs).image_embeds
                    gen_embeds = F.normalize(gen_embeds, dim=-1)
                
                # Calculate cosine similarity
                ref_embeds = self.character_embeddings[character_name]
                similarity = F.cosine_similarity(gen_embeds, ref_embeds, dim=-1)
                
                # Return consistency loss (1 - similarity)
                return 1.0 - similarity.mean()
        
        except Exception as e:
            print(f"✗ Failed to calculate consistency loss: {e}")
        
        return torch.tensor(0.0, device=self.device)
    
    def register_character(self, character_name: str, reference_images: List[str], 
                          face_images: List[str] = None):
        """Register a character with reference images"""
        print(f"Registering character: {character_name}")
        
        # Extract character embeddings from reference images
        for i, img_path in enumerate(reference_images):
            if os.path.exists(img_path):
                self.extract_character_embedding(img_path, f"{character_name}_ref_{i}")
        
        # Extract face embeddings if provided
        if face_images:
            for i, img_path in enumerate(face_images):
                if os.path.exists(img_path):
                    self.extract_face_embedding(img_path, f"{character_name}_face_{i}")
    
    def get_character_embedding(self, character_name: str) -> Optional[torch.Tensor]:
        """Get character embedding by name"""
        if character_name in self.character_embeddings:
            return self.character_embeddings[character_name]
        
        # Try to find by prefix
        for key, embedding in self.character_embeddings.items():
            if key.startswith(character_name):
                return embedding
        
        return None
    
    def list_registered_characters(self) -> List[str]:
        """List all registered characters"""
        characters = set()
        for key in self.character_embeddings.keys():
            # Extract character name from key
            if '_ref_' in key:
                char_name = key.split('_ref_')[0]
                characters.add(char_name)
        
        return list(characters)
    
    def clear_character(self, character_name: str):
        """Clear character embeddings"""
        keys_to_remove = []
        for key in self.character_embeddings.keys():
            if key.startswith(character_name):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.character_embeddings[key]
        
        print(f"Cleared character: {character_name}")

class CharacterConsistencyEnhancer:
    """Enhanced character consistency using multiple techniques"""
    
    def __init__(self, ip_adapter_manager: IPAdapterManager):
        self.ip_manager = ip_adapter_manager
        self.character_templates = {}
        self.face_templates = {}
        
    def create_character_template(self, character_name: str, reference_images: List[str]):
        """Create a character template for consistency"""
        template = {
            'name': character_name,
            'embeddings': [],
            'face_embeddings': [],
            'attributes': {}
        }
        
        # Extract embeddings from reference images
        for img_path in reference_images:
            if os.path.exists(img_path):
                embedding = self.ip_manager.extract_character_embedding(img_path, f"{character_name}_temp")
                if embedding is not None:
                    template['embeddings'].append(embedding)
        
        self.character_templates[character_name] = template
        print(f"✓ Created template for character: {character_name}")
    
    def enhance_prompt_with_character(self, prompt: str, character_name: str, 
                                    style: str = "realistic") -> str:
        """Enhance prompt with character-specific details"""
        if character_name not in self.character_templates:
            return prompt
        
        template = self.character_templates[character_name]
        
        # Add character-specific enhancements
        enhancements = [
            "consistent character appearance",
            "detailed facial features",
            "high quality rendering",
            "stable character design"
        ]
        
        if style == "anime":
            enhancements.extend(["anime style", "cel-shaded", "vibrant colors"])
        elif style == "realistic":
            enhancements.extend(["photorealistic", "detailed textures", "natural lighting"])
        elif style == "cartoon":
            enhancements.extend(["cartoon style", "simple shapes", "bold colors"])
        
        enhanced_prompt = f"{prompt}, {', '.join(enhancements)}"
        
        return enhanced_prompt
    
    def get_character_embedding_ensemble(self, character_name: str) -> Optional[torch.Tensor]:
        """Get ensemble embedding for character"""
        if character_name not in self.character_templates:
            return None
        
        template = self.character_templates[character_name]
        if not template['embeddings']:
            return None
        
        # Average all embeddings for the character
        ensemble_embedding = torch.stack(template['embeddings']).mean(dim=0)
        return ensemble_embedding
    
    def apply_multi_character_conditioning(self, prompt: str, characters: List[str], 
                                         strengths: List[float] = None) -> Tuple[str, List[torch.Tensor]]:
        """Apply conditioning for multiple characters"""
        if strengths is None:
            strengths = [1.0] * len(characters)
        
        enhanced_prompt = prompt
        embeddings = []
        
        for char_name, strength in zip(characters, strengths):
            if char_name in self.character_templates:
                # Enhance prompt
                enhanced_prompt = self.enhance_prompt_with_character(enhanced_prompt, char_name)
                
                # Get embedding
                embedding = self.get_character_embedding_ensemble(char_name)
                if embedding is not None:
                    embeddings.append(embedding * strength)
        
        return enhanced_prompt, embeddings

# Utility functions for IP-Adapter integration
def load_ip_adapter_weights(model_path: str) -> Dict:
    """Load IP-Adapter weights from file"""
    if os.path.exists(model_path):
        return torch.load(model_path, map_location='cpu')
    return {}

def save_character_embeddings(embeddings: Dict, save_path: str):
    """Save character embeddings to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings, save_path)
    print(f"✓ Saved character embeddings to {save_path}")

def load_character_embeddings(load_path: str) -> Dict:
    """Load character embeddings from file"""
    if os.path.exists(load_path):
        embeddings = torch.load(load_path, map_location='cpu')
        print(f"✓ Loaded character embeddings from {load_path}")
        return embeddings
    return {}

def create_character_dataset(character_images: List[str], output_dir: str):
    """Create a dataset for character training"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_path in enumerate(character_images):
        if os.path.exists(img_path):
            # Copy and rename image
            img = Image.open(img_path)
            output_path = os.path.join(output_dir, f"character_{i:03d}.png")
            img.save(output_path)
    
    print(f"✓ Created character dataset in {output_dir}")

if __name__ == "__main__":
    # Test IP-Adapter integration
    ip_manager = IPAdapterManager()
    
    # Test character registration
    test_images = ["examples/test_character.png"]
    if os.path.exists(test_images[0]):
        ip_manager.register_character("test_character", test_images)
        print("✓ Test character registered successfully")
    else:
        print("⚠ Test image not found, skipping test")