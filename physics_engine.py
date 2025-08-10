#!/usr/bin/env python3
"""
Physics Engine for AI Movie Maker
Realistic motion and interactions using Pymunk physics engine
"""

import os
import sys
import warnings
import math
import random
import time
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np

warnings.filterwarnings('ignore')

try:
    import pymunk as pm
    import pygame
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False
    print("Physics engine not available")

@dataclass
class PhysicsObject:
    """Physics object configuration"""
    name: str
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0, 0)
    mass: float = 1.0
    shape_type: str = "circle"  # circle, box, polygon
    size: Tuple[float, float] = (10, 10)
    elasticity: float = 0.8
    friction: float = 0.7
    collision_type: int = 1
    is_static: bool = False
    color: Tuple[int, int, int] = (255, 255, 255)

@dataclass
class PhysicsConstraint:
    """Physics constraint configuration"""
    name: str
    object1: str
    object2: str
    constraint_type: str = "pin"  # pin, slide, pivot, gear
    anchor1: Tuple[float, float] = (0, 0)
    anchor2: Tuple[float, float] = (0, 0)
    distance: float = 100.0
    stiffness: float = 1000.0
    damping: float = 10.0

class PhysicsWorld:
    """Advanced physics world for realistic motion"""
    
    def __init__(self, width: int, height: int, gravity: Tuple[float, float] = (0, 981)):
        self.width = width
        self.height = height
        self.gravity = gravity
        
        # Initialize physics space
        self.space = pm.Space()
        self.space.gravity = gravity
        
        # Object storage
        self.objects = {}
        self.constraints = {}
        self.collision_handlers = {}
        
        # Physics properties
        self.time_step = 1.0 / 60.0
        self.iterations = 10
        
        # Visualization
        self.screen = None
        self.clock = None
        self.font = None
        
        self.setup_collision_handlers()
    
    def setup_collision_handlers(self):
        """Setup collision detection handlers"""
        # Character-character collision
        handler = self.space.add_collision_handler(1, 1)
        handler.begin = self.on_character_collision
        
        # Character-ground collision
        handler = self.space.add_collision_handler(1, 2)
        handler.begin = self.on_ground_collision
        
        # Object-object collision
        handler = self.space.add_collision_handler(3, 3)
        handler.begin = self.on_object_collision
    
    def on_character_collision(self, arbiter, space, data):
        """Handle character-character collisions"""
        shape_a, shape_b = arbiter.shapes
        # Add collision effects (sound, particles, etc.)
        return True
    
    def on_ground_collision(self, arbiter, space, data):
        """Handle character-ground collisions"""
        shape_a, shape_b = arbiter.shapes
        # Add ground contact effects
        return True
    
    def on_object_collision(self, arbiter, space, data):
        """Handle object-object collisions"""
        shape_a, shape_b = arbiter.shapes
        # Add object interaction effects
        return True
    
    def add_object(self, obj_config: PhysicsObject) -> pm.Body:
        """Add a physics object to the world"""
        # Create body
        if obj_config.is_static:
            body = pm.Body(body_type=pm.Body.STATIC)
        else:
            moment = pm.moment_for_circle(obj_config.mass, 0, obj_config.size[0])
            body = pm.Body(obj_config.mass, moment)
        
        body.position = obj_config.position
        body.velocity = obj_config.velocity
        
        # Create shape
        if obj_config.shape_type == "circle":
            shape = pm.Circle(body, obj_config.size[0])
        elif obj_config.shape_type == "box":
            shape = pm.Poly.create_box(body, obj_config.size)
        else:
            # Default to circle
            shape = pm.Circle(body, obj_config.size[0])
        
        # Set shape properties
        shape.elasticity = obj_config.elasticity
        shape.friction = obj_config.friction
        shape.collision_type = obj_config.collision_type
        
        # Add to space
        self.space.add(body, shape)
        
        # Store object
        self.objects[obj_config.name] = {
            'body': body,
            'shape': shape,
            'config': obj_config
        }
        
        return body
    
    def add_character(self, name: str, position: Tuple[float, float], 
                     mass: float = 70.0, size: float = 25.0) -> pm.Body:
        """Add a character to the physics world"""
        config = PhysicsObject(
            name=name,
            position=position,
            mass=mass,
            shape_type="circle",
            size=(size, size),
            collision_type=1,
            color=(100, 150, 255)
        )
        
        return self.add_object(config)
    
    def add_ground(self, y: float, width: float = None) -> pm.Body:
        """Add ground plane"""
        if width is None:
            width = self.width
        
        config = PhysicsObject(
            name="ground",
            position=(width/2, y),
            shape_type="box",
            size=(width, 10),
            is_static=True,
            collision_type=2,
            color=(100, 100, 100)
        )
        
        return self.add_object(config)
    
    def add_wall(self, name: str, position: Tuple[float, float], 
                size: Tuple[float, float]) -> pm.Body:
        """Add a wall or barrier"""
        config = PhysicsObject(
            name=name,
            position=position,
            shape_type="box",
            size=size,
            is_static=True,
            collision_type=2,
            color=(150, 150, 150)
        )
        
        return self.add_object(config)
    
    def add_constraint(self, constraint_config: PhysicsConstraint):
        """Add a constraint between objects"""
        if constraint_config.object1 not in self.objects or constraint_config.object2 not in self.objects:
            print(f"Objects not found for constraint: {constraint_config.name}")
            return
        
        body1 = self.objects[constraint_config.object1]['body']
        body2 = self.objects[constraint_config.object2]['body']
        
        if constraint_config.constraint_type == "pin":
            constraint = pm.PinJoint(body1, body2, constraint_config.anchor1, constraint_config.anchor2)
        elif constraint_config.constraint_type == "slide":
            constraint = pm.SlideJoint(body1, body2, constraint_config.anchor1, constraint_config.anchor2, 
                                     constraint_config.distance, constraint_config.distance)
        elif constraint_config.constraint_type == "pivot":
            constraint = pm.PivotJoint(body1, body2, constraint_config.anchor1)
        else:
            constraint = pm.PinJoint(body1, body2, constraint_config.anchor1, constraint_config.anchor2)
        
        # Set constraint properties
        if hasattr(constraint, 'distance'):
            constraint.distance = constraint_config.distance
        if hasattr(constraint, 'stiffness'):
            constraint.stiffness = constraint_config.stiffness
        if hasattr(constraint, 'damping'):
            constraint.damping = constraint_config.damping
        
        self.space.add(constraint)
        self.constraints[constraint_config.name] = constraint
    
    def apply_force(self, object_name: str, force: Tuple[float, float], 
                   point: Tuple[float, float] = None):
        """Apply force to an object"""
        if object_name not in self.objects:
            return
        
        body = self.objects[object_name]['body']
        if point:
            body.apply_force_at_local_point(force, point)
        else:
            body.apply_force_at_world_point(force, body.position)
    
    def apply_impulse(self, object_name: str, impulse: Tuple[float, float], 
                     point: Tuple[float, float] = None):
        """Apply impulse to an object"""
        if object_name not in self.objects:
            return
        
        body = self.objects[object_name]['body']
        if point:
            body.apply_impulse_at_local_point(impulse, point)
        else:
            body.apply_impulse_at_world_point(impulse, body.position)
    
    def get_object_position(self, object_name: str) -> Tuple[float, float]:
        """Get object position"""
        if object_name not in self.objects:
            return (0, 0)
        
        pos = self.objects[object_name]['body'].position
        return (pos.x, pos.y)
    
    def get_object_velocity(self, object_name: str) -> Tuple[float, float]:
        """Get object velocity"""
        if object_name not in self.objects:
            return (0, 0)
        
        vel = self.objects[object_name]['body'].velocity
        return (vel.x, vel.y)
    
    def set_object_position(self, object_name: str, position: Tuple[float, float]):
        """Set object position"""
        if object_name not in self.objects:
            return
        
        self.objects[object_name]['body'].position = position
    
    def set_object_velocity(self, object_name: str, velocity: Tuple[float, float]):
        """Set object velocity"""
        if object_name not in self.objects:
            return
        
        self.objects[object_name]['body'].velocity = velocity
    
    def step(self, dt: float = None):
        """Step physics simulation"""
        if dt is None:
            dt = self.time_step
        
        self.space.step(dt)
    
    def step_multiple(self, steps: int, dt: float = None):
        """Step physics simulation multiple times"""
        if dt is None:
            dt = self.time_step
        
        for _ in range(steps):
            self.space.step(dt)
    
    def setup_visualization(self, width: int = 800, height: int = 600):
        """Setup pygame visualization"""
        if not PHYSICS_AVAILABLE:
            return
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Physics World Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
    
    def draw_world(self, scale: float = 1.0, offset: Tuple[float, float] = (0, 0)):
        """Draw physics world"""
        if not self.screen:
            return
        
        self.screen.fill((0, 0, 0))
        
        # Draw objects
        for name, obj_data in self.objects.items():
            body = obj_data['body']
            shape = obj_data['shape']
            config = obj_data['config']
            
            # Get position
            pos = body.position
            x = int(pos.x * scale + offset[0])
            y = int(pos.y * scale + offset[1])
            
            # Draw based on shape type
            if isinstance(shape, pm.Circle):
                radius = int(shape.radius * scale)
                pygame.draw.circle(self.screen, config.color, (x, y), radius)
            elif isinstance(shape, pm.Poly):
                # Draw polygon (simplified)
                pygame.draw.rect(self.screen, config.color, 
                               (x - config.size[0]/2, y - config.size[1]/2, 
                                config.size[0], config.size[1]))
        
        # Draw constraints
        for name, constraint in self.constraints.items():
            if hasattr(constraint, 'a') and hasattr(constraint, 'b'):
                pos_a = constraint.a.position
                pos_b = constraint.b.position
                x1 = int(pos_a.x * scale + offset[0])
                y1 = int(pos_a.y * scale + offset[1])
                x2 = int(pos_b.x * scale + offset[1])
                y2 = int(pos_b.y * scale + offset[1])
                pygame.draw.line(self.screen, (255, 255, 0), (x1, y1), (x2, y2), 2)
        
        pygame.display.flip()
    
    def run_visualization(self, duration: float = 10.0, fps: int = 60):
        """Run physics simulation with visualization"""
        if not self.screen:
            self.setup_visualization()
        
        start_time = time.time()
        running = True
        
        while running and (time.time() - start_time) < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Step physics
            self.step()
            
            # Draw world
            self.draw_world()
            
            self.clock.tick(fps)
        
        pygame.quit()

class CharacterController:
    """Character movement controller"""
    
    def __init__(self, physics_world: PhysicsWorld, character_name: str):
        self.physics_world = physics_world
        self.character_name = character_name
        self.movement_speed = 200.0
        self.jump_force = 400.0
        self.ground_friction = 0.8
        
    def move_left(self):
        """Move character left"""
        self.physics_world.apply_force(self.character_name, (-self.movement_speed, 0))
    
    def move_right(self):
        """Move character right"""
        self.physics_world.apply_force(self.character_name, (self.movement_speed, 0))
    
    def jump(self):
        """Make character jump"""
        self.physics_world.apply_impulse(self.character_name, (0, -self.jump_force))
    
    def stop_movement(self):
        """Stop character movement"""
        velocity = self.physics_world.get_object_velocity(self.character_name)
        self.physics_world.set_object_velocity(self.character_name, (velocity[0] * self.ground_friction, velocity[1]))

class PhysicsScene:
    """Physics scene for movie generation"""
    
    def __init__(self, width: int, height: int):
        self.physics_world = PhysicsWorld(width, height)
        self.characters = {}
        self.scene_objects = {}
        self.animation_frames = []
        
    def setup_basic_scene(self):
        """Setup a basic scene with ground and walls"""
        # Add ground
        self.physics_world.add_ground(self.physics_world.height - 50)
        
        # Add walls
        wall_thickness = 20
        self.physics_world.add_wall("left_wall", (wall_thickness/2, self.physics_world.height/2), 
                                  (wall_thickness, self.physics_world.height))
        self.physics_world.add_wall("right_wall", (self.physics_world.width - wall_thickness/2, self.physics_world.height/2), 
                                  (wall_thickness, self.physics_world.height))
    
    def add_character(self, name: str, position: Tuple[float, float]) -> CharacterController:
        """Add a character to the scene"""
        self.physics_world.add_character(name, position)
        controller = CharacterController(self.physics_world, name)
        self.characters[name] = controller
        return controller
    
    def add_object(self, name: str, config: PhysicsObject):
        """Add an object to the scene"""
        self.physics_world.add_object(config)
        self.scene_objects[name] = config
    
    def animate_scene(self, duration: float, fps: int = 24) -> List[Dict]:
        """Animate the scene and return frame data"""
        frames = []
        total_steps = int(duration * fps)
        step_time = duration / total_steps
        
        for step in range(total_steps):
            # Step physics
            self.physics_world.step(step_time)
            
            # Record frame data
            frame_data = {
                'step': step,
                'time': step * step_time,
                'character_positions': {},
                'object_positions': {}
            }
            
            # Record character positions
            for char_name in self.characters.keys():
                pos = self.physics_world.get_object_position(char_name)
                frame_data['character_positions'][char_name] = pos
            
            # Record object positions
            for obj_name in self.scene_objects.keys():
                pos = self.physics_world.get_object_position(obj_name)
                frame_data['object_positions'][obj_name] = pos
            
            frames.append(frame_data)
        
        self.animation_frames = frames
        return frames
    
    def get_frame_data(self, frame_index: int) -> Dict:
        """Get data for a specific frame"""
        if 0 <= frame_index < len(self.animation_frames):
            return self.animation_frames[frame_index]
        return {}

def create_physics_config(scene_type: str) -> Dict:
    """Create physics configuration for different scene types"""
    configs = {
        'action': {
            'gravity': (0, 981),
            'friction': 0.3,
            'elasticity': 0.9,
            'character_mass': 70.0,
            'movement_speed': 300.0,
            'jump_force': 500.0
        },
        'drama': {
            'gravity': (0, 981),
            'friction': 0.8,
            'elasticity': 0.5,
            'character_mass': 70.0,
            'movement_speed': 150.0,
            'jump_force': 300.0
        },
        'comedy': {
            'gravity': (0, 981),
            'friction': 0.2,
            'elasticity': 0.95,
            'character_mass': 70.0,
            'movement_speed': 400.0,
            'jump_force': 600.0
        },
        'fantasy': {
            'gravity': (0, 400),  # Lower gravity for fantasy
            'friction': 0.5,
            'elasticity': 0.8,
            'character_mass': 50.0,
            'movement_speed': 250.0,
            'jump_force': 400.0
        }
    }
    
    return configs.get(scene_type, configs['action'])

if __name__ == "__main__":
    # Test physics engine
    if PHYSICS_AVAILABLE:
        # Create physics world
        world = PhysicsWorld(800, 600)
        
        # Setup basic scene
        world.add_ground(550)
        
        # Add characters
        char1 = world.add_character("hero", (200, 100))
        char2 = world.add_character("villain", (600, 100))
        
        # Add some objects
        obj1 = PhysicsObject("ball", (400, 50), mass=5.0, size=(15, 15), collision_type=3)
        world.add_object(obj1)
        
        # Run simulation
        print("Running physics simulation...")
        world.run_visualization(duration=5.0)
    else:
        print("Physics engine not available")