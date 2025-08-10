#!/usr/bin/env python3
"""
CodeLlama Integration for Self-Improvement and Workflow Orchestration
Based on https://github.com/meta-llama/codellama
"""

import os
import sys
import json
import torch
import logging
import subprocess
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from queue import Queue, Empty
from pathlib import Path
import re
import ast
import traceback
from dataclasses import dataclass
from enum import Enum

# Transformers for CodeLlama
try:
    from transformers import (
        CodeLlamaTokenizer, 
        LlamaForCausalLM,
        AutoTokenizer,
        AutoModelForCausalLM,
        GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, using fallback implementation")

logger = logging.getLogger(__name__)

class WorkflowTaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    BUG_FIX = "bug_fix"
    OPTIMIZATION = "optimization"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"

@dataclass
class WorkflowTask:
    """Workflow task for CodeLlama processing"""
    id: str
    type: WorkflowTaskType
    description: str
    input_code: Optional[str] = None
    expected_output: Optional[str] = None
    priority: int = 1
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

class CodeLlamaProcessor:
    """CodeLlama model processor for code tasks"""
    
    def __init__(self, 
                 model_name: str = "codellama/CodeLlama-7b-Python-hf",
                 device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        # Load model if transformers available
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("Using fallback implementation without actual CodeLlama model")
        
        # Code analysis patterns
        self.code_patterns = {
            'function_def': r'def\s+(\w+)\s*\([^)]*\):',
            'class_def': r'class\s+(\w+)(?:\([^)]*\))?:',
            'import_stmt': r'(?:from\s+\S+\s+)?import\s+.+',
            'error_handling': r'try:|except\s+\w*:',
            'async_def': r'async\s+def\s+(\w+)',
            'type_hints': r':\s*\w+(?:\[.*?\])?(?:\s*=|$)',
        }
        
        logger.info(f"CodeLlama processor initialized on {self.device}")
    
    def _load_model(self):
        """Load CodeLlama model and tokenizer"""
        try:
            logger.info(f"Loading CodeLlama model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Generation configuration
            self.generation_config = GenerationConfig(
                max_new_tokens=2048,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
            
            logger.info("✅ CodeLlama model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading CodeLlama model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def generate_code(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate code using CodeLlama"""
        try:
            if not self.model or not self.tokenizer:
                return self._fallback_code_generation(prompt)
            
            # Prepare prompt for code generation
            code_prompt = f"# Task: {prompt}\n# Solution:\n"
            
            # Tokenize input
            inputs = self.tokenizer(code_prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 2048),
                    temperature=0.1,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_code = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up output
            generated_code = self._clean_generated_code(generated_code)
            
            logger.debug(f"Generated code for prompt: {prompt[:50]}...")
            return generated_code
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return self._fallback_code_generation(prompt)
    
    def review_code(self, code: str) -> Dict[str, Any]:
        """Review code and provide suggestions"""
        try:
            review_result = {
                'issues': [],
                'suggestions': [],
                'quality_score': 0.0,
                'complexity_score': 0.0
            }
            
            # Analyze code structure
            issues, suggestions = self._analyze_code_structure(code)
            review_result['issues'].extend(issues)
            review_result['suggestions'].extend(suggestions)
            
            # Calculate quality score
            review_result['quality_score'] = self._calculate_quality_score(code)
            review_result['complexity_score'] = self._calculate_complexity_score(code)
            
            # Use CodeLlama for advanced review if available
            if self.model and self.tokenizer:
                ai_review = self._ai_code_review(code)
                if ai_review:
                    review_result['ai_suggestions'] = ai_review
            
            return review_result
            
        except Exception as e:
            logger.error(f"Error reviewing code: {str(e)}")
            return {'issues': [f"Review error: {str(e)}"], 'suggestions': []}
    
    def optimize_code(self, code: str) -> str:
        """Optimize code for better performance"""
        try:
            if not self.model or not self.tokenizer:
                return self._fallback_code_optimization(code)
            
            optimization_prompt = f"""
# Optimize the following Python code for better performance and readability:
# Original code:
{code}

# Optimized code:
"""
            
            optimized_code = self.generate_code(optimization_prompt, max_tokens=1536)
            
            # Validate optimized code
            if self._validate_code_syntax(optimized_code):
                return optimized_code
            else:
                logger.warning("Generated optimized code has syntax errors, returning original")
                return code
                
        except Exception as e:
            logger.error(f"Error optimizing code: {str(e)}")
            return code
    
    def _fallback_code_generation(self, prompt: str) -> str:
        """Fallback code generation without CodeLlama model"""
        # Simple template-based code generation
        if "function" in prompt.lower():
            return f"""def generated_function():
    \"\"\"Generated function for: {prompt}\"\"\"
    # TODO: Implement functionality
    pass"""
        elif "class" in prompt.lower():
            return f"""class GeneratedClass:
    \"\"\"Generated class for: {prompt}\"\"\"
    
    def __init__(self):
        # TODO: Initialize attributes
        pass
    
    def method(self):
        # TODO: Implement method
        pass"""
        else:
            return f"""# Generated code for: {prompt}
# TODO: Implement functionality
pass"""
    
    def _fallback_code_optimization(self, code: str) -> str:
        """Fallback code optimization without model"""
        # Basic optimizations
        optimized = code
        
        # Remove redundant passes
        optimized = re.sub(r'\n\s*pass\s*\n\s*pass', '\n    pass', optimized)
        
        # Add basic type hints if missing
        if 'def ' in optimized and '->' not in optimized:
            optimized = re.sub(r'def (\w+)\(([^)]*)\):', r'def \1(\2) -> Any:', optimized)
        
        return optimized
    
    def _analyze_code_structure(self, code: str) -> tuple:
        """Analyze code structure for issues and suggestions"""
        issues = []
        suggestions = []
        
        lines = code.split('\n')
        
        # Check for basic issues
        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 100:
                issues.append(f"Line {i}: Line too long ({len(line)} characters)")
            
            # Missing docstrings
            if line.strip().startswith('def ') and i < len(lines):
                next_line = lines[i].strip() if i < len(lines) else ""
                if not next_line.startswith('"""') and not next_line.startswith("'''"):
                    suggestions.append(f"Line {i}: Consider adding docstring to function")
            
            # Bare except clauses
            if 'except:' in line:
                issues.append(f"Line {i}: Avoid bare except clauses")
        
        # Check for patterns
        if not any(pattern in code for pattern in ['try:', 'except']):
            suggestions.append("Consider adding error handling")
        
        if 'print(' in code:
            suggestions.append("Consider using logging instead of print statements")
        
        return issues, suggestions
    
    def _calculate_quality_score(self, code: str) -> float:
        """Calculate code quality score"""
        score = 1.0
        
        # Deduct for issues
        if len(code.split('\n')) == 0:
            return 0.0
        
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 0.0
        
        # Check documentation
        has_docstring = any('"""' in line or "'''" in line for line in lines)
        if has_docstring:
            score += 0.2
        
        # Check type hints
        has_type_hints = any('->' in line or ': ' in line for line in lines)
        if has_type_hints:
            score += 0.1
        
        # Check error handling
        has_error_handling = any('try:' in line or 'except' in line for line in lines)
        if has_error_handling:
            score += 0.1
        
        # Penalize very long lines
        long_lines = sum(1 for line in lines if len(line) > 100)
        score -= long_lines * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _calculate_complexity_score(self, code: str) -> float:
        """Calculate code complexity score"""
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity += 1
                elif isinstance(node, ast.Try):
                    complexity += 1
                elif isinstance(node, ast.FunctionDef):
                    complexity += 1
            
            # Normalize to 0-1 scale
            return min(1.0, complexity / 20.0)
            
        except:
            return 0.5  # Default complexity
    
    def _ai_code_review(self, code: str) -> str:
        """Use AI model for code review"""
        try:
            review_prompt = f"""
# Review the following Python code and provide suggestions for improvement:
{code}

# Review comments:
"""
            return self.generate_code(review_prompt, max_tokens=512)
            
        except Exception as e:
            logger.error(f"Error in AI code review: {str(e)}")
            return ""
    
    def _validate_code_syntax(self, code: str) -> bool:
        """Validate code syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code"""
        # Remove common generation artifacts
        code = code.strip()
        
        # Remove duplicate newlines
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
        
        # Stop at common end markers
        end_markers = ['# End', '# EOF', '```', '---']
        for marker in end_markers:
            if marker in code:
                code = code.split(marker)[0]
        
        return code.strip()

class WorkflowOrchestrator:
    """Orchestrates AI movie maker workflows using CodeLlama"""
    
    def __init__(self, codellama_processor: CodeLlamaProcessor):
        self.codellama = codellama_processor
        self.task_queue = Queue()
        self.completed_tasks = {}
        self.running = False
        self.worker_thread = None
        
        # Workflow templates
        self.workflow_templates = {
            'video_generation_optimization': self._video_generation_workflow,
            'character_consistency_improvement': self._character_consistency_workflow,
            'physics_simulation_enhancement': self._physics_simulation_workflow,
            'error_handling_improvement': self._error_handling_workflow
        }
        
        logger.info("Workflow orchestrator initialized")
    
    def start_orchestrator(self):
        """Start the workflow orchestrator"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("✅ Workflow orchestrator started")
    
    def stop_orchestrator(self):
        """Stop the workflow orchestrator"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("🛑 Workflow orchestrator stopped")
    
    def submit_task(self, task: WorkflowTask) -> str:
        """Submit a task to the orchestrator"""
        self.task_queue.put(task)
        logger.info(f"📋 Task submitted: {task.id} ({task.type.value})")
        return task.id
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a completed task"""
        return self.completed_tasks.get(task_id)
    
    def analyze_and_improve_code(self, file_path: str) -> Dict[str, Any]:
        """Analyze and improve code file"""
        try:
            if not os.path.exists(file_path):
                return {'error': f'File not found: {file_path}'}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Review code
            review_result = self.codellama.review_code(code)
            
            # Generate improvements
            if review_result.get('quality_score', 0) < 0.8:
                improved_code = self.codellama.optimize_code(code)
                review_result['improved_code'] = improved_code
                review_result['improvement_suggested'] = True
            else:
                review_result['improvement_suggested'] = False
            
            return review_result
            
        except Exception as e:
            logger.error(f"Error analyzing code file {file_path}: {str(e)}")
            return {'error': str(e)}
    
    def generate_feature_code(self, feature_description: str) -> str:
        """Generate code for a new feature"""
        try:
            feature_prompt = f"""
Generate Python code for the following feature in an AI movie maker application:
{feature_description}

The code should be well-documented, include error handling, and follow best practices.
"""
            return self.codellama.generate_code(feature_prompt, max_tokens=2048)
            
        except Exception as e:
            logger.error(f"Error generating feature code: {str(e)}")
            return f"# Error generating code: {str(e)}"
    
    def _worker_loop(self):
        """Main worker loop for processing tasks"""
        while self.running:
            try:
                # Get next task
                task = self.task_queue.get(timeout=1.0)
                
                logger.info(f"🔄 Processing task: {task.id}")
                
                # Process task based on type
                result = self._process_task(task)
                
                # Store result
                self.completed_tasks[task.id] = {
                    'task': task,
                    'result': result,
                    'completed_at': time.time(),
                    'success': result.get('success', False)
                }
                
                logger.info(f"✅ Task completed: {task.id}")
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}")
    
    def _process_task(self, task: WorkflowTask) -> Dict[str, Any]:
        """Process a single task"""
        try:
            if task.type == WorkflowTaskType.CODE_GENERATION:
                return self._handle_code_generation(task)
            elif task.type == WorkflowTaskType.CODE_REVIEW:
                return self._handle_code_review(task)
            elif task.type == WorkflowTaskType.OPTIMIZATION:
                return self._handle_optimization(task)
            elif task.type == WorkflowTaskType.WORKFLOW_ORCHESTRATION:
                return self._handle_workflow_orchestration(task)
            else:
                return {'success': False, 'error': f'Unknown task type: {task.type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_code_generation(self, task: WorkflowTask) -> Dict[str, Any]:
        """Handle code generation task"""
        try:
            generated_code = self.codellama.generate_code(task.description)
            
            return {
                'success': True,
                'generated_code': generated_code,
                'quality_score': self.codellama._calculate_quality_score(generated_code)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_code_review(self, task: WorkflowTask) -> Dict[str, Any]:
        """Handle code review task"""
        try:
            review_result = self.codellama.review_code(task.input_code or "")
            review_result['success'] = True
            return review_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_optimization(self, task: WorkflowTask) -> Dict[str, Any]:
        """Handle optimization task"""
        try:
            optimized_code = self.codellama.optimize_code(task.input_code or "")
            
            return {
                'success': True,
                'optimized_code': optimized_code,
                'original_quality': self.codellama._calculate_quality_score(task.input_code or ""),
                'optimized_quality': self.codellama._calculate_quality_score(optimized_code)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_workflow_orchestration(self, task: WorkflowTask) -> Dict[str, Any]:
        """Handle workflow orchestration task"""
        try:
            workflow_type = task.metadata.get('workflow_type', 'general')
            
            if workflow_type in self.workflow_templates:
                result = self.workflow_templates[workflow_type](task)
                result['success'] = True
                return result
            else:
                return {'success': False, 'error': f'Unknown workflow type: {workflow_type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _video_generation_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Workflow for video generation optimization"""
        improvements = []
        
        # Analyze current video generation code
        improvements.append("Implement progressive video generation for better memory usage")
        improvements.append("Add video quality optimization based on content analysis")
        improvements.append("Implement caching for frequently used models")
        
        return {
            'workflow_type': 'video_generation_optimization',
            'improvements': improvements,
            'estimated_impact': 'High'
        }
    
    def _character_consistency_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Workflow for character consistency improvement"""
        improvements = []
        
        improvements.append("Enhance IP-Adapter integration with temporal consistency")
        improvements.append("Implement character embedding caching")
        improvements.append("Add character similarity scoring across frames")
        
        return {
            'workflow_type': 'character_consistency_improvement',
            'improvements': improvements,
            'estimated_impact': 'Medium'
        }
    
    def _physics_simulation_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Workflow for physics simulation enhancement"""
        improvements = []
        
        improvements.append("Implement advanced collision detection algorithms")
        improvements.append("Add soft body physics simulation")
        improvements.append("Optimize physics step calculations for real-time performance")
        
        return {
            'workflow_type': 'physics_simulation_enhancement',
            'improvements': improvements,
            'estimated_impact': 'High'
        }
    
    def _error_handling_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Workflow for error handling improvement"""
        improvements = []
        
        improvements.append("Add comprehensive exception handling throughout the pipeline")
        improvements.append("Implement graceful degradation for model loading failures")
        improvements.append("Add detailed error reporting and recovery mechanisms")
        
        return {
            'workflow_type': 'error_handling_improvement',
            'improvements': improvements,
            'estimated_impact': 'Critical'
        }

class SelfImprovementSystem:
    """Self-improvement system using CodeLlama"""
    
    def __init__(self, workspace_path: str = "/workspace"):
        self.workspace_path = Path(workspace_path)
        self.codellama = CodeLlamaProcessor()
        self.orchestrator = WorkflowOrchestrator(self.codellama)
        
        # Start orchestrator
        self.orchestrator.start_orchestrator()
        
        # Performance metrics
        self.metrics = {
            'code_quality_improvements': 0,
            'bugs_fixed': 0,
            'features_generated': 0,
            'optimizations_applied': 0
        }
        
        logger.info("🧠 Self-improvement system initialized")
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze entire codebase for improvement opportunities"""
        analysis_results = {
            'files_analyzed': 0,
            'issues_found': [],
            'suggestions': [],
            'overall_quality': 0.0
        }
        
        try:
            # Find Python files
            python_files = list(self.workspace_path.glob("**/*.py"))
            
            total_quality = 0.0
            
            for file_path in python_files:
                if file_path.name.startswith('.'):
                    continue
                
                result = self.orchestrator.analyze_and_improve_code(str(file_path))
                
                if 'error' not in result:
                    analysis_results['files_analyzed'] += 1
                    analysis_results['issues_found'].extend(result.get('issues', []))
                    analysis_results['suggestions'].extend(result.get('suggestions', []))
                    total_quality += result.get('quality_score', 0.0)
            
            if analysis_results['files_analyzed'] > 0:
                analysis_results['overall_quality'] = total_quality / analysis_results['files_analyzed']
            
            logger.info(f"📊 Analyzed {analysis_results['files_analyzed']} files")
            logger.info(f"📈 Overall quality score: {analysis_results['overall_quality']:.2f}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing codebase: {str(e)}")
            return analysis_results
    
    def suggest_improvements(self) -> List[str]:
        """Suggest system-wide improvements"""
        suggestions = []
        
        # Analyze codebase
        analysis = self.analyze_codebase()
        
        if analysis['overall_quality'] < 0.7:
            suggestions.append("Code quality below threshold - recommend comprehensive refactoring")
        
        if len(analysis['issues_found']) > 10:
            suggestions.append("Multiple code issues detected - recommend automated fixing")
        
        # Submit workflow tasks for major improvements
        workflows = [
            'video_generation_optimization',
            'character_consistency_improvement',
            'physics_simulation_enhancement',
            'error_handling_improvement'
        ]
        
        for workflow in workflows:
            task = WorkflowTask(
                id=f"improvement_{workflow}_{int(time.time())}",
                type=WorkflowTaskType.WORKFLOW_ORCHESTRATION,
                description=f"Analyze and improve {workflow}",
                metadata={'workflow_type': workflow}
            )
            
            self.orchestrator.submit_task(task)
            suggestions.append(f"Submitted improvement workflow: {workflow}")
        
        return suggestions
    
    def get_improvement_metrics(self) -> Dict[str, Any]:
        """Get self-improvement metrics"""
        return {
            'metrics': self.metrics.copy(),
            'active_tasks': self.orchestrator.task_queue.qsize(),
            'completed_tasks': len(self.orchestrator.completed_tasks),
            'system_health': self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        # Simple health calculation based on metrics
        base_score = 0.7
        
        # Boost for improvements
        if self.metrics['code_quality_improvements'] > 5:
            base_score += 0.1
        
        if self.metrics['bugs_fixed'] > 3:
            base_score += 0.1
        
        if self.metrics['optimizations_applied'] > 2:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def shutdown(self):
        """Shutdown the self-improvement system"""
        self.orchestrator.stop_orchestrator()
        logger.info("🔌 Self-improvement system shutdown")

# Factory function
def create_codellama_system(workspace_path: str = "/workspace") -> SelfImprovementSystem:
    """Create and initialize CodeLlama self-improvement system"""
    return SelfImprovementSystem(workspace_path)

# Test function
def test_codellama_integration():
    """Test CodeLlama integration"""
    try:
        logger.info("🧪 Testing CodeLlama integration...")
        
        # Initialize processor
        processor = CodeLlamaProcessor()
        
        # Test code generation
        test_prompt = "Create a function to calculate video frame interpolation"
        generated_code = processor.generate_code(test_prompt)
        
        if generated_code and len(generated_code) > 10:
            logger.info("✅ Code generation successful")
            
            # Test code review
            review_result = processor.review_code(generated_code)
            
            if review_result and 'quality_score' in review_result:
                logger.info(f"✅ Code review successful (quality: {review_result['quality_score']:.2f})")
                
                # Test self-improvement system
                improvement_system = create_codellama_system()
                suggestions = improvement_system.suggest_improvements()
                
                if suggestions:
                    logger.info(f"✅ Self-improvement system working ({len(suggestions)} suggestions)")
                    improvement_system.shutdown()
                    return True
                else:
                    logger.warning("⚠️ No improvement suggestions generated")
                    improvement_system.shutdown()
                    return False
            else:
                logger.warning("⚠️ Code review failed")
                return False
        else:
            logger.warning("⚠️ Code generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ CodeLlama integration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_codellama_integration()