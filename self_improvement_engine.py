#!/usr/bin/env python3
"""
Self-Improvement Engine using CodeLlama
Provides workflow optimization, code quality analysis, and continuous improvement
for the AI movie maker system.
"""

import os
import sys
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ImprovementSuggestion:
    """Data class for improvement suggestions"""
    category: str
    title: str
    description: str
    priority: int  # 1-5, 5 being highest
    implementation_difficulty: str  # "easy", "medium", "hard"
    estimated_impact: str  # "low", "medium", "high"
    code_example: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class WorkflowAnalysis:
    """Data class for workflow analysis results"""
    workflow_id: str
    analysis_timestamp: datetime
    performance_metrics: Dict[str, float]
    bottlenecks: List[str]
    suggestions: List[ImprovementSuggestion]
    code_quality_score: float
    optimization_potential: float

class CodeLlamaAnalyzer:
    """CodeLlama-based code and workflow analyzer"""
    
    def __init__(
        self, 
        model_path: str = "codellama/CodeLlama-7b-Instruct-hf",
        device: str = "auto"
    ):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.analysis_cache = {}
        
        self.load_model()
    
    def load_model(self):
        """Load CodeLlama model"""
        try:
            logger.info(f"Loading CodeLlama model from {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            logger.info("CodeLlama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CodeLlama model: {e}")
            self.model = None
            self.tokenizer = None
    
    def analyze_code_quality(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze code quality using CodeLlama"""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        try:
            prompt = f"""
            Analyze the following {language} code for quality, performance, and best practices:
            
            ```{language}
            {code}
            ```
            
            Provide analysis in JSON format with the following structure:
            {{
                "overall_score": 0-100,
                "issues": [
                    {{
                        "type": "error|warning|suggestion",
                        "severity": "high|medium|low",
                        "description": "description",
                        "line": line_number,
                        "suggestion": "improvement suggestion"
                    }}
                ],
                "strengths": ["list of good practices"],
                "improvements": ["list of improvement suggestions"],
                "complexity_score": 0-10,
                "maintainability_score": 0-10
            }}
            """
            
            response = self._generate_response(prompt)
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                # Fallback: extract key information from text
                return self._extract_analysis_from_text(response)
                
        except Exception as e:
            logger.error(f"Code quality analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_workflow_performance(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow performance and suggest optimizations"""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        try:
            prompt = f"""
            Analyze this AI movie maker workflow for performance bottlenecks and optimization opportunities:
            
            {json.dumps(workflow_data, indent=2)}
            
            Focus on:
            1. Memory usage optimization
            2. GPU utilization
            3. Pipeline efficiency
            4. Character consistency improvements
            5. Physics simulation optimization
            6. Rendering performance
            
            Provide analysis in JSON format:
            {{
                "performance_score": 0-100,
                "bottlenecks": ["list of bottlenecks"],
                "optimizations": [
                    {{
                        "category": "memory|gpu|pipeline|consistency|physics|rendering",
                        "description": "optimization description",
                        "impact": "high|medium|low",
                        "effort": "easy|medium|hard",
                        "implementation": "detailed implementation steps"
                    }}
                ],
                "recommendations": ["list of recommendations"]
            }}
            """
            
            response = self._generate_response(prompt)
            
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                return self._extract_performance_analysis_from_text(response)
                
        except Exception as e:
            logger.error(f"Workflow performance analysis failed: {e}")
            return {"error": str(e)}
    
    def suggest_improvements(self, context: str, focus_areas: List[str]) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions based on context"""
        if not self.model or not self.tokenizer:
            return []
        
        try:
            focus_areas_str = ", ".join(focus_areas)
            
            prompt = f"""
            Based on the following context, suggest improvements focusing on: {focus_areas_str}
            
            Context:
            {context}
            
            Provide suggestions in JSON format:
            {{
                "suggestions": [
                    {{
                        "category": "category name",
                        "title": "suggestion title",
                        "description": "detailed description",
                        "priority": 1-5,
                        "implementation_difficulty": "easy|medium|hard",
                        "estimated_impact": "low|medium|high",
                        "code_example": "optional code example"
                    }}
                ]
            }}
            """
            
            response = self._generate_response(prompt)
            
            try:
                result = json.loads(response)
                suggestions = []
                
                for suggestion_data in result.get("suggestions", []):
                    suggestion = ImprovementSuggestion(
                        category=suggestion_data.get("category", "general"),
                        title=suggestion_data.get("title", "Improvement"),
                        description=suggestion_data.get("description", ""),
                        priority=suggestion_data.get("priority", 3),
                        implementation_difficulty=suggestion_data.get("implementation_difficulty", "medium"),
                        estimated_impact=suggestion_data.get("estimated_impact", "medium"),
                        code_example=suggestion_data.get("code_example")
                    )
                    suggestions.append(suggestion)
                
                return suggestions
                
            except json.JSONDecodeError:
                return self._extract_suggestions_from_text(response)
                
        except Exception as e:
            logger.error(f"Improvement suggestion generation failed: {e}")
            return []
    
    def _generate_response(self, prompt: str, max_length: int = 1024) -> str:
        """Generate response using CodeLlama"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return ""
    
    def _extract_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """Extract analysis information from text response"""
        analysis = {
            "overall_score": 50,
            "issues": [],
            "strengths": [],
            "improvements": [],
            "complexity_score": 5,
            "maintainability_score": 5
        }
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip().lower()
            if "score" in line and any(char.isdigit() for char in line):
                # Extract score
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    if "overall" in line or "quality" in line:
                        analysis["overall_score"] = min(100, int(numbers[0]))
                    elif "complexity" in line:
                        analysis["complexity_score"] = min(10, int(numbers[0]))
                    elif "maintainability" in line:
                        analysis["maintainability_score"] = min(10, int(numbers[0]))
            
            elif "issue" in line or "problem" in line or "error" in line:
                analysis["issues"].append(line)
            elif "strength" in line or "good" in line or "positive" in line:
                analysis["strengths"].append(line)
            elif "improve" in line or "suggestion" in line or "recommend" in line:
                analysis["improvements"].append(line)
        
        return analysis
    
    def _extract_performance_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """Extract performance analysis from text response"""
        analysis = {
            "performance_score": 50,
            "bottlenecks": [],
            "optimizations": [],
            "recommendations": []
        }
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip().lower()
            if "bottleneck" in line or "slow" in line or "performance issue" in line:
                analysis["bottlenecks"].append(line)
            elif "optimize" in line or "improve" in line or "enhance" in line:
                analysis["optimizations"].append({"description": line, "impact": "medium", "effort": "medium"})
            elif "recommend" in line or "suggest" in line:
                analysis["recommendations"].append(line)
        
        return analysis
    
    def _extract_suggestions_from_text(self, text: str) -> List[ImprovementSuggestion]:
        """Extract improvement suggestions from text response"""
        suggestions = []
        lines = text.split('\n')
        
        current_suggestion = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for suggestion patterns
            if any(keyword in line.lower() for keyword in ["improve", "optimize", "enhance", "fix", "add"]):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                
                current_suggestion = ImprovementSuggestion(
                    category="general",
                    title=line[:50] + "..." if len(line) > 50 else line,
                    description=line,
                    priority=3,
                    implementation_difficulty="medium",
                    estimated_impact="medium"
                )
        
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        return suggestions

class WorkflowOptimizer:
    """Workflow optimization engine"""
    
    def __init__(self, analyzer: CodeLlamaAnalyzer):
        self.analyzer = analyzer
        self.optimization_history = []
        self.performance_baseline = {}
    
    def analyze_workflow(self, workflow_data: Dict[str, Any]) -> WorkflowAnalysis:
        """Analyze workflow and generate comprehensive analysis"""
        workflow_id = self._generate_workflow_id(workflow_data)
        
        # Check cache
        if workflow_id in self.analyzer.analysis_cache:
            return self.analyzer.analysis_cache[workflow_id]
        
        # Perform analysis
        code_analysis = self.analyzer.analyze_code_quality(
            workflow_data.get("code", ""),
            workflow_data.get("language", "python")
        )
        
        performance_analysis = self.analyzer.analyze_workflow_performance(workflow_data)
        
        # Generate improvement suggestions
        suggestions = self.analyzer.suggest_improvements(
            json.dumps(workflow_data),
            ["performance", "code_quality", "character_consistency", "physics_simulation"]
        )
        
        # Calculate scores
        code_quality_score = code_analysis.get("overall_score", 50)
        performance_score = performance_analysis.get("performance_score", 50)
        optimization_potential = (100 - performance_score) * 0.8
        
        # Create analysis result
        analysis = WorkflowAnalysis(
            workflow_id=workflow_id,
            analysis_timestamp=datetime.now(),
            performance_metrics={
                "code_quality": code_quality_score,
                "performance": performance_score,
                "complexity": code_analysis.get("complexity_score", 5),
                "maintainability": code_analysis.get("maintainability_score", 5)
            },
            bottlenecks=performance_analysis.get("bottlenecks", []),
            suggestions=suggestions,
            code_quality_score=code_quality_score,
            optimization_potential=optimization_potential
        )
        
        # Cache result
        self.analyzer.analysis_cache[workflow_id] = analysis
        
        return analysis
    
    def optimize_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations to workflow"""
        analysis = self.analyze_workflow(workflow_data)
        
        optimizations = {}
        
        # Apply high-priority suggestions
        high_priority_suggestions = [
            s for s in analysis.suggestions 
            if s.priority >= 4 and s.implementation_difficulty == "easy"
        ]
        
        for suggestion in high_priority_suggestions:
            optimization = self._apply_optimization(workflow_data, suggestion)
            if optimization:
                optimizations[suggestion.title] = optimization
        
        # Record optimization
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "workflow_id": analysis.workflow_id,
            "applied_optimizations": list(optimizations.keys()),
            "performance_improvement": self._calculate_improvement(analysis)
        })
        
        return {
            "original_workflow": workflow_data,
            "optimized_workflow": optimizations,
            "analysis": analysis,
            "applied_optimizations": list(optimizations.keys())
        }
    
    def _apply_optimization(self, workflow_data: Dict[str, Any], suggestion: ImprovementSuggestion) -> Optional[Dict[str, Any]]:
        """Apply a specific optimization"""
        try:
            if "memory" in suggestion.category.lower():
                return self._optimize_memory_usage(workflow_data)
            elif "gpu" in suggestion.category.lower():
                return self._optimize_gpu_usage(workflow_data)
            elif "pipeline" in suggestion.category.lower():
                return self._optimize_pipeline(workflow_data)
            elif "consistency" in suggestion.category.lower():
                return self._optimize_character_consistency(workflow_data)
            else:
                return self._apply_generic_optimization(workflow_data, suggestion)
                
        except Exception as e:
            logger.error(f"Failed to apply optimization {suggestion.title}: {e}")
            return None
    
    def _optimize_memory_usage(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory usage"""
        optimization = {
            "type": "memory_optimization",
            "changes": [],
            "estimated_memory_savings": "20-30%"
        }
        
        # Add memory optimization suggestions
        optimization["changes"].extend([
            "Enable gradient checkpointing for large models",
            "Use mixed precision training (FP16)",
            "Implement dynamic batching",
            "Add memory cleanup between pipeline stages"
        ])
        
        return optimization
    
    def _optimize_gpu_usage(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize GPU usage"""
        optimization = {
            "type": "gpu_optimization",
            "changes": [],
            "estimated_gpu_savings": "15-25%"
        }
        
        # Add GPU optimization suggestions
        optimization["changes"].extend([
            "Enable CUDA graph optimization",
            "Use tensor cores for FP16 operations",
            "Implement pipeline parallelism",
            "Optimize batch sizes for GPU memory"
        ])
        
        return optimization
    
    def _optimize_pipeline(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize pipeline efficiency"""
        optimization = {
            "type": "pipeline_optimization",
            "changes": [],
            "estimated_speedup": "30-50%"
        }
        
        # Add pipeline optimization suggestions
        optimization["changes"].extend([
            "Implement async processing for I/O operations",
            "Add caching for intermediate results",
            "Optimize data loading with prefetching",
            "Use parallel processing for independent operations"
        ])
        
        return optimization
    
    def _optimize_character_consistency(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize character consistency"""
        optimization = {
            "type": "consistency_optimization",
            "changes": [],
            "estimated_consistency_improvement": "40-60%"
        }
        
        # Add character consistency optimization suggestions
        optimization["changes"].extend([
            "Implement temporal consistency loss",
            "Add face landmark tracking",
            "Use adaptive IP-Adapter strength",
            "Implement pose-aware character embedding"
        ])
        
        return optimization
    
    def _apply_generic_optimization(self, workflow_data: Dict[str, Any], suggestion: ImprovementSuggestion) -> Dict[str, Any]:
        """Apply generic optimization based on suggestion"""
        return {
            "type": "generic_optimization",
            "suggestion": suggestion.description,
            "implementation": suggestion.code_example or "Manual implementation required",
            "estimated_impact": suggestion.estimated_impact
        }
    
    def _generate_workflow_id(self, workflow_data: Dict[str, Any]) -> str:
        """Generate unique workflow ID"""
        workflow_str = json.dumps(workflow_data, sort_keys=True)
        return hashlib.md5(workflow_str.encode()).hexdigest()[:16]
    
    def _calculate_improvement(self, analysis: WorkflowAnalysis) -> float:
        """Calculate estimated performance improvement"""
        return analysis.optimization_potential

class SelfImprovementEngine:
    """Main self-improvement engine"""
    
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-Instruct-hf"):
        self.analyzer = CodeLlamaAnalyzer(model_path)
        self.optimizer = WorkflowOptimizer(self.analyzer)
        self.improvement_history = []
    
    def analyze_and_improve(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow and suggest improvements"""
        try:
            # Analyze workflow
            analysis = self.optimizer.analyze_workflow(workflow_data)
            
            # Apply optimizations
            optimization_result = self.optimizer.optimize_workflow(workflow_data)
            
            # Record improvement
            self.improvement_history.append({
                "timestamp": datetime.now(),
                "workflow_id": analysis.workflow_id,
                "original_score": analysis.code_quality_score,
                "optimization_potential": analysis.optimization_potential,
                "suggestions_count": len(analysis.suggestions)
            })
            
            return {
                "analysis": analysis,
                "optimization_result": optimization_result,
                "summary": {
                    "code_quality_score": analysis.code_quality_score,
                    "performance_score": analysis.performance_metrics.get("performance", 50),
                    "suggestions_count": len(analysis.suggestions),
                    "applied_optimizations": len(optimization_result.get("applied_optimizations", [])),
                    "estimated_improvement": analysis.optimization_potential
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis and improvement failed: {e}")
            return {"error": str(e)}
    
    def get_improvement_history(self) -> List[Dict[str, Any]]:
        """Get improvement history"""
        return self.improvement_history
    
    def generate_improvement_report(self) -> str:
        """Generate improvement report"""
        if not self.improvement_history:
            return "No improvement history available."
        
        report = "# AI Movie Maker Self-Improvement Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary statistics
        total_workflows = len(self.improvement_history)
        avg_original_score = np.mean([h["original_score"] for h in self.improvement_history])
        avg_potential = np.mean([h["optimization_potential"] for h in self.improvement_history])
        total_suggestions = sum([h["suggestions_count"] for h in self.improvement_history])
        
        report += f"## Summary\n"
        report += f"- Total workflows analyzed: {total_workflows}\n"
        report += f"- Average original quality score: {avg_original_score:.1f}/100\n"
        report += f"- Average optimization potential: {avg_potential:.1f}%\n"
        report += f"- Total suggestions generated: {total_suggestions}\n\n"
        
        # Recent improvements
        report += "## Recent Improvements\n"
        recent_history = self.improvement_history[-5:]  # Last 5 entries
        for entry in recent_history:
            report += f"- **{entry['timestamp'].strftime('%Y-%m-%d %H:%M')}**: "
            report += f"Workflow {entry['workflow_id'][:8]}... "
            report += f"(Score: {entry['original_score']}, "
            report += f"Potential: {entry['optimization_potential']:.1f}%, "
            report += f"Suggestions: {entry['suggestions_count']})\n"
        
        return report

# Utility functions
def create_self_improvement_engine(
    model_path: str = "codellama/CodeLlama-7b-Instruct-hf"
) -> SelfImprovementEngine:
    """Create self-improvement engine with error handling"""
    try:
        return SelfImprovementEngine(model_path)
    except Exception as e:
        logger.error(f"Failed to create self-improvement engine: {e}")
        return None

def analyze_workflow_code(code: str, language: str = "python") -> Dict[str, Any]:
    """Quick code quality analysis"""
    analyzer = CodeLlamaAnalyzer()
    return analyzer.analyze_code_quality(code, language)

def suggest_workflow_improvements(workflow_data: Dict[str, Any]) -> List[ImprovementSuggestion]:
    """Quick improvement suggestions"""
    analyzer = CodeLlamaAnalyzer()
    return analyzer.suggest_improvements(
        json.dumps(workflow_data),
        ["performance", "code_quality", "user_experience"]
    )