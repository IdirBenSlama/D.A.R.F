"""
LLM Registry Module

This module provides a registry for managing and interacting with
multiple LLM models through various backends like Ollama. It implements
the Component interface and integrates with the circuit breaker pattern
for resilient external API calls.
"""

import json
import logging
import os
import re
import time
import requests
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from src.types.common_types import Result

from src.interfaces.component import Component
from src.errors import DARFError, ExternalServiceError
from src.config.config_manager import config_manager
from src.utils.circuit_breaker import CircuitBreaker, circuit_breaker, circuit_breaker_registry

logger = logging.getLogger("DARF.LLM")

class ModelNotFoundError(DARFError):
    """Raised when a requested model is not found in the registry."""
    pass

class ModelQueryError(ExternalServiceError):
    """Raised when there is an error querying a model."""
    pass

class LLMRegistry(Component):
    """
    Registry for managing multiple LLM models.
    
    This class manages the lifecycle of LLM models, handles model discovery,
    provides model selection based on capabilities, and implements resilient
    querying with circuit breakers.
    """
    
    def __init__(self, component_id: str = "llm_registry", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM Registry.
        
        Args:
            component_id: Unique identifier for this component
            config: Configuration dictionary
        """
        super().__init__(component_id, config)
        
        # Load configuration
        config_dict = config or {}
        if not config:
            # Try to load from global config
            llm_config = config_manager.get_section("llm", {})
            config_dict = llm_config
            
        # Initialize configuration
        self.models = {}
        self.default_model_id = config_dict.get("default_model", "llama3")
        self.ollama_api_url = config_dict.get("ollama_api_url", "http://localhost:11434")
        self.request_timeout = config_dict.get("request_timeout", 60)
        self.max_retries = config_dict.get("max_retries", 3)
        self.retry_delay = config_dict.get("retry_delay", 1.0)
        
        # Model performance metrics
        self.model_performance = {}
        
        # Set up circuit breakers
        self.ollama_cb = circuit_breaker_registry.get_or_create(
            "ollama_api",
            failure_threshold=3,
            success_threshold=2,
            timeout=self.request_timeout,
            reset_timeout=30.0,
            fallback_function=self._api_fallback
        )
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Status tracking
        self._initialized = False
        self._last_refresh = None
        
        self.logger.info(f"LLM Registry initialized with default model: {self.default_model_id}")
        
    async def start(self) -> Result[bool]:
        """
        Start the LLM Registry component.
        
        This method initializes the model list and ensures the default model
        is available.
        
        Returns:
            Result containing success status or error
        """
        try:
            # Initialize models
            await self._initialize_models()
            self._is_running = True
            self._initialized = True
            self._last_refresh = time.time()
            return Result.success(True)
        except Exception as e:
            self.logger.error(f"Error starting LLM Registry: {e}")
            return Result.failure(ExternalServiceError(f"Failed to start LLM Registry: {e}", "llm_registry"))
    
    async def stop(self) -> Result[bool]:
        """
        Stop the LLM Registry component.
        
        Returns:
            Result containing success status or error
        """
        try:
            self.executor.shutdown(wait=False)
            self._is_running = False
            return Result.success(True)
        except Exception as e:
            self.logger.error(f"Error stopping LLM Registry: {e}")
            return Result.failure(ExternalServiceError(f"Failed to stop LLM Registry: {e}", "llm_registry"))
    
    def get_dependencies(self) -> List[str]:
        """
        Get the dependencies for this component.
        
        Returns:
            List of dependency component names
        """
        return []
    
    async def _initialize_models(self) -> None:
        """Initialize models by querying the Ollama API with circuit breaker protection."""
        start_time = time.time()
        self.logger.info("Initializing LLM models from Ollama API...")
        
        try:
            # Use circuit breaker pattern
            def get_models():
                response = requests.get(
                    f"{self.ollama_api_url}/api/tags", 
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                return response.json()
            
            # Execute with circuit breaker
            data = self.ollama_cb.execute(get_models)
            models = data.get("models", [])
            
            # Process models in parallel for efficiency
            model_entries = []
            with ThreadPoolExecutor(max_workers=min(8, len(models))) as executor:
                model_entries = list(executor.map(self._process_model, models))
            
            # Filter out None entries and add to models dict
            for entry in model_entries:
                if entry:
                    model_id, model_info = entry
                    self.models[model_id] = model_info
            
            self.logger.info(f"Initialized {len(self.models)} models from Ollama")
            
            # Set default model if not available
            if self.default_model_id not in self.models:
                # Try to find a close match
                for model_id in self.models:
                    if self.default_model_id in model_id:
                        self.default_model_id = model_id
                        self.logger.info(f"Default model not found, using {model_id} instead")
                        break
                else:
                    # Otherwise use the first model
                    if self.models:
                        self.default_model_id = next(iter(self.models))
                        self.logger.warning(f"Default model not found, using {self.default_model_id} instead")
                    else:
                        # No models found, add fallback
                        self._add_fallback_models()
                        
            # Record initialization time
            initialization_time = time.time() - start_time
            self.logger.info(f"Model initialization completed in {initialization_time:.2f}s")
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            self._add_fallback_models()
    
    def _process_model(self, model_data: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Process a single model from Ollama API response.
        
        Args:
            model_data: Model data from API
            
        Returns:
            Tuple of (model_id, model_info) or None if invalid model
        """
        try:
            model_name = model_data.get("name", "")
            if not model_name:
                return None
                
            # Extract model info
            parts = model_name.split(":")
            base_name = parts[0]
            version = parts[1] if len(parts) > 1 else "latest"
            
            model_id = model_name
            
            # Get model details
            details = model_data.get("details", {})
            parameter_count = self._parse_parameter_size(details.get("parameter_size", ""))
            
            # Create model entry
            model_info = {
                "id": model_id,
                "name": base_name,
                "version": version,
                "parameters": parameter_count,
                "context_window": self._estimate_context_window(parameter_count),
                "provider": "Ollama",
                "capabilities": self._estimate_capabilities(base_name, parameter_count),
                "created_at": model_data.get("modified_at"),
                "format": details.get("format"),
                "family": details.get("family", base_name),
                "quantization": details.get("quantization_level")
            }
            
            return (model_id, model_info)
        except Exception as e:
            self.logger.warning(f"Error processing model {model_data.get('name', 'unknown')}: {e}")
            return None
            
    def _api_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Fallback function for the circuit breaker.
        
        Returns:
            Fallback response
        """
        self.logger.warning("Ollama API circuit breaker is open, using fallback response")
        return {"models": []}
    
    def _add_fallback_models(self) -> None:
        """Add fallback models when Ollama API is unavailable."""
        # Add default fallback model
        self.models[self.default_model_id] = {
            "id": self.default_model_id,
            "name": self.default_model_id.split(":")[0],
            "version": self.default_model_id.split(":")[-1] if ":" in self.default_model_id else "latest",
            "parameters": 14_800_000_000,  # 14.8B
            "context_window": 8192,
            "provider": "Ollama",
            "capabilities": ["code", "chat", "reasoning"],
            "is_fallback": True
        }
        
        # Add a smaller fallback model for low-latency requests
        small_model_id = "tinyllama:latest"
        self.models[small_model_id] = {
            "id": small_model_id,
            "name": "tinyllama",
            "version": "latest",
            "parameters": 1_300_000_000,  # 1.3B
            "context_window": 2048,
            "provider": "Ollama",
            "capabilities": ["chat"],
            "is_fallback": True
        }
        
        self.logger.info(f"Added fallback models: {self.default_model_id}, {small_model_id}")
    
    def _parse_parameter_size(self, size_str: str) -> int:
        """
        Parse parameter size string (like '14B' or '7b') to integer.
        
        Args:
            size_str: Parameter size string
            
        Returns:
            Number of parameters as integer
        """
        if not size_str:
            return 0
            
        # Remove any non-digit, non-decimal characters except B/b suffix
        clean_str = re.sub(r'[^0-9.]', '', size_str.rstrip('Bb'))
        
        try:
            # Convert to float and then to int with proper scaling
            if size_str.lower().endswith('b'):
                multiplier = 1_000_000_000  # billions
            elif size_str.lower().endswith('m'):
                multiplier = 1_000_000  # millions
            else:
                multiplier = 1_000_000_000  # default to billions
                
            size = float(clean_str) * multiplier
            return int(size)
        except ValueError:
            return 0
    
    def _estimate_context_window(self, parameter_count: int) -> int:
        """
        Estimate context window size based on parameter count.
        
        Args:
            parameter_count: Number of parameters
            
        Returns:
            Estimated context window size
        """
        # Very rough estimate based on general trends
        if parameter_count >= 70_000_000_000:  # 70B+
            return 32768
        elif parameter_count >= 30_000_000_000:  # 30B+
            return 16384
        elif parameter_count >= 10_000_000_000:  # 10B+
            return 8192
        elif parameter_count >= 5_000_000_000:  # 5B+
            return 4096
        elif parameter_count >= 1_000_000_000:  # 1B+
            return 2048
        else:
            return 1024
    
    def _estimate_capabilities(self, model_name: str, parameter_count: int) -> List[str]:
        """
        Estimate model capabilities based on name and size.
        
        Args:
            model_name: Model name
            parameter_count: Number of parameters
            
        Returns:
            List of capability strings
        """
        capabilities = ["chat"]  # All models support chat
        
        # Check for specific capabilities based on name
        model_name_lower = model_name.lower()
        
        if "code" in model_name_lower or "coder" in model_name_lower or parameter_count >= 7_000_000_000:
            capabilities.append("code")
            
        if parameter_count >= 10_000_000_000:
            capabilities.append("reasoning")
            
        if "opus" in model_name_lower or "gpt-4" in model_name_lower or parameter_count >= 70_000_000_000:
            capabilities.append("complex_reasoning")
            
        if "vision" in model_name_lower or "clip" in model_name_lower:
            capabilities.append("vision")
            
        return capabilities
    
    async def refresh_model_information(self) -> bool:
        """
        Refresh model information from Ollama API.
        
        Returns:
            Success status
        """
        try:
            # Clear existing models
            self.models = {}
            
            # Re-initialize from Ollama
            await self._initialize_models()
            
            # Update refresh timestamp
            self._last_refresh = time.time()
            
            self.logger.info(f"Refreshed model information. {len(self.models)} models available.")
            return True
        except Exception as e:
            self.logger.error(f"Error refreshing model information: {e}")
            return False
        
    def select_model_for_task(self, query: str, task_type: Optional[str] = None, 
                             prefer_low_latency: bool = False, 
                             max_parameters: Optional[int] = None) -> str:
        """
        Select the most appropriate model for a task.
        
        Args:
            query: User query
            task_type: Optional task type hint
            prefer_low_latency: Whether to prefer low-latency models
            max_parameters: Maximum parameter count
            
        Returns:
            Model ID
        """
        # Ensure models are initialized
        if not self._initialized:
            self.logger.warning("Models not initialized, using default model")
            return self.default_model_id
            
        if not self.models:
            self.logger.warning("No models available, using default model")
            return self.default_model_id
            
        # If only one model, return it
        if len(self.models) == 1:
            return next(iter(self.models))
            
        # Determine task type from query if not provided
        if not task_type:
            task_type = self._detect_task_type(query)
            
        # Filter models by task capability
        capable_models = self._filter_models_by_capability(task_type)
        
        # If no capable models, use all models
        if not capable_models:
            capable_models = list(self.models.keys())
            
        # Apply parameter constraint if specified
        if max_parameters:
            capable_models = [
                model_id for model_id in capable_models
                if self.models[model_id].get("parameters", 0) <= max_parameters
            ]
            
        # If no models left, return default
        if not capable_models:
            self.logger.warning(f"No models meet the constraints, using default model")
            return self.default_model_id
            
        # Consider performance metrics if available
        if self.model_performance and prefer_low_latency:
            # Sort by average response time if we have that data
            models_with_metrics = []
            for model_id in capable_models:
                if model_id in self.model_performance:
                    stats = self.model_performance[model_id]
                    if stats["total_queries"] > 0:
                        avg_time = stats["total_time"] / stats["total_queries"]
                        models_with_metrics.append((model_id, avg_time))
            
            if models_with_metrics:
                # Sort by average response time (ascending)
                models_with_metrics.sort(key=lambda x: x[1])
                return models_with_metrics[0][0]
            
        # If preferring low latency, pick the smallest capable model
        if prefer_low_latency:
            return min(capable_models, key=lambda m: self.models[m].get("parameters", 0))
            
        # Otherwise pick the largest capable model that fits constraints
        return max(capable_models, key=lambda m: self.models[m].get("parameters", 0))
    
    def _detect_task_type(self, query: str) -> str:
        """
        Detect the task type from a query.
        
        Args:
            query: User query
            
        Returns:
            Task type string
        """
        query_lower = query.lower()
        
        # Check for code generation task
        code_keywords = ["code", "function", "class", "algorithm", "script", 
                        "implement", "programming", "bug", "error", "fix",
                        "json", "html", "css", "javascript", "python", "sql",
                        "syntax", "compile", "runtime", "exception", "library",
                        "framework", "api", "sdk", "regex"]
        
        if any(kw in query_lower for kw in code_keywords) or re.search(r'```|def |class |function|import |module|package', query_lower):
            return "code"
        
        # Check for complex reasoning / analysis task
        reasoning_keywords = ["explain", "analyze", "compare", "critique", 
                             "evaluate", "design", "architecture", "system",
                             "complex", "problem", "solution", "optimize",
                             "strategy", "framework", "methodology", "approach",
                             "assessment", "recommendation", "proposal"]
                             
        if any(kw in query_lower for kw in reasoning_keywords) or len(query.split()) > 50:
            # Long queries often need more reasoning capability
            if len(query.split()) > 100:
                return "complex_reasoning"
            return "reasoning"
        
        # Check for vision/image analysis
        vision_keywords = ["image", "picture", "photo", "describe", "visual",
                           "looking at", "what's in", "can you see"]
        
        if any(kw in query_lower for kw in vision_keywords):
            return "vision"
            
        # Default to chat
        return "chat"
    
    def _filter_models_by_capability(self, task_type: str) -> List[str]:
        """
        Filter models by capability.
        
        Args:
            task_type: Task type to filter for
            
        Returns:
            List of model IDs with the capability
        """
        capability_map = {
            "code": "code",
            "reasoning": "reasoning",
            "complex_reasoning": "complex_reasoning",
            "vision": "vision",
            "chat": "chat"
        }
        
        capability = capability_map.get(task_type, "chat")
        
        # Special handling for complex_reasoning - models with complex_reasoning
        # can handle regular reasoning tasks too
        if capability == "reasoning":
            return [
                model_id for model_id, model_info in self.models.items()
                if "reasoning" in model_info.get("capabilities", []) or
                   "complex_reasoning" in model_info.get("capabilities", [])
            ]
        
        return [
            model_id for model_id, model_info in self.models.items()
            if capability in model_info.get("capabilities", [])
        ]
    
    def query_model(self, model_id: str, prompt: str, system_prompt: Optional[str] = None,
                   stream: bool = False, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Query a model with a prompt.
        
        Args:
            model_id: Model ID
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            conversation_history: Optional conversation history
            
        Returns:
            Response dictionary
            
        Raises:
            ModelNotFoundError: If the model is not found
            ModelQueryError: If there is an error querying the model
        """
        # Check if model exists
        if model_id not in self.models:
            available_models = ", ".join(self.models.keys())
            logger.error(f"Model {model_id} not found. Available models: {available_models}")
            raise ModelNotFoundError(f"Model {model_id} not found")
            
        try:
            # Log model query
            logger.info(f"Querying model {model_id} with prompt: {prompt[:50]}...")
            
            # Start measuring performance
            start_time = time.time()
            
            # Format the conversation history for Ollama
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
                
            # Add conversation history if provided
            if conversation_history:
                for message in conversation_history:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    
                    if role and content:
                        messages.append({
                            "role": role,
                            "content": content
                        })
            
            # Add the current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Prepare request payload
            payload = {
                "model": model_id,
                "messages": messages,
                "stream": stream
            }
            
            # Send request to Ollama API
            url = f"{self.ollama_api_url}/api/chat"
            
            if stream:
                # For streaming, we return a generator
                response = requests.post(url, json=payload, stream=True, timeout=self.request_timeout)
                response.raise_for_status()
                
                return response.iter_lines()
            else:
                # For non-streaming, we return a complete response
                response = requests.post(url, json=payload, timeout=self.request_timeout)
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                
                # Extract response text
                generated_text = response_data.get("message", {}).get("content", "")
                
                # Calculate performance metrics
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Estimate token count (crude approximation)
                prompt_tokens = len(prompt.split())
                response_tokens = len(generated_text.split())
                total_tokens = prompt_tokens + response_tokens
                
                # Update model performance cache
                if model_id not in self.model_performance:
                    self.model_performance[model_id] = {
                        "total_queries": 0,
                        "total_tokens": 0,
                        "total_time": 0
                    }
                    
                self.model_performance[model_id]["total_queries"] += 1
                self.model_performance[model_id]["total_tokens"] += total_tokens
                self.model_performance[model_id]["total_time"] += processing_time
                
                # Prepare response
                result = {
                    "response": generated_text,
                    "tokens_used": total_tokens,
                    "processing_time": processing_time
                }
                
                return result
                
        except requests.RequestException as e:
            logger.error(f"Error querying model {model_id}: {e}")
            
            # For actual production, we'd want to implement more robust fallback strategies
            fallback_response = {
                "response": "I apologize, but I'm currently experiencing technical difficulties connecting to the language model. Please try again in a moment.",
                "tokens_used": 20,
                "error": str(e)
            }
            
            raise ModelQueryError(f"Error querying model: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error querying model {model_id}: {e}")
            raise ModelQueryError(f"Unexpected error: {e}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        
        Returns:
            List of model information dictionaries
        """
        return [
            {
                "id": model_id,
                **model_info
            }
            for model_id, model_info in self.models.items()
        ]
    
    def get_model_performance(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics for models.
        
        Args:
            model_id: Optional model ID to get stats for
            
        Returns:
            Performance statistics dictionary
        """
        if model_id:
            if model_id not in self.model_performance:
                return {"error": f"No performance data available for model {model_id}"}
                
            stats = self.model_performance[model_id]
            
            # Calculate derived metrics
            avg_time = stats["total_time"] / stats["total_queries"] if stats["total_queries"] > 0 else 0
            avg_tokens = stats["total_tokens"] / stats["total_queries"] if stats["total_queries"] > 0 else 0
            
            return {
                "model_id": model_id,
                "total_queries": stats["total_queries"],
                "total_tokens": stats["total_tokens"],
                "avg_response_time": avg_time,
                "avg_tokens_per_query": avg_tokens
            }
        else:
            # Return stats for all models
            all_stats = {}
            
            for model_id, stats in self.model_performance.items():
                avg_time = stats["total_time"] / stats["total_queries"] if stats["total_queries"] > 0 else 0
                avg_tokens = stats["total_tokens"] / stats["total_queries"] if stats["total_queries"] > 0 else 0
                
                all_stats[model_id] = {
                    "total_queries": stats["total_queries"],
                    "total_tokens": stats["total_tokens"],
                    "avg_response_time": avg_time,
                    "avg_tokens_per_query": avg_tokens
                }
                
            return all_stats
