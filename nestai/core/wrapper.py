"""
Core NestAI implementation.
"""

import asyncio
from typing import Dict, List, Any, Callable, Optional, Union, Set
from datetime import datetime
import uuid
import numpy as np
import logging
import json
import os
from abc import ABC, abstractmethod

from nestai.utils.metrics import MetricsTracker
from nestai.utils.semantic_cache import SemanticCache
from nestai.utils.logging import AILogger
from nestai.utils.security import PIIDetector, DataEncryptor, SecurityAuditor
from nestai.utils.transparency import DecisionLogger, CostEstimator
from nestai.core.plugins import PluginManager, Plugin
from nestai.core.config import AIConfig, PresetConfig


class NestAI:
    """
    Universal NestAI - 2025 Edition
    Combines simplicity with advanced features for AI model integration.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        monitoring: Optional[Dict[str, Any]] = None
    ):
        # Initialize core components
        self.metrics = MetricsTracker()
        self.cache = SemanticCache()
        self.logger = AILogger()
        self.security = PIIDetector()
        self.encryptor = DataEncryptor()
        self.auditor = SecurityAuditor()
        self.decision_logger = DecisionLogger()
        self.cost_estimator = CostEstimator()
        self.plugin_manager = PluginManager()
        self.resource_manager = ResourceManager()
        self.event_system = EventSystem()
        
        # Configuration
        self.config = config or {}
        self.default_provider = self.config.get("default_provider", "openai")
        self.default_model = self.config.get("default_model", "gpt-3.5-turbo")
        self.api_keys = self.config.get("api_keys", {})
        
        # Set up monitoring if provided
        if monitoring:
            self._setup_monitoring(monitoring)
            
        # Load plugins if specified
        if "plugins" in self.config:
            self._load_plugins(self.config["plugins"])
    
    def _setup_monitoring(self, monitoring: Dict[str, Any]) -> None:
        """Set up monitoring configuration."""
        if "log_level" in monitoring:
            self.logger = AILogger(log_level=getattr(logging, monitoring["log_level"]))
        
        if "metrics_dir" in monitoring:
            self.metrics = MetricsTracker(metrics_dir=monitoring["metrics_dir"])
            
        if "audit_dir" in monitoring:
            self.auditor = SecurityAuditor(audit_dir=monitoring["audit_dir"])
    
    def _load_plugins(self, plugins_config: List[Dict[str, Any]]) -> None:
        """Load plugins from configuration."""
        for plugin_config in plugins_config:
            plugin_path = plugin_config.get("path")
            plugin_class = plugin_config.get("class")
            plugin_options = plugin_config.get("options", {})
            
            if plugin_path:
                plugin = self.plugin_manager.load_plugin_from_path(
                    plugin_path, plugin_class, plugin_options
                )
                if plugin:
                    self.plugin_manager.register_plugin(plugin)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Synchronous entry point for simple usage.
        
        Args:
            prompt: The prompt to send to the AI model
            **kwargs: Additional options
            
        Returns:
            The generated response
        """
        return asyncio.run(self.async_call(prompt, **kwargs))
    
    async def async_call(self, prompt: str, **kwargs) -> str:
        """
        Asynchronous entry point for simple usage.
        
        Args:
            prompt: The prompt to send to the AI model
            **kwargs: Additional options
            
        Returns:
            The generated response
        """
        async with self.resource_manager:
            return await self.process_request(prompt, **kwargs)
    
    async def process_request(self, prompt: str, **kwargs) -> str:
        """
        Process an AI request.
        
        Args:
            prompt: The prompt to send to the AI model
            **kwargs: Additional options
            
        Returns:
            The generated response
        """
        # Generate request ID and start timing
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Extract common parameters
        system_prompt = kwargs.get("system")
        provider = kwargs.get("provider", self.default_provider)
        model = kwargs.get("model", self.default_model)
        preset = kwargs.get("preset")
        stream = kwargs.get("stream", False)
        optimize_cost = kwargs.get("optimize_cost", True)
        use_cache = kwargs.get("cache", True)
        
        # Log the request
        self.logger.log_request(
            provider=provider,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            request_id=request_id
        )
        
        try:
            # Apply security checks
            sanitized_prompt = prompt
            if kwargs.get("security_checks", True):
                redacted_prompt, detected_pii = self.security.redact(prompt)
                if detected_pii:
                    self.logger.warning(
                        f"PII detected in prompt",
                        pii_types=[item["type"] for item in detected_pii],
                        request_id=request_id
                    )
                    self.auditor.log_event("pii_detected", {
                        "request_id": request_id,
                        "pii_types": [item["type"] for item in detected_pii]
                    })
                    
                    if kwargs.get("redact_pii", True):
                        sanitized_prompt = redacted_prompt
                        self.logger.info("PII redacted from prompt", request_id=request_id)
            
            # Apply plugins to prompt
            if self.plugin_manager.prompt_plugins:
                plugin_result = self.plugin_manager.process_prompt(
                    sanitized_prompt, system_prompt, **kwargs
                )
                sanitized_prompt = plugin_result["prompt"]
                system_prompt = plugin_result["system_prompt"]
            
            # Check cache if enabled
            if use_cache:
                cache_key = self.cache.generate_key(
                    sanitized_prompt, system_prompt, model, provider, preset, kwargs
                )
                cached_response, is_semantic = await self._check_cache(cache_key, sanitized_prompt)
                
                if cached_response:
                    # Record metrics for cache hit
                    self.metrics.track_request(
                        provider=provider,
                        model=model,
                        prompt_tokens=self.cost_estimator.estimate_tokens(sanitized_prompt),
                        completion_tokens=self.cost_estimator.estimate_tokens(cached_response),
                        latency=0.01,  # Negligible latency for cache hit
                        cost=0.0,  # No cost for cache hit
                        cached=True,
                        semantic_cache=is_semantic
                    )
                    
                    self.logger.info(
                        f"Cache hit for request {request_id}",
                        semantic=is_semantic,
                        request_id=request_id
                    )
                    
                    # Apply response plugins
                    if self.plugin_manager.response_plugins:
                        cached_response = self.plugin_manager.process_response(
                            cached_response, **kwargs
                        )
                    
                    # Emit event
                    self.event_system.emit("request_completed", {
                        "request_id": request_id,
                        "cached": True,
                        "semantic": is_semantic
                    })
                    
                    return cached_response
            
            # Optimize provider/model selection if enabled
            if optimize_cost:
                selected_provider, selected_model = await self._optimize_provider_model(
                    sanitized_prompt, system_prompt, provider, model
                )
                
                if selected_provider != provider or selected_model != model:
                    self.logger.info(
                        f"Provider/model optimized: {provider}/{model} -> {selected_provider}/{selected_model}",
                        request_id=request_id
                    )
                    
                    self.decision_logger.log_decision(
                        decision_type="provider_model_optimization",
                        context={
                            "original_provider": provider,
                            "original_model": model,
                            "prompt_length": len(sanitized_prompt),
                            "system_prompt_length": len(system_prompt) if system_prompt else 0
                        },
                        outcome={
                            "provider": selected_provider,
                            "model": selected_model
                        },
                        explanation="Selected based on cost optimization and capability requirements"
                    )
                    
                    provider = selected_provider
                    model = selected_model
            
            # Process with selected provider/model
            response, usage = await self._process_with_provider(
                sanitized_prompt,
                system_prompt,
                provider,
                model,
                stream,
                **kwargs
            )
            
            # Calculate metrics
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            # Estimate cost
            cost = self._calculate_cost(provider, model, usage)
            
            # Record metrics
            self.metrics.track_request(
                provider=provider,
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency=latency,
                cost=cost,
                cached=False
            )
            
            # Log response
            self.logger.log_response(
                provider=provider,
                model=model,
                response=response[:1000] + "..." if len(response) > 1000 else response,
                latency=latency,
                tokens=usage,
                request_id=request_id
            )
            
            # Store in cache if enabled
            if use_cache:
                await self._store_in_cache(cache_key, sanitized_prompt, system_prompt, response)
            
            # Apply response plugins
            if self.plugin_manager.response_plugins:
                response = self.plugin_manager.process_response(response, **kwargs)
            
            # Emit event
            self.event_system.emit("request_completed", {
                "request_id": request_id,
                "cached": False,
                "latency": latency,
                "cost": cost
            })
            
            return response
            
        except Exception as e:
            # Log error
            self.logger.error(
                f"Error processing request: {str(e)}",
                exc_info=True,
                request_id=request_id
            )
            
            # Track error in metrics
            self.metrics.track_error(
                error_type=type(e).__name__,
                provider=provider,
                model=model
            )
            
            # Emit event
            self.event_system.emit("request_error", {
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            # Re-raise the exception
            raise
    
    async def _check_cache(self, cache_key: str, prompt: str) -> tuple[Optional[str], bool]:
        """
        Check if a response is in the cache.
        
        Args:
            cache_key: The cache key
            prompt: The original prompt
            
        Returns:
            A tuple of (cached_response, is_semantic_match) or (None, False) if not found
        """
        # Try exact match first
        exact_match = self.cache.get(cache_key)
        if exact_match:
            return exact_match, False
        
        # Try semantic match
        semantic_result = await self.cache._find_semantic_match(prompt)
        if semantic_result:
            return semantic_result
        
        return None, False
    
    async def _store_in_cache(self, cache_key: str, prompt: str, system_prompt: Optional[str], response: str) -> None:
        """
        Store a response in the cache.
        
        Args:
            cache_key: The cache key
            prompt: The original prompt
            system_prompt: The system prompt
            response: The response to cache
        """
        self.cache.set(cache_key, prompt, system_prompt, response)
    
    async def _optimize_provider_model(
        self, 
        prompt: str, 
        system_prompt: Optional[str],
        current_provider: str,
        current_model: str
    ) -> tuple[str, str]:
        """
        Optimize provider and model selection based on cost and capabilities.
        
        Args:
            prompt: The prompt
            system_prompt: The system prompt
            current_provider: The current provider
            current_model: The current model
            
        Returns:
            A tuple of (provider, model)
        """
        # Get cost comparison
        comparison = self.cost_estimator.compare_costs(prompt, system_prompt)
        
        # Find the cheapest option that meets requirements
        cheapest_provider = current_provider
        cheapest_model = current_model
        lowest_cost = float('inf')
        
        for provider, models in comparison.items():
            for model, estimate in models.items():
                # Skip if we don't have an API key for this provider
                if provider not in self.api_keys and provider != current_provider:
                    continue
                
                # Check if this model is cheaper
                if estimate["estimated_cost"] < lowest_cost:
                    cheapest_provider = provider
                    cheapest_model = model
                    lowest_cost = estimate["estimated_cost"]
        
        return cheapest_provider, cheapest_model
    
    async def _process_with_provider(
        self,
        prompt: str,
        system_prompt: Optional[str],
        provider: str,
        model: str,
        stream: bool,
        **kwargs
    ) -> tuple[str, Dict[str, int]]:
        """
        Process a request with a specific provider and model.
        
        Args:
            prompt: The prompt
            system_prompt: The system prompt
            provider: The provider
            model: The model
            stream: Whether to stream the response
            **kwargs: Additional options
            
        Returns:
            A tuple of (response, usage)
        """
        # This is a placeholder - in a real implementation, this would use the actual provider
        # For now, we'll simulate a response
        import time
        import random
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Simulate a response
        response = f"This is a simulated response to: {prompt}"
        
        # Simulate token usage
        usage = {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(response) // 4,
            "total_tokens": (len(prompt) + len(response)) // 4
        }
        
        return response, usage
    
    def _calculate_cost(self, provider: str, model: str, usage: Dict[str, int]) -> float:
        """
        Calculate the cost of a request.
        
        Args:
            provider: The provider
            model: The model
            usage: Token usage information
            
        Returns:
            The cost in USD
        """
        # Get cost per token
        provider_costs = self.cost_estimator.provider_costs.get(provider.lower(), {})
        cost_per_1k_tokens = provider_costs.get(model, 0.01)  # Default to $0.01 per 1K tokens
        
        # Calculate cost
        total_tokens = usage.get("total_tokens", 0)
        return (total_tokens / 1000) * cost_per_1k_tokens
    
    def batch(self) -> 'AIBatchProcessor':
        """
        Create a batch processor.
        
        Returns:
            A batch processor
        """
        return AIBatchProcessor(self)
    
    def conversation(self) -> 'AIConversationManager':
        """
        Create a conversation manager.
        
        Returns:
            A conversation manager
        """
        return AIConversationManager(self)
    
    def pipeline(self, steps: List[Callable]) -> 'AIPipeline':
        """
        Create a processing pipeline.
        
        Args:
            steps: The processing steps
            
        Returns:
            A processing pipeline
        """
        return AIPipeline(self, steps)


class ResourceManager:
    """
    Manages resources and concurrency.
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize the resource manager.
        
        Args:
            max_concurrent: Maximum number of concurrent requests
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
    
    async def __aenter__(self):
        """Enter the context manager."""
        await self.semaphore.acquire()
        self.active_requests += 1
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.active_requests -= 1
        self.semaphore.release()


class EventSystem:
    """
    Event system for callbacks.
    """
    
    def __init__(self):
        """Initialize the event system."""
        self.listeners: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, callback: Callable) -> None:
        """
        Register a callback for an event.
        
        Args:
            event: The event name
            callback: The callback function
        """
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(callback)
    
    def emit(self, event: str, data: Any) -> None:
        """
        Emit an event.
        
        Args:
            event: The event name
            data: The event data
        """
        for callback in self.listeners.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logging.error(f"Error in event callback: {str(e)}")


class AIBatchProcessor:
    """
    Batch processor for AI requests.
    """
    
    def __init__(self, wrapper: NestAI):
        """
        Initialize the batch processor.
        
        Args:
            wrapper: The NestAI instance
        """
        self.wrapper = wrapper
        self.queue: List[Dict[str, Any]] = []
    
    def add(self, prompt: str, **kwargs) -> None:
        """
        Add a request to the batch.
        
        Args:
            prompt: The prompt
            **kwargs: Additional options
        """
        self.queue.append({"prompt": prompt, "options": kwargs})
    
    async def process(self) -> List[str]:
        """
        Process all requests in the batch.
        
        Returns:
            A list of responses
        """
        tasks = []
        for item in self.queue:
            task = asyncio.create_task(
                self.wrapper.process_request(item["prompt"], **item["options"])
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def __aenter__(self) -> 'AIBatchProcessor':
        """Enter the context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        self.queue = []


class AIConversationManager:
    """
    Manages AI conversations with context and memory.
    """
    
    def __init__(self, wrapper: NestAI):
        """
        Initialize the conversation manager.
        
        Args:
            wrapper: The NestAI instance
        """
        self.wrapper = wrapper
        self.context: List[str] = []
        self.memory: List[Dict[str, str]] = []
        self.system_prompt: Optional[str] = None
    
    def add_context(self, context: str) -> None:
        """
        Add context to the conversation.
        
        Args:
            context: The context to add
        """
        self.context.append(context)
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set the system prompt for the conversation.
        
        Args:
            system_prompt: The system prompt
        """
        self.system_prompt = system_prompt
    
    def _build_prompt(self, prompt: str) -> str:
        """
        Build a prompt with context and memory.
        
        Args:
            prompt: The user prompt
            
        Returns:
            The full prompt
        """
        # Build context
        context_str = "\n".join(self.context)
        
        # Build memory
        memory_str = ""
        for item in self.memory:
            memory_str += f"User: {item['prompt']}\nAssistant: {item['response']}\n"
        
        # Combine everything
        if context_str and memory_str:
            return f"{context_str}\n\n{memory_str}\nUser: {prompt}"
        elif context_str:
            return f"{context_str}\n\nUser: {prompt}"
        elif memory_str:
            return f"{memory_str}\nUser: {prompt}"
        else:
            return prompt
    
    async def ask(self, prompt: str, **kwargs) -> str:
        """
        Ask a question in the conversation.
        
        Args:
            prompt: The prompt
            **kwargs: Additional options
            
        Returns:
            The response
        """
        full_prompt = self._build_prompt(prompt)
        
        # Merge options with system prompt
        options = dict(kwargs)
        if self.system_prompt and "system" not in options:
            options["system"] = self.system_prompt
        
        response = await self.wrapper.process_request(full_prompt, **options)
        self.memory.append({"prompt": prompt, "response": response})
        return response
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory = []
    
    def __enter__(self) -> 'AIConversationManager':
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        pass


class AIPipeline:
    """
    Processing pipeline for AI requests.
    """
    
    def __init__(self, wrapper: NestAI, steps: List[Callable]):
        """
        Initialize the pipeline.
        
        Args:
            wrapper: The NestAI instance
            steps: The processing steps
        """
        self.wrapper = wrapper
        self.steps = steps
    
    async def run(self, input_data: Any) -> Any:
        """
        Run the pipeline.
        
        Args:
            input_data: The input data
            
        Returns:
            The processed data
        """
        result = input_data
        for step in self.steps:
            if asyncio.iscoroutinefunction(step):
                result = await step(result)
            else:
                result = step(result)
        return result


# Create a global instance for simple usage
nestai = NestAI()

# Simple API functions
async def async_nest(prompt: str, **kwargs) -> str:
    """
    Asynchronous API for AI requests.
    
    Args:
        prompt: The prompt
        **kwargs: Additional options
        
    Returns:
        The response
    """
    return await nestai.async_call(prompt, **kwargs)

def nest(prompt: str, **kwargs) -> str:
    """
    Synchronous API for AI requests.
    
    Args:
        prompt: The prompt
        **kwargs: Additional options
        
    Returns:
        The response
    """
    return nestai(prompt, **kwargs)

def set_api_key(provider: str, api_key: str) -> None:
    """
    Set an API key for a provider.
    
    Args:
        provider: The provider
        api_key: The API key
    """
    nestai.api_keys[provider] = api_key
    os.environ[f"{provider.upper()}_API_KEY"] = api_key

def create_conversation() -> AIConversationManager:
    """
    Create a conversation manager.
    
    Returns:
        A conversation manager
    """
    return nestai.conversation()

def create_batch() -> AIBatchProcessor:
    """
    Create a batch processor.
    
    Returns:
        A batch processor
    """
    return nestai.batch()

def create_pipeline(steps: List[Callable]) -> AIPipeline:
    """
    Create a processing pipeline.
    
    Args:
        steps: The processing steps
        
    Returns:
        A processing pipeline
    """
    return nestai.pipeline(steps)

