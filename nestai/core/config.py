"""
Configuration classes for NestAI.
"""

from typing import Dict, Any, Optional, List


class AIConfig:
    """
    Configuration for AI models.
    """
    
    def __init__(
        self,
        default_provider: str = "openai",
        default_model: str = "gpt-3.5-turbo",
        api_keys: Optional[Dict[str, str]] = None,
        plugins: Optional[List[Dict[str, Any]]] = None,
        presets: Optional[Dict[str, "PresetConfig"]] = None,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        semantic_cache_enabled: bool = True,
        semantic_cache_threshold: float = 0.9,
        max_concurrent_requests: int = 10,
        timeout: int = 60,
        retry_attempts: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the AI configuration.
        
        Args:
            default_provider: Default provider to use
            default_model: Default model to use
            api_keys: API keys for providers
            plugins: Plugin configurations
            presets: Preset configurations
            cache_enabled: Whether caching is enabled
            cache_ttl: Time-to-live for cache entries in seconds
            semantic_cache_enabled: Whether semantic caching is enabled
            semantic_cache_threshold: Threshold for semantic similarity
            max_concurrent_requests: Maximum number of concurrent requests
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.default_provider = default_provider
        self.default_model = default_model
        self.api_keys = api_keys or {}
        self.plugins = plugins or []
        self.presets = presets or {}
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.semantic_cache_enabled = semantic_cache_enabled
        self.semantic_cache_threshold = semantic_cache_threshold
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            A dictionary representation of the configuration
        """
        return {
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "api_keys": self.api_keys,
            "plugins": self.plugins,
            "presets": {name: preset.to_dict() for name, preset in self.presets.items()},
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "semantic_cache_enabled": self.semantic_cache_enabled,
            "semantic_cache_threshold": self.semantic_cache_threshold,
            "max_concurrent_requests": self.max_concurrent_requests,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            data: The dictionary data
            
        Returns:
            An AIConfig instance
        """
        presets = {}
        if "presets" in data:
            for name, preset_data in data["presets"].items():
                presets[name] = PresetConfig.from_dict(preset_data)
        
        return cls(
            default_provider=data.get("default_provider", "openai"),
            default_model=data.get("default_model", "gpt-3.5-turbo"),
            api_keys=data.get("api_keys", {}),
            plugins=data.get("plugins", []),
            presets=presets,
            cache_enabled=data.get("cache_enabled", True),
            cache_ttl=data.get("cache_ttl", 3600),
            semantic_cache_enabled=data.get("semantic_cache_enabled", True),
            semantic_cache_threshold=data.get("semantic_cache_threshold", 0.9),
            max_concurrent_requests=data.get("max_concurrent_requests", 10),
            timeout=data.get("timeout", 60),
            retry_attempts=data.get("retry_attempts", 3),
            retry_delay=data.get("retry_delay", 1)
        )


class PresetConfig:
    """
    Preset configuration for AI models.
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the preset configuration.
        
        Args:
            provider: Provider to use
            model: Model to use
            system_prompt: System prompt
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop_sequences: Stop sequences
            additional_params: Additional parameters
        """
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop_sequences = stop_sequences
        self.additional_params = additional_params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the preset to a dictionary.
        
        Returns:
            A dictionary representation of the preset
        """
        result = {}
        
        if self.provider is not None:
            result["provider"] = self.provider
        
        if self.model is not None:
            result["model"] = self.model
        
        if self.system_prompt is not None:
            result["system_prompt"] = self.system_prompt
        
        if self.temperature is not None:
            result["temperature"] = self.temperature
        
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        
        if self.top_p is not None:
            result["top_p"] = self.top_p
        
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty
        
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty
        
        if self.stop_sequences is not None:
            result["stop_sequences"] = self.stop_sequences
        
        if self.additional_params:
            result["additional_params"] = self.additional_params
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PresetConfig":
        """
        Create a preset from a dictionary.
        
        Args:
            data: The dictionary data
            
        Returns:
            A PresetConfig instance
        """
        return cls(
            provider=data.get("provider"),
            model=data.get("model"),
            system_prompt=data.get("system_prompt"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            top_p=data.get("top_p"),
            frequency_penalty=data.get("frequency_penalty"),
            presence_penalty=data.get("presence_penalty"),
            stop_sequences=data.get("stop_sequences"),
            additional_params=data.get("additional_params", {})
        )


# Predefined presets
CREATIVE_PRESET = PresetConfig(
    temperature=0.9,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    system_prompt="You are a creative assistant. Be imaginative and original in your responses."
)

BALANCED_PRESET = PresetConfig(
    temperature=0.7,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    system_prompt="You are a helpful assistant. Provide balanced and informative responses."
)

PRECISE_PRESET = PresetConfig(
    temperature=0.2,
    top_p=0.8,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    system_prompt="You are a precise assistant. Provide accurate and concise responses."
)

CODE_PRESET = PresetConfig(
    temperature=0.2,
    top_p=0.95,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    system_prompt="You are a coding assistant. Provide clean, efficient, and well-documented code."
)

# Default presets
DEFAULT_PRESETS = {
    "creative": CREATIVE_PRESET,
    "balanced": BALANCED_PRESET,
    "precise": PRECISE_PRESET,
    "code": CODE_PRESET
}

