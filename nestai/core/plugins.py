"""
Plugin system for NestAI.
"""

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Type, Tuple


class Plugin(ABC):
    """
    Base class for NestAI plugins.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the plugin.
        
        Returns:
            The plugin name
        """
        pass
    
    @property
    def version(self) -> str:
        """
        Get the version of the plugin.
        
        Returns:
            The plugin version
        """
        return "1.0.0"
    
    @property
    def description(self) -> str:
        """
        Get the description of the plugin.
        
        Returns:
            The plugin description
        """
        return "A NestAI plugin"
    
    @property
    def plugin_type(self) -> str:
        """
        Get the type of the plugin.
        
        Returns:
            The plugin type
        """
        return "generic"
    
    def initialize(self, options: Dict[str, Any]) -> None:
        """
        Initialize the plugin with options.
        
        Args:
            options: Plugin options
        """
        pass
    
    def process_prompt(self, prompt: str, system_prompt: Optional[str], **kwargs) -> Dict[str, Any]:
        """
        Process a prompt before sending it to the AI model.
        
        Args:
            prompt: The prompt
            system_prompt: The system prompt
            **kwargs: Additional options
            
        Returns:
            A dictionary with the processed prompt and system prompt
        """
        return {
            "prompt": prompt,
            "system_prompt": system_prompt
        }
    
    def process_response(self, response: str, **kwargs) -> str:
        """
        Process a response from the AI model.
        
        Args:
            response: The response
            **kwargs: Additional options
            
        Returns:
            The processed response
        """
        return response
    
    def on_error(self, error: Exception, **kwargs) -> None:
        """
        Handle an error.
        
        Args:
            error: The error
            **kwargs: Additional options
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the plugin.
        """
        pass


class PromptPlugin(Plugin):
    """
    Plugin for processing prompts.
    """
    
    @property
    def plugin_type(self) -> str:
        """
        Get the type of the plugin.
        
        Returns:
            The plugin type
        """
        return "prompt"


class ResponsePlugin(Plugin):
    """
    Plugin for processing responses.
    """
    
    @property
    def plugin_type(self) -> str:
        """
        Get the type of the plugin.
        
        Returns:
            The plugin type
        """
        return "response"


class ErrorPlugin(Plugin):
    """
    Plugin for handling errors.
    """
    
    @property
    def plugin_type(self) -> str:
        """
        Get the type of the plugin.
        
        Returns:
            The plugin type
        """
        return "error"


class PluginManager:
    """
    Manager for NestAI plugins.
    """
    
    def __init__(self):
        """
        Initialize the plugin manager.
        """
        self.plugins: Dict[str, Plugin] = {}
        self.prompt_plugins: List[Plugin] = []
        self.response_plugins: List[Plugin] = []
        self.error_plugins: List[Plugin] = []
    
    def register_plugin(self, plugin: Plugin) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: The plugin to register
        """
        self.plugins[plugin.name] = plugin
        
        if plugin.plugin_type == "prompt" or plugin.plugin_type == "generic":
            self.prompt_plugins.append(plugin)
        
        if plugin.plugin_type == "response" or plugin.plugin_type == "generic":
            self.response_plugins.append(plugin)
        
        if plugin.plugin_type == "error" or plugin.plugin_type == "generic":
            self.error_plugins.append(plugin)
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: The name of the plugin to unregister
        """
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            
            if plugin in self.prompt_plugins:
                self.prompt_plugins.remove(plugin)
            
            if plugin in self.response_plugins:
                self.response_plugins.remove(plugin)
            
            if plugin in self.error_plugins:
                self.error_plugins.remove(plugin)
            
            plugin.cleanup()
            del self.plugins[plugin_name]
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: The name of the plugin
            
        Returns:
            The plugin, or None if not found
        """
        return self.plugins.get(plugin_name)
    
    def process_prompt(self, prompt: str, system_prompt: Optional[str], **kwargs) -> Dict[str, Any]:
        """
        Process a prompt through all prompt plugins.
        
        Args:
            prompt: The prompt
            system_prompt: The system prompt
            **kwargs: Additional options
            
        Returns:
            A dictionary with the processed prompt and system prompt
        """
        result = {
            "prompt": prompt,
            "system_prompt": system_prompt
        }
        
        for plugin in self.prompt_plugins:
            try:
                plugin_result = plugin.process_prompt(
                    result["prompt"],
                    result["system_prompt"],
                    **kwargs
                )
                
                result["prompt"] = plugin_result["prompt"]
                result["system_prompt"] = plugin_result["system_prompt"]
            except Exception as e:
                print(f"Error in plugin {plugin.name}: {str(e)}")
        
        return result
    
    def process_response(self, response: str, **kwargs) -> str:
        """
        Process a response through all response plugins.
        
        Args:
            response: The response
            **kwargs: Additional options
            
        Returns:
            The processed response
        """
        result = response
        
        for plugin in self.response_plugins:
            try:
                result = plugin.process_response(result, **kwargs)
            except Exception as e:
                print(f"Error in plugin {plugin.name}: {str(e)}")
        
        return result
    
    def handle_error(self, error: Exception, **kwargs) -> None:
        """
        Handle an error through all error plugins.
        
        Args:
            error: The error
            **kwargs: Additional options
        """
        for plugin in self.error_plugins:
            try:
                plugin.on_error(error, **kwargs)
            except Exception as e:
                print(f"Error in plugin {plugin.name}: {str(e)}")
    
    def load_plugin_from_path(self, path: str, class_name: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Optional[Plugin]:
        """
        Load a plugin from a module path.
        
        Args:
            path: The module path
            class_name: The plugin class name
            options: Plugin options
            
        Returns:
            The loaded plugin, or None if loading failed
        """
        try:
            module = importlib.import_module(path)
            
            if class_name:
                # Load specific class
                if hasattr(module, class_name):
                    plugin_class = getattr(module, class_name)
                    if inspect.isclass(plugin_class) and issubclass(plugin_class, Plugin):
                        plugin = plugin_class()
                        if options:
                            plugin.initialize(options)
                        return plugin
            else:
                # Find first Plugin subclass
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, Plugin) and obj != Plugin and obj != PromptPlugin and obj != ResponsePlugin and obj != ErrorPlugin:
                        plugin = obj()
                        if options:
                            plugin.initialize(options)
                        return plugin
        except Exception as e:
            print(f"Error loading plugin from {path}: {str(e)}")
        
        return None
    
    def cleanup(self) -> None:
        """
        Clean up all plugins.
        """
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                print(f"Error cleaning up plugin {plugin.name}: {str(e)}")
        
        self.plugins = {}
        self.prompt_plugins = []
        self.response_plugins = []
        self.error_plugins = []


# Example plugins

class DebugPlugin(Plugin):
    """
    Plugin for debugging.
    """
    
    @property
    def name(self) -> str:
        return "debug"
    
    @property
    def description(self) -> str:
        return "Prints debug information for prompts and responses"
    
    def process_prompt(self, prompt: str, system_prompt: Optional[str], **kwargs) -> Dict[str, Any]:
        print(f"DEBUG - Prompt: {prompt[:100]}...")
        if system_prompt:
            print(f"DEBUG - System Prompt: {system_prompt[:100]}...")
        return {"prompt": prompt, "system_prompt": system_prompt}
    
    def process_response(self, response: str, **kwargs) -> str:
        print(f"DEBUG - Response: {response[:100]}...")
        return response


class ProfanityFilterPlugin(PromptPlugin):
    """
    Plugin for filtering profanity.
    """
    
    def __init__(self):
        self.profanity_list = [
            "badword1",
            "badword2",
            "badword3"
        ]
    
    @property
    def name(self) -> str:
        return "profanity_filter"
    
    @property
    def description(self) -> str:
        return "Filters profanity from prompts"
    
    def process_prompt(self, prompt: str, system_prompt: Optional[str], **kwargs) -> Dict[str, Any]:
        filtered_prompt = prompt
        for word in self.profanity_list:
            filtered_prompt = filtered_prompt.replace(word, "[FILTERED]")
        
        return {"prompt": filtered_prompt, "system_prompt": system_prompt}


class ResponseFormatterPlugin(ResponsePlugin):
    """
    Plugin for formatting responses.
    """
    
    def __init__(self):
        self.format_type = "default"
    
    @property
    def name(self) -> str:
        return "response_formatter"
    
    @property
    def description(self) -> str:
        return "Formats responses"
    
    def initialize(self, options: Dict[str, Any]) -> None:
        if "format_type" in options:
            self.format_type = options["format_type"]
    
    def process_response(self, response: str, **kwargs) -> str:
        if self.format_type == "uppercase":
            return response.upper()
        elif self.format_type == "lowercase":
            return response.lower()
        elif self.format_type == "title":
            return response.title()
        else:
            return response

