"""
Logging utilities for NestAI.
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Union


class AILogger:
    """
    Logger for AI operations.
    """
    
    def __init__(
        self,
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        log_format: Optional[str] = None,
        console_output: bool = True,
        structured_logging: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            log_level: Logging level
            log_file: Log file path
            log_format: Log format
            console_output: Whether to output to console
            structured_logging: Whether to use structured logging
        """
        self.logger = logging.getLogger("nestai")
        self.logger.setLevel(log_level)
        self.structured_logging = structured_logging
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Set up console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            
            if log_format:
                formatter = logging.Formatter(log_format)
            else:
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Set up file handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            if log_format:
                formatter = logging.Formatter(log_format)
            else:
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _format_message(self, message: str, **kwargs) -> str:
        """
        Format a log message with additional data.
        
        Args:
            message: The message
            **kwargs: Additional data
            
        Returns:
            The formatted message
        """
        if not self.structured_logging or not kwargs:
            return message
        
        data = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        return json.dumps(data)
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message.
        
        Args:
            message: The message
            **kwargs: Additional data
        """
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: The message
            **kwargs: Additional data
        """
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: The message
            **kwargs: Additional data
        """
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: The message
            exc_info: Whether to include exception info
            **kwargs: Additional data
        """
        self.logger.error(self._format_message(message, **kwargs), exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """
        Log a critical message.
        
        Args:
            message: The message
            exc_info: Whether to include exception info
            **kwargs: Additional data
        """
        self.logger.critical(self._format_message(message, **kwargs), exc_info=exc_info)
    
    def log_request(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log an AI request.
        
        Args:
            provider: The provider
            model: The model
            prompt: The prompt
            system_prompt: The system prompt
            request_id: The request ID
            **kwargs: Additional data
        """
        data = {
            "provider": provider,
            "model": model,
            "prompt_length": len(prompt),
            "request_id": request_id,
            **kwargs
        }
        
        if system_prompt:
            data["system_prompt_length"] = len(system_prompt)
        
        self.info(f"AI request to {provider}/{model}", **data)
    
    def log_response(
        self,
        provider: str,
        model: str,
        response: str,
        latency: float,
        tokens: Dict[str, int],
        request_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log an AI response.
        
        Args:
            provider: The provider
            model: The model
            response: The response
            latency: The latency in seconds
            tokens: Token usage information
            request_id: The request ID
            **kwargs: Additional data
        """
        data = {
            "provider": provider,
            "model": model,
            "response_length": len(response),
            "latency": latency,
            "tokens": tokens,
            "request_id": request_id,
            **kwargs
        }
        
        self.info(f"AI response from {provider}/{model}", **data)

