"""
Transparency utilities for NestAI.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set, Callable


class DecisionLogger:
    """
    Logs decisions made by the AI system.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the decision logger.
        
        Args:
            log_dir: Directory for storing decision logs
        """
        self.log_dir = log_dir
        
        # Create log directory if needed
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
    
    def log_decision(
        self,
        decision_type: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any],
        explanation: Optional[str] = None
    ) -> None:
        """
        Log a decision.
        
        Args:
            decision_type: The type of decision
            context: The context of the decision
            outcome: The outcome of the decision
            explanation: An explanation of the decision
        """
        # Create decision record
        decision = {
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "context": context,
            "outcome": outcome
        }
        
        if explanation:
            decision["explanation"] = explanation
        
        # Log to file if enabled
        if self.log_dir:
            # Generate filename based on date
            date_str = datetime.now().strftime("%Y%m%d")
            filename = os.path.join(self.log_dir, f"decisions_{date_str}.jsonl")
            
            # Append decision to file
            with open(filename, "a") as f:
                f.write(json.dumps(decision) + "\n")
    
    def get_decisions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        decision_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get decisions.
        
        Args:
            start_date: Start date
            end_date: End date
            decision_types: Decision types to include
            
        Returns:
            A list of decisions
        """
        if not self.log_dir or not os.path.exists(self.log_dir):
            return []
        
        decisions = []
        
        # Get list of decision files
        decision_files = [f for f in os.listdir(self.log_dir) if f.startswith("decisions_") and f.endswith(".jsonl")]
        
        for filename in decision_files:
            file_path = os.path.join(self.log_dir, filename)
            
            with open(file_path, "r") as f:
                for line in f:
                    decision = json.loads(line)
                    
                    # Apply filters
                    decision_timestamp = datetime.fromisoformat(decision["timestamp"])
                    
                    if start_date and decision_timestamp < start_date:
                        continue
                    
                    if end_date and decision_timestamp > end_date:
                        continue
                    
                    if decision_types and decision["decision_type"] not in decision_types:
                        continue
                    
                    decisions.append(decision)
        
        # Sort by timestamp
        decisions.sort(key=lambda x: x["timestamp"])
        
        return decisions


class CostEstimator:
    """
    Estimates costs for AI operations.
    """
    
    def __init__(self):
        """
        Initialize the cost estimator.
        """
        # Provider costs per 1K tokens (as of 2025)
        self.provider_costs = {
            "openai": {
                "gpt-3.5-turbo": 0.0015,
                "gpt-4o": 0.01,
                "o1-mini": 0.015,
                "o1": 0.03
            },
            "anthropic": {
                "claude-3-5-sonnet-20240620": 0.003,
                "claude-3-5-haiku-20241022": 0.00025,
                "claude-3-opus-20240229": 0.015
            },
            "mistral": {
                "mistral-small-latest": 0.002,
                "mistral-medium-latest": 0.006,
                "mistral-large-latest": 0.008
            },
            "cohere": {
                "command": 0.005,
                "command-r": 0.003,
                "command-r-plus": 0.01
            }
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        
        Args:
            text: The text
            
        Returns:
            The estimated number of tokens
        """
        if not text:
            return 0
        
        # Simple estimation: ~4 characters per token
        return len(text) // 4 + 1
    
    def estimate_cost(
        self,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        estimated_completion_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate the cost of a request.
        
        Args:
            provider: The provider
            model: The model
            prompt: The prompt
            system_prompt: The system prompt
            estimated_completion_length: Estimated completion length in tokens
            
        Returns:
            Cost estimate information
        """
        # Estimate prompt tokens
        prompt_tokens = self.estimate_tokens(prompt)
        
        # Estimate system tokens
        system_tokens = 0
        if system_prompt:
            system_tokens = self.estimate_tokens(system_prompt)
        
        # Estimate completion tokens
        completion_tokens = 0
        if estimated_completion_length:
            completion_tokens = estimated_completion_length
        else:
            # Default to 1.5x prompt tokens
            completion_tokens = int(prompt_tokens * 1.5)
        
        # Calculate total tokens
        total_tokens = prompt_tokens + system_tokens + completion_tokens
        
        # Get cost per token
        provider_costs = self.provider_costs.get(provider.lower(), {})
        cost_per_1k_tokens = provider_costs.get(model, 0.01)  # Default to $0.01 per 1K tokens
        
        # Calculate cost
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            "prompt_tokens": prompt_tokens,
            "system_tokens": system_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_per_1k_tokens": cost_per_1k_tokens,
            "estimated_cost": estimated_cost
        }
    
    def compare_costs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        estimated_completion_length: Optional[int] = None
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compare costs across providers and models.
        
        Args:
            prompt: The prompt
            system_prompt: The system prompt
            estimated_completion_length: Estimated completion length in tokens
            
        Returns:
            Cost comparison information
        """
        comparison = {}
        
        for provider, models in self.provider_costs.items():
            comparison[provider] = {}
            
            for model, _ in models.items():
                comparison[provider][model] = self.estimate_cost(
                    provider,
                    model,
                    prompt,
                    system_prompt,
                    estimated_completion_length
                )
        
        return comparison
    
    def _calculate_cost(
        self,
        provider: str,
        model: str,
        usage: Dict[str, int]
    ) -> float:
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
        provider_costs = self.provider_costs.get(provider.lower(), {})
        cost_per_1k_tokens = provider_costs.get(model, 0.01)  # Default to $0.01 per 1K tokens
        
        # Calculate cost
        total_tokens = usage.get("total_tokens", 0)
        return (total_tokens / 1000) * cost_per_1k_tokens

