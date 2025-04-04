"""
Metrics tracking for NestAI.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import threading


class MetricsTracker:
    """
    Tracks metrics for AI operations.
    """
    
    def __init__(
        self,
        metrics_dir: Optional[str] = None,
        auto_save: bool = True,
        save_interval: int = 300  # 5 minutes
    ):
        """
        Initialize the metrics tracker.
        
        Args:
            metrics_dir: Directory for storing metrics
            auto_save: Whether to automatically save metrics
            save_interval: Interval for auto-saving in seconds
        """
        self.metrics_dir = metrics_dir
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # Initialize metrics
        self.reset_metrics()
        
        # Set up auto-save if enabled
        if self.auto_save and self.metrics_dir:
            self._setup_auto_save()
    
    def reset_metrics(self) -> None:
        """
        Reset all metrics.
        """
        self.metrics = {
            "requests": 0,
            "errors": {
                "count": 0,
                "types": {}
            },
            "providers": {},
            "models": {},
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            },
            "latency": {
                "total": 0.0,
                "min": float('inf'),
                "max": 0.0,
                "average": 0.0
            },
            "costs": {
                "total": 0.0,
                "by_provider": {},
                "by_model": {}
            },
            "cache": {
                "hits": 0,
                "semantic_hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "semantic_hit_rate": 0.0
            },
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat()
        }
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
    
    def _setup_auto_save(self) -> None:
        """
        Set up automatic saving of metrics.
        """
        def auto_save_worker():
            while True:
                time.sleep(self.save_interval)
                self.save_metrics()
        
        # Start auto-save thread
        save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        save_thread.start()
    
    def track_request(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        cost: float,
        cached: bool = False,
        semantic_cache: bool = False
    ) -> None:
        """
        Track an AI request.
        
        Args:
            provider: The provider
            model: The model
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            latency: The latency in seconds
            cost: The cost in USD
            cached: Whether the response was cached
            semantic_cache: Whether the response was from semantic cache
        """
        with self.lock:
            # Update request count
            self.metrics["requests"] += 1
            
            # Update provider and model counts
            self.metrics["providers"][provider] = self.metrics["providers"].get(provider, 0) + 1
            self.metrics["models"][model] = self.metrics["models"].get(model, 0) + 1
            
            # Update token counts
            self.metrics["tokens"]["prompt"] += prompt_tokens
            self.metrics["tokens"]["completion"] += completion_tokens
            self.metrics["tokens"]["total"] += prompt_tokens + completion_tokens
            
            # Update cache metrics
            if cached:
                if semantic_cache:
                    self.metrics["cache"]["semantic_hits"] += 1
                else:
                    self.metrics["cache"]["hits"] += 1
            else:
                self.metrics["cache"]["misses"] += 1
            
            total_cache_requests = self.metrics["cache"]["hits"] + self.metrics["cache"]["semantic_hits"] + self.metrics["cache"]["misses"]
            if total_cache_requests > 0:
                self.metrics["cache"]["hit_rate"] = (self.metrics["cache"]["hits"] + self.metrics["cache"]["semantic_hits"]) / total_cache_requests
                self.metrics["cache"]["semantic_hit_rate"] = self.metrics["cache"]["semantic_hits"] / total_cache_requests
            
            # Update latency metrics (only for non-cached requests)
            if not cached:
                self.metrics["latency"]["total"] += latency
                self.metrics["latency"]["min"] = min(self.metrics["latency"]["min"], latency) if self.metrics["latency"]["min"] != float('inf') else latency
                self.metrics["latency"]["max"] = max(self.metrics["latency"]["max"], latency)
                non_cached_requests = self.metrics["requests"] - self.metrics["cache"]["hits"] - self.metrics["cache"]["semantic_hits"]
                if non_cached_requests > 0:
                    self.metrics["latency"]["average"] = self.metrics["latency"]["total"] / non_cached_requests
            
            # Update cost metrics
            self.metrics["costs"]["total"] += cost
            self.metrics["costs"]["by_provider"][provider] = self.metrics["costs"]["by_provider"].get(provider, 0.0) + cost
            self.metrics["costs"]["by_model"][model] = self.metrics["costs"]["by_model"].get(model, 0.0) + cost
            
            # Update timestamp
            self.metrics["last_update"] = datetime.now().isoformat()
    
    def track_error(
        self,
        error_type: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:
        """
        Track an error.
        
        Args:
            error_type: The type of error
            provider: The provider
            model: The model
        """
        with self.lock:
            # Update error count
            self.metrics["errors"]["count"] += 1
            self.metrics["errors"]["types"][error_type] = self.metrics["errors"]["types"].get(error_type, 0) + 1
            
            # Update timestamp
            self.metrics["last_update"] = datetime.now().isoformat()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics.
        
        Returns:
            The current metrics
        """
        with self.lock:
            return self.metrics.copy()
    
    def save_metrics(self) -> None:
        """
        Save metrics to a file.
        """
        if not self.metrics_dir:
            return
        
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        with self.lock:
            # Generate filename based on current time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
            
            # Save metrics to file
            with open(filename, "w") as f:
                json.dump(self.metrics, f, indent=2)
    
    def load_metrics(self, filename: str) -> None:
        """
        Load metrics from a file.
        
        Args:
            filename: The filename
        """
        if not os.path.exists(filename):
            return
        
        with open(filename, "r") as f:
            loaded_metrics = json.load(f)
        
        with self.lock:
            self.metrics = loaded_metrics
    
    def generate_report(self) -> str:
        """
        Generate a human-readable report of the metrics.
        
        Returns:
            A report string
        """
        with self.lock:
            metrics = self.metrics.copy()
        
        # Calculate time period
        start_time = datetime.fromisoformat(metrics["start_time"])
        end_time = datetime.fromisoformat(metrics["last_update"])
        duration = end_time - start_time
        
        # Format the report
        report = []
        report.append("=" * 50)
        report.append("NestAI Metrics Report")
        report.append("=" * 50)
        report.append(f"Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} ({duration})")
        report.append("")
        
        report.append("Request Statistics:")
        report.append(f"  Total Requests: {metrics['requests']}")
        report.append(f"  Total Errors: {metrics['errors']['count']}")
        if metrics['errors']['count'] > 0:
            report.append("  Error Types:")
            for error_type, count in metrics['errors']['types'].items():
                report.append(f"    {error_type}: {count}")
        report.append("")
        
        report.append("Provider Usage:")
        for provider, count in sorted(metrics['providers'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {provider}: {count} requests ({count / metrics['requests'] * 100:.1f}%)")
        report.append("")
        
        report.append("Model Usage:")
        for model, count in sorted(metrics['models'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {model}: {count} requests ({count / metrics['requests'] * 100:.1f}%)")
        report.append("")
        
        report.append("Token Usage:")
        report.append(f"  Prompt Tokens: {metrics['tokens']['prompt']}")
        report.append(f"  Completion Tokens: {metrics['tokens']['completion']}")
        report.append(f"  Total Tokens: {metrics['tokens']['total']}")
        report.append("")
        
        report.append("Latency:")
        report.append(f"  Average: {metrics['latency']['average']:.4f}s")
        report.append(f"  Min: {metrics['latency']['min'] if metrics['latency']['min'] != float('inf') else 0:.4f}s")
        report.append(f"  Max: {metrics['latency']['max']:.4f}s")
        report.append("")
        
        report.append("Cost:")
        report.append(f"  Total: ${metrics['costs']['total']:.4f}")
        report.append("  By Provider:")
        for provider, cost in sorted(metrics['costs']['by_provider'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"    {provider}: ${cost:.4f} ({cost / metrics['costs']['total'] * 100:.1f}%)")
        report.append("")
        
        report.append("Cache Performance:")
        total_cache_requests = metrics['cache']['hits'] + metrics['cache']['semantic_hits'] + metrics['cache']['misses']
        if total_cache_requests > 0:
            report.append(f"  Exact Cache Hits: {metrics['cache']['hits']} ({metrics['cache']['hits'] / total_cache_requests * 100:.1f}%)")
            report.append(f"  Semantic Cache Hits: {metrics['cache']['semantic_hits']} ({metrics['cache']['semantic_hits'] / total_cache_requests * 100:.1f}%)")
            report.append(f"  Cache Misses: {metrics['cache']['misses']} ({metrics['cache']['misses'] / total_cache_requests * 100:.1f}%)")
            report.append(f"  Overall Hit Rate: {metrics['cache']['hit_rate'] * 100:.1f}%")
        else:
            report.append("  No cache usage recorded")
        report.append("")
        
        report.append("=" * 50)
        
        return "\n".join(report)

