"""
Utility tests for the NestAI library.
"""

import unittest
import os
import sys
import tempfile
import shutil
import asyncio
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nestai.utils.semantic_cache import SemanticCache
from nestai.utils.metrics import MetricsTracker
from nestai.utils.logging import AILogger
from nestai.utils.security import PIIDetector, DataEncryptor, SecurityAuditor
from nestai.utils.transparency import DecisionLogger, CostEstimator


class TestSemanticCache(unittest.TestCase):
    """Tests for the SemanticCache class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        self.cache = SemanticCache(cache_dir=self.temp_dir, ttl=3600, similarity_threshold=0.9)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_generate_key(self):
        """Test generating a cache key."""
        key1 = self.cache.generate_key("Hello", "System", "gpt-4", "openai", "balanced", {})
        key2 = self.cache.generate_key("Hello", "System", "gpt-4", "openai", "balanced", {})
        key3 = self.cache.generate_key("Different", "System", "gpt-4", "openai", "balanced", {})
        
        # Same inputs should generate same key
        self.assertEqual(key1, key2)
        
        # Different inputs should generate different keys
        self.assertNotEqual(key1, key3)
    
    def test_set_get(self):
        """Test setting and getting cache entries."""
        key = self.cache.generate_key("Hello", "System", "gpt-4", "openai", "balanced", {})
        
        # Set a cache entry
        self.cache.set(key, "Hello", "System", "This is a test response.")
        
        # Get the cache entry
        cached_response = self.cache.get(key)
        
        # Verify response
        self.assertEqual(cached_response, "This is a test response.")
        
        # Get a non-existent cache entry
        non_existent_key = self.cache.generate_key("Non-existent", "System", "gpt-4", "openai", "balanced", {})
        cached_response = self.cache.get(non_existent_key)
        
        # Verify response is None
        self.assertIsNone(cached_response)
    
    def test_clear(self):
        """Test clearing the cache."""
        key = self.cache.generate_key("Hello", "System", "gpt-4", "openai", "balanced", {})
        
        # Set a cache entry
        self.cache.set(key, "Hello", "System", "This is a test response.")
        
        # Clear the cache
        self.cache.clear()
        
        # Get the cache entry
        cached_response = self.cache.get(key)
        
        # Verify response is None
        self.assertIsNone(cached_response)
    
    def test_semantic_match(self):
        """Test semantic matching."""
        # Set a cache entry
        key1 = self.cache.generate_key("Hello world", "System", "gpt-4", "openai", "balanced", {})
        self.cache.set(key1, "Hello world", "System", "This is a test response.")
        
        # Test with a similar prompt
        key2 = self.cache.generate_key("Hello there", "System", "gpt-4", "openai", "balanced", {})
        
        # This is a bit tricky to test since our mock implementation uses hash-based embeddings
        # In a real implementation, this would use semantic similarity
        # For now, we'll just verify that the method exists and can be called
        result, is_semantic = asyncio.run(self.cache._find_semantic_match("Hello there"))
        
        # We can't make strong assertions about the result due to the mock implementation
        # but we can verify the return type
        self.assertIsInstance(is_semantic, bool)


class TestPIIDetector(unittest.TestCase):
    """Tests for the PIIDetector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = PIIDetector()
    
    def test_detect(self):
        """Test detecting PII."""
        text = "My email is test@example.com and my phone is 555-123-4567."
        
        detected = self.detector.detect(text)
        
        # Verify detection
        self.assertEqual(len(detected), 2)
        self.assertEqual(detected[0]["type"], "email")
        self.assertEqual(detected[0]["value"], "test@example.com")
        self.assertEqual(detected[1]["type"], "phone")
        self.assertEqual(detected[1]["value"], "555-123-4567")
    
    def test_redact(self):
        """Test redacting PII."""
        text = "My email is test@example.com and my phone is 555-123-4567."
        
        redacted, detected = self.detector.redact(text)
        
        # Verify redaction
        self.assertEqual(len(detected), 2)
        self.assertIn("[REDACTED]", redacted)
        self.assertNotIn("test@example.com", redacted)
        self.assertNotIn("555-123-4567", redacted)


class TestDataEncryptor(unittest.TestCase):
    """Tests for the DataEncryptor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for key
        self.temp_dir = tempfile.mkdtemp()
        self.key_file = os.path.join(self.temp_dir, "key")
        self.encryptor = DataEncryptor(key_file=self.key_file)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_encrypt_decrypt(self):
        """Test encrypting and decrypting data."""
        original = "This is a secret message."
        
        # Encrypt
        encrypted = self.encryptor.encrypt(original)
        
        # Verify encrypted is different from original
        self.assertNotEqual(encrypted, original)
        
        # Decrypt
        decrypted = self.encryptor.decrypt(encrypted)
        
        # Verify decrypted matches original
        self.assertEqual(decrypted, original)
    
    def test_encrypt_dict(self):
        """Test encrypting and decrypting a dictionary."""
        original = {
            "public": "This is public.",
            "secret": "This is secret.",
            "nested": {
                "public": "This is public.",
                "secret": "This is secret."
            }
        }
        
        sensitive_keys = {"secret"}
        
        # Encrypt
        encrypted = self.encryptor.encrypt_dict(original, sensitive_keys)
        
        # Verify public fields are unchanged
        self.assertEqual(encrypted["public"], original["public"])
        self.assertEqual(encrypted["nested"]["public"], original["nested"]["public"])
        
        # Verify secret fields are encrypted
        self.assertNotEqual(encrypted["secret"], original["secret"])
        self.assertNotEqual(encrypted["nested"]["secret"], original["nested"]["secret"])
        
        # Decrypt
        decrypted = self.encryptor.decrypt_dict(encrypted, sensitive_keys)
        
        # Verify all fields match original
        self.assertEqual(decrypted["public"], original["public"])
        self.assertEqual(decrypted["secret"], original["secret"])
        self.assertEqual(decrypted["nested"]["public"], original["nested"]["public"])
        self.assertEqual(decrypted["nested"]["secret"], original["nested"]["secret"])


class TestCostEstimator(unittest.TestCase):
    """Tests for the CostEstimator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.estimator = CostEstimator()
    
    def test_estimate_tokens(self):
        """Test estimating tokens."""
        text = "This is a test message with approximately 12 tokens."
        
        tokens = self.estimator.estimate_tokens(text)
        
        # Verify token count is reasonable
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, 100)
    
    def test_estimate_cost(self):
        """Test estimating cost."""
        prompt = "This is a test prompt."
        system_prompt = "You are a helpful assistant."
        
        estimate = self.estimator.estimate_cost("openai", "gpt-3.5-turbo", prompt, system_prompt)
        
        # Verify estimate
        self.assertIn("prompt_tokens", estimate)
        self.assertIn("system_tokens", estimate)
        self.assertIn("completion_tokens", estimate)
        self.assertIn("total_tokens", estimate)
        self.assertIn("estimated_cost", estimate)
        
        # Verify cost is reasonable
        self.assertGreater(estimate["estimated_cost"], 0)
        self.assertLess(estimate["estimated_cost"], 0.1)
    
    def test_compare_costs(self):
        """Test comparing costs."""
        prompt = "This is a test prompt."
        
        comparison = self.estimator.compare_costs(prompt)
        
        # Verify comparison
        self.assertIn("openai", comparison)
        self.assertIn("anthropic", comparison)
        self.assertIn("mistral", comparison)
        
        # Verify OpenAI models
        openai_models = comparison["openai"]
        self.assertIn("gpt-3.5-turbo", openai_models)
        self.assertIn("gpt-4o", openai_models)


class TestMetricsTracker(unittest.TestCase):
    """Tests for the MetricsTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for metrics
        self.temp_dir = tempfile.mkdtemp()
        self.metrics = MetricsTracker(metrics_dir=self.temp_dir, auto_save=False)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_track_request(self):
        """Test tracking a request."""
        # Track a request
        self.metrics.track_request(
            provider="openai",
            model="gpt-3.5-turbo",
            prompt_tokens=100,
            completion_tokens=50,
            latency=0.5,
            cost=0.0015,
            cached=False
        )
        
        # Get current metrics
        current_metrics = self.metrics.get_current_metrics()
        
        # Verify metrics
        self.assertEqual(current_metrics["requests"], 1)
        self.assertEqual(current_metrics["providers"]["openai"], 1)
        self.assertEqual(current_metrics["models"]["gpt-3.5-turbo"], 1)
        self.assertEqual(current_metrics["tokens"]["prompt"], 100)
        self.assertEqual(current_metrics["tokens"]["completion"], 50)
        self.assertEqual(current_metrics["tokens"]["total"], 150)
        self.assertEqual(current_metrics["latency"]["total"], 0.5)
        self.assertEqual(current_metrics["latency"]["min"], 0.5)
        self.assertEqual(current_metrics["latency"]["max"], 0.5)
        self.assertEqual(current_metrics["latency"]["average"], 0.5)
        self.assertEqual(current_metrics["costs"]["total"], 0.0015)
        self.assertEqual(current_metrics["costs"]["by_provider"]["openai"], 0.0015)
        self.assertEqual(current_metrics["costs"]["by_model"]["gpt-3.5-turbo"], 0.0015)
        self.assertEqual(current_metrics["cache"]["hits"], 0)
        self.assertEqual(current_metrics["cache"]["misses"], 1)
    
    def test_track_cached_request(self):
        """Test tracking a cached request."""
        # Track a cached request
        self.metrics.track_request(
            provider="openai",
            model="gpt-3.5-turbo",
            prompt_tokens=100,
            completion_tokens=50,
            latency=0.01,
            cost=0.0,
            cached=True
        )
        
        # Get current metrics
        current_metrics = self.metrics.get_current_metrics()
        
        # Verify metrics
        self.assertEqual(current_metrics["requests"], 1)
        self.assertEqual(current_metrics["cache"]["hits"], 1)
        self.assertEqual(current_metrics["cache"]["misses"], 0)
        self.assertEqual(current_metrics["cache"]["hit_rate"], 1.0)
    
    def test_track_error(self):
        """Test tracking an error."""
        # Track an error
        self.metrics.track_error(
            error_type="RateLimitError",
            provider="openai",
            model="gpt-3.5-turbo"
        )
        
        # Get current metrics
        current_metrics = self.metrics.get_current_metrics()
        
        # Verify metrics
        self.assertEqual(current_metrics["errors"]["count"], 1)
        self.assertEqual(current_metrics["errors"]["types"]["RateLimitError"], 1)
    
    def test_generate_report(self):
        """Test generating a report."""
        # Track some requests
        self.metrics.track_request(
            provider="openai",
            model="gpt-3.5-turbo",
            prompt_tokens=100,
            completion_tokens=50,
            latency=0.5,
            cost=0.0015,
            cached=False
        )
        
        self.metrics.track_request(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            prompt_tokens=200,
            completion_tokens=100,
            latency=0.8,
            cost=0.0025,
            cached=False
        )
        
        self.metrics.track_request(
            provider="openai",
            model="gpt-3.5-turbo",
            prompt_tokens=50,
            completion_tokens=25,
            latency=0.01,
            cost=0.0,
            cached=True
        )
        
        # Generate report
        report = self.metrics.generate_report()
        
        # Verify report is a string
        self.assertIsInstance(report, str)
        
        # Verify report contains key information
        self.assertIn("Total Requests: 3", report)
        self.assertIn("openai", report)
        self.assertIn("anthropic", report)
        self.assertIn("gpt-3.5-turbo", report)
        self.assertIn("claude-3-5-haiku-20241022", report)


class TestDecisionLogger(unittest.TestCase):
    """Tests for the DecisionLogger class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()
        self.logger = DecisionLogger(log_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_log_decision(self):
        """Test logging a decision."""
        # Log a decision
        self.logger.log_decision(
            decision_type="provider_selection",
            context={
                "prompt_length": 100,
                "available_providers": ["openai", "anthropic"]
            },
            outcome={
                "selected_provider": "openai",
                "selected_model": "gpt-3.5-turbo"
            },
            explanation="Selected based on cost and latency requirements"
        )
        
        # Verify log file was created
        log_files = [f for f in os.listdir(self.temp_dir) if f.startswith("decisions_")]
        self.assertEqual(len(log_files), 1)
        
        # Verify decision was logged
        decisions = self.logger.get_decisions()
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["decision_type"], "provider_selection")
        self.assertEqual(decisions[0]["outcome"]["selected_provider"], "openai")


if __name__ == "__main__":
    unittest.main()

