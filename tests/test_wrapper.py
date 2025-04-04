"""
Tests for the NestAI core functionality.
"""

import unittest
import asyncio
import os
import sys
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nestai.core.wrapper import (
    NestAI, ResourceManager, EventSystem, 
    AIBatchProcessor, AIConversationManager, AIPipeline,
    nest, async_nest, set_api_key
)


class TestNestAI(unittest.TestCase):
    """Tests for the NestAI class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for cache and metrics
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a wrapper with test configuration
        self.wrapper = NestAI(
            config={
                "default_provider": "test_provider",
                "default_model": "test_model",
                "api_keys": {"test_provider": "test_key"}
            },
            monitoring={
                "log_level": "INFO",
                "metrics_dir": os.path.join(self.temp_dir, "metrics"),
                "audit_dir": os.path.join(self.temp_dir, "audit")
            }
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.wrapper.default_provider, "test_provider")
        self.assertEqual(self.wrapper.default_model, "test_model")
        self.assertEqual(self.wrapper.api_keys["test_provider"], "test_key")
    
    def test_call(self):
        """Test synchronous call."""
        with patch.object(self.wrapper, 'process_request', return_value=asyncio.Future()) as mock_process:
            mock_process.return_value.set_result("Test response")
            
            response = self.wrapper("Test prompt")
            
            mock_process.assert_called_once()
            self.assertEqual(response, "Test response")
    
    @patch('nestai.core.wrapper.NestAI._process_with_provider')
    def test_process_request(self, mock_process):
        """Test processing a request."""
        # Set up mock
        mock_process.return_value = asyncio.Future()
        mock_process.return_value.set_result(("Test response", {"total_tokens": 10}))
        
        # Run test
        response = asyncio.run(self.wrapper.process_request("Test prompt"))
        
        # Verify
        mock_process.assert_called_once()
        self.assertEqual(response, "Test response")
    
    @patch('nestai.core.wrapper.NestAI._process_with_provider')
    def test_process_request_with_cache(self, mock_process):
        """Test processing a request with cache."""
        # Set up cache
        cache_key = "test_key"
        self.wrapper.cache.memory_cache[cache_key] = {
            "response": "Cached response",
            "timestamp": 9999999999  # Far in the future
        }
        
        # Patch generate_key to return our test key
        with patch.object(self.wrapper.cache, 'generate_key', return_value=cache_key):
            # Run test
            response = asyncio.run(self.wrapper.process_request("Test prompt", cache=True))
            
            # Verify
            mock_process.assert_not_called()
            self.assertEqual(response, "Cached response")
    
    @patch('nestai.core.wrapper.NestAI._process_with_provider')
    def test_process_request_with_error(self, mock_process):
        """Test processing a request with an error."""
        # Set up mock to raise an exception
        mock_process.side_effect = Exception("Test error")
        
        # Run test
        with self.assertRaises(Exception):
            asyncio.run(self.wrapper.process_request("Test prompt"))
    
    def test_batch(self):
        """Test batch processor."""
        batch = self.wrapper.batch()
        self.assertIsInstance(batch, AIBatchProcessor)
    
    def test_conversation(self):
        """Test conversation manager."""
        conversation = self.wrapper.conversation()
        self.assertIsInstance(conversation, AIConversationManager)
    
    def test_pipeline(self):
        """Test pipeline."""
        pipeline = self.wrapper.pipeline([lambda x: x])
        self.assertIsInstance(pipeline, AIPipeline)


class TestResourceManager(unittest.TestCase):
    """Tests for the ResourceManager class."""
    
    def test_context_manager(self):
        """Test context manager."""
        manager = ResourceManager(max_concurrent=2)
        
        async def test():
            # First two should work immediately
            async with manager:
                self.assertEqual(manager.active_requests, 1)
            
            async with manager:
                self.assertEqual(manager.active_requests, 1)
            
            # Test multiple concurrent
            tasks = []
            for _ in range(5):
                tasks.append(asyncio.create_task(self._task_with_manager(manager)))
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
            self.assertEqual(manager.active_requests, 0)
        
        asyncio.run(test())
    
    async def _task_with_manager(self, manager):
        """Helper task for testing the resource manager."""
        async with manager:
            # Simulate some work
            await asyncio.sleep(0.1)


class TestEventSystem(unittest.TestCase):
    """Tests for the EventSystem class."""
    
    def test_on_emit(self):
        """Test registering and emitting events."""
        events = EventSystem()
        
        # Create a mock callback
        callback = MagicMock()
        
        # Register the callback
        events.on("test_event", callback)
        
        # Emit the event
        events.emit("test_event", {"data": "test"})
        
        # Verify callback was called
        callback.assert_called_once_with({"data": "test"})
    
    def test_multiple_callbacks(self):
        """Test multiple callbacks for an event."""
        events = EventSystem()
        
        # Create mock callbacks
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        # Register callbacks
        events.on("test_event", callback1)
        events.on("test_event", callback2)
        
        # Emit the event
        events.emit("test_event", {"data": "test"})
        
        # Verify callbacks were called
        callback1.assert_called_once_with({"data": "test"})
        callback2.assert_called_once_with({"data": "test"})
    
    def test_callback_error(self):
        """Test error in callback."""
        events = EventSystem()
        
        # Create a callback that raises an exception
        def error_callback(data):
            raise Exception("Test error")
        
        # Register the callback
        events.on("test_event", error_callback)
        
        # Emit the event (should not raise an exception)
        events.emit("test_event", {"data": "test"})


class TestAIBatchProcessor(unittest.TestCase):
    """Tests for the AIBatchProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock wrapper
        self.wrapper = MagicMock()
        self.batch = AIBatchProcessor(self.wrapper)
    
    def test_add(self):
        """Test adding requests to the batch."""
        self.batch.add("Test prompt 1")
        self.batch.add("Test prompt 2", system="Test system")
        
        self.assertEqual(len(self.batch.queue), 2)
        self.assertEqual(self.batch.queue[0]["prompt"], "Test prompt 1")
        self.assertEqual(self.batch.queue[1]["prompt"], "Test prompt 2")
        self.assertEqual(self.batch.queue[1]["options"]["system"], "Test system")
    
    def test_process(self):
        """Test processing the batch."""
        # Set up mock
        self.wrapper.process_request = MagicMock()
        self.wrapper.process_request.side_effect = [
            asyncio.Future(),
            asyncio.Future()
        ]
        self.wrapper.process_request.side_effect[0].set_result("Response 1")
        self.wrapper.process_request.side_effect[1].set_result("Response 2")
        
        # Add requests
        self.batch.add("Test prompt 1")
        self.batch.add("Test prompt 2")
        
        # Process
        responses = asyncio.run(self.batch.process())
        
        # Verify
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0], "Response 1")
        self.assertEqual(responses[1], "Response 2")
        self.wrapper.process_request.assert_called()


class TestAIConversationManager(unittest.TestCase):
    """Tests for the AIConversationManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock wrapper
        self.wrapper = MagicMock()
        self.conversation = AIConversationManager(self.wrapper)
    
    def test_add_context(self):
        """Test adding context."""
        self.conversation.add_context("Test context")
        self.assertEqual(self.conversation.context, ["Test context"])
    
    def test_set_system_prompt(self):
        """Test setting system prompt."""
        self.conversation.set_system_prompt("Test system prompt")
        self.assertEqual(self.conversation.system_prompt, "Test system prompt")
    
    def test_build_prompt(self):
        """Test building a prompt."""
        # Add context and memory
        self.conversation.add_context("Test context")
        self.conversation.memory = [
            {"prompt": "Test prompt 1", "response": "Test response 1"}
        ]
        
        # Build prompt
        prompt = self.conversation._build_prompt("Test prompt 2")
        
        # Verify
        self.assertIn("Test context", prompt)
        self.assertIn("Test prompt 1", prompt)
        self.assertIn("Test response 1", prompt)
        self.assertIn("Test prompt 2", prompt)
    
    def test_ask(self):
        """Test asking a question."""
        # Set up mock
        self.wrapper.process_request = MagicMock()
        future = asyncio.Future()
        future.set_result("Test response")
        self.wrapper.process_request.return_value = future
        
        # Ask a question
        response = asyncio.run(self.conversation.ask("Test prompt"))
        
        # Verify
        self.assertEqual(response, "Test response")
        self.wrapper.process_request.assert_called_once()
        self.assertEqual(self.conversation.memory[0]["prompt"], "Test prompt")
        self.assertEqual(self.conversation.memory[0]["response"], "Test response")
    
    def test_clear_memory(self):
        """Test clearing memory."""
        # Add some memory
        self.conversation.memory = [
            {"prompt": "Test prompt", "response": "Test response"}
        ]
        
        # Clear memory
        self.conversation.clear_memory()
        
        # Verify
        self.assertEqual(self.conversation.memory, [])


class TestAIPipeline(unittest.TestCase):
    """Tests for the AIPipeline class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock wrapper
        self.wrapper = MagicMock()
        
        # Create some test steps
        def step1(x):
            return x + " step1"
        
        async def step2(x):
            return x + " step2"
        
        def step3(x):
            return x + " step3"
        
        # Create a pipeline
        self.pipeline = AIPipeline(self.wrapper, [step1, step2, step3])
    
    def test_run(self):
        """Test running the pipeline."""
        # Run the pipeline
        result = asyncio.run(self.pipeline.run("input"))
        
        # Verify
        self.assertEqual(result, "input step1 step2 step3")


class TestAPIFunctions(unittest.TestCase):
    """Tests for the API functions."""
    
    @patch('nestai.core.wrapper.nestai')
    def test_nest(self, mock_wrapper):
        """Test the nest function."""
        # Set up mock
        mock_wrapper.__call__ = MagicMock(return_value="Test response")
        
        # Call the function
        response = nest("Test prompt")
        
        # Verify
        mock_wrapper.__call__.assert_called_once_with("Test prompt")
        self.assertEqual(response, "Test response")
    
    @patch('nestai.core.wrapper.nestai')
    def test_async_nest(self, mock_wrapper):
        """Test the async_nest function."""
        # Set up mock
        mock_wrapper.async_call = MagicMock()
        future = asyncio.Future()
        future.set_result("Test response")
        mock_wrapper.async_call.return_value = future
        
        # Call the function
        response = asyncio.run(async_nest("Test prompt"))
        
        # Verify
        mock_wrapper.async_call.assert_called_once_with("Test prompt")
        self.assertEqual(response, "Test response")
    
    @patch('nestai.core.wrapper.nestai')
    def test_set_api_key(self, mock_wrapper):
        """Test the set_api_key function."""
        # Set up mock
        mock_wrapper.api_keys = {}
        
        # Call the function
        set_api_key("test_provider", "test_key")
        
        # Verify
        self.assertEqual(mock_wrapper.api_keys["test_provider"], "test_key")
        self.assertEqual(os.environ["TEST_PROVIDER_API_KEY"], "test_key")


if __name__ == "__main__":
    unittest.main()

