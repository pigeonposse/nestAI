"""
Integration tests for the NestAI library.
"""

import unittest
import asyncio
import os
import sys
import time
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nestai import nest, async_nest, set_api_key
from nestai.core.wrapper import (
    NestAI, AIBatchProcessor, AIConversationManager, AIPipeline
)


class TestIntegration(unittest.TestCase):
    """Integration tests for the NestAI library."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a wrapper with mocked provider
        self.wrapper = NestAI()
        
        # Patch the _process_with_provider method to return a mock response
        patcher = patch.object(self.wrapper, '_process_with_provider')
        self.mock_process = patcher.start()
        self.addCleanup(patcher.stop)
        
        # Set up the mock to return a response
        self.mock_process.side_effect = self._mock_process_with_provider
    
    async def _mock_process_with_provider(self, prompt, system_prompt, provider, model, stream, **kwargs):
        """Mock implementation of _process_with_provider."""
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Generate a response based on the prompt
        if "error" in prompt.lower():
            raise Exception("Test error")
        
        response = f"Response to: {prompt}"
        if system_prompt:
            response = f"[System: {system_prompt}] {response}"
        
        # Generate usage statistics
        usage = {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(response) // 4,
            "total_tokens": (len(prompt) + len(response)) // 4
        }
        
        return response, usage
    
    def test_simple_request(self):
        """Test a simple request."""
        # Make a request
        response = asyncio.run(self.wrapper.process_request("Hello, world!"))
        
        # Verify
        self.assertEqual(response, "Response to: Hello, world!")
        self.mock_process.assert_called_once()
    
    def test_request_with_system_prompt(self):
        """Test a request with a system prompt."""
        # Make a request
        response = asyncio.run(self.wrapper.process_request(
            "Hello, world!",
            system="You are a helpful assistant."
        ))
        
        # Verify
        self.assertEqual(response, "[System: You are a helpful assistant.] Response to: Hello, world!")
        self.mock_process.assert_called_once()
    
    def test_request_with_error(self):
        """Test a request that causes an error."""
        # Make a request that will cause an error
        with self.assertRaises(Exception):
            asyncio.run(self.wrapper.process_request("This will cause an error"))
    
    def test_caching(self):
        """Test caching."""
        # Make a request
        response1 = asyncio.run(self.wrapper.process_request("Hello, world!"))
        
        # Reset the mock
        self.mock_process.reset_mock()
        
        # Make the same request again
        response2 = asyncio.run(self.wrapper.process_request("Hello, world!"))
        
        # Verify
        self.assertEqual(response1, response2)
        self.mock_process.assert_not_called()  # Should use cache
    
    def test_batch_processing(self):
        """Test batch processing."""
        # Create a batch processor
        batch = self.wrapper.batch()
        
        # Add requests
        batch.add("Hello, world!")
        batch.add("How are you?")
        batch.add("What is AI?")
        
        # Process the batch
        responses = asyncio.run(batch.process())
        
        # Verify
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0], "Response to: Hello, world!")
        self.assertEqual(responses[1], "Response to: How are you?")
        self.assertEqual(responses[2], "Response to: What is AI?")
        self.assertEqual(self.mock_process.call_count, 3)
    
    def test_conversation(self):
        """Test conversation management."""
        # Create a conversation manager
        conversation = self.wrapper.conversation()
        
        # Add context
        conversation.add_context("The user speaks Spanish.")
        
        # Set system prompt
        conversation.set_system_prompt("You are a helpful assistant.")
        
        # Ask questions
        response1 = asyncio.run(conversation.ask("Hello!"))
        response2 = asyncio.run(conversation.ask("How are you?"))
        
        # Verify
        self.assertIn("System: You are a helpful assistant.", response1)
        self.assertIn("Response to:", response1)
        self.assertIn("System: You are a helpful assistant.", response2)
        self.assertIn("Response to:", response2)
        self.assertEqual(len(conversation.memory), 2)
    
    def test_pipeline(self):
        """Test pipeline processing."""
        # Create some pipeline steps
        def step1(text):
            return text.upper()
        
        async def step2(text):
            return f"{text}!"
        
        # Create a pipeline
        pipeline = self.wrapper.pipeline([step1, step2])
        
        # Run the pipeline
        result = asyncio.run(pipeline.run("hello"))
        
        # Verify
        self.assertEqual(result, "HELLO!")


class TestRealWorldScenarios(unittest.TestCase):
    """Tests for real-world scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a wrapper with mocked provider
        self.wrapper = NestAI()
        
        # Patch the _process_with_provider method to return a mock response
        patcher = patch.object(self.wrapper, '_process_with_provider')
        self.mock_process = patcher.start()
        self.addCleanup(patcher.stop)
        
        # Set up the mock to return a response
        self.mock_process.side_effect = self._mock_process_with_provider
    
    async def _mock_process_with_provider(self, prompt, system_prompt, provider, model, stream, **kwargs):
        """Mock implementation of _process_with_provider."""
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Generate a response based on the prompt
        if "error" in prompt.lower():
            raise Exception("Test error")
        
        response = f"Response to: {prompt}"
        if system_prompt:
            response = f"[System: {system_prompt}] {response}"
        
        # Generate usage statistics
        usage = {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(response) // 4,
            "total_tokens": (len(prompt) + len(response)) // 4
        }
        
        return response, usage
    
    def test_customer_support_scenario(self):
        """Test a customer support scenario."""
        # Create a conversation manager
        conversation = self.wrapper.conversation()
        
        # Set up the conversation
        conversation.set_system_prompt(
            "You are a customer support agent for Acme Inc. "
            "Be helpful, friendly, and professional."
        )
        
        # Simulate a conversation
        responses = []
        
        # Customer inquiry
        responses.append(asyncio.run(conversation.ask(
            "Hi, I'm having trouble with my Acme Widget. It won't turn on."
        )))
        
        # Follow-up questions
        responses.append(asyncio.run(conversation.ask(
            "Yes, I've tried charging it overnight but it still doesn't work."
        )))
        
        responses.append(asyncio.run(conversation.ask(
            "The model number is AW-2023-X."
        )))
        
        # Verify
        for response in responses:
            self.assertIn("System: You are a customer support agent", response)
            self.assertIn("Response to:", response)
        
        self.assertEqual(len(conversation.memory), 3)
    
    def test_content_generation_scenario(self):
        """Test a content generation scenario."""
        # Create a batch processor
        batch = self.wrapper.batch()
        
        # Add content generation requests
        batch.add(
            "Write a short blog post about AI ethics.",
            system="You are a professional content writer specializing in technology topics."
        )
        
        batch.add(
            "Create a product description for a smart water bottle.",
            system="You are a marketing copywriter with expertise in consumer products."
        )
        
        batch.add(
            "Draft a social media post announcing a new feature.",
            system="You are a social media manager for a tech company."
        )
        
        # Process the batch
        responses = asyncio.run(batch.process())
        
        # Verify
        self.assertEqual(len(responses), 3)
        for response in responses:
            self.assertIn("System:", response)
            self.assertIn("Response to:", response)
    
    def test_data_processing_pipeline(self):
        """Test a data processing pipeline."""
        # Create pipeline steps
        def extract_keywords(text):
            # Simulate keyword extraction
            return f"Keywords from: {text}"
        
        async def categorize(text):
            # Simulate categorization
            return f"Category for: {text}"
        
        def sentiment_analysis(text):
            # Simulate sentiment analysis
            return f"Sentiment of: {text}"
        
        # Create a pipeline
        pipeline = self.wrapper.pipeline([
            extract_keywords,
            categorize,
            sentiment_analysis
        ])
        
        # Run the pipeline
        result = asyncio.run(pipeline.run("This is a sample text for processing."))
        
        # Verify
        self.assertIn("Sentiment of: Category for: Keywords from:", result)


if __name__ == "__main__":
    unittest.main()

