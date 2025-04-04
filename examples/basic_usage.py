"""
Basic usage examples for NestAI.
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nestai import nest, async_nest, set_api_key, create_conversation, create_batch, create_pipeline
from nestai.core.wrapper import NestAI


def simple_example():
    """Simple synchronous example."""
    # Set API key
    set_api_key("openai", os.environ.get("OPENAI_API_KEY", "your-api-key"))
    
    # Make a request
    response = nest("What is artificial intelligence?")
    
    print("Simple Example Response:")
    print(response)
    print()


async def async_example():
    """Simple asynchronous example."""
    # Set API key
    set_api_key("openai", os.environ.get("OPENAI_API_KEY", "your-api-key"))
    
    # Make a request
    response = await async_nest("What are the benefits of AI?")
    
    print("Async Example Response:")
    print(response)
    print()


async def conversation_example():
    """Conversation example."""
    # Set API key
    set_api_key("openai", os.environ.get("OPENAI_API_KEY", "your-api-key"))
    
    # Create a conversation
    conversation = create_conversation()
    
    # Set system prompt
    conversation.set_system_prompt("You are a helpful assistant specializing in technology.")
    
    # Add context
    conversation.add_context("The user is interested in learning about AI technologies.")
    
    # Ask questions
    print("Conversation Example:")
    
    response1 = await conversation.ask("What is machine learning?")
    print("Q: What is machine learning?")
    print(f"A: {response1}")
    
    response2 = await conversation.ask("How is it different from deep learning?")
    print("Q: How is it different from deep learning?")
    print(f"A: {response2}")
    
    print()


async def batch_example():
    """Batch processing example."""
    # Set API key
    set_api_key("openai", os.environ.get("OPENAI_API_KEY", "your-api-key"))
    
    # Create a batch processor
    batch = create_batch()
    
    # Add requests
    batch.add("What is artificial intelligence?")
    batch.add("What is machine learning?")
    batch.add("What is deep learning?")
    
    # Process the batch
    responses = await batch.process()
    
    print("Batch Example Responses:")
    for i, response in enumerate(responses):
        print(f"Response {i+1}:")
        print(response[:100] + "..." if len(response) > 100 else response)
        print()


async def pipeline_example():
    """Pipeline example."""
    # Define pipeline steps
    def extract_keywords(text):
        # Simple keyword extraction (in a real app, use NLP)
        words = text.split()
        return [word for word in words if len(word) > 5]
    
    async def get_definitions(keywords):
        # Get definitions for keywords
        definitions = {}
        for keyword in keywords:
            # In a real app, use an API or database
            definitions[keyword] = f"Definition of {keyword}"
        return definitions
    
    def format_output(definitions):
        # Format the output
        result = "Keyword Definitions:\n\n"
        for keyword, definition in definitions.items():
            result += f"{keyword}: {definition}\n"
        return result
    
    # Create a pipeline
    pipeline = create_pipeline([
        extract_keywords,
        get_definitions,
        format_output
    ])
    
    # Run the pipeline
    result = await pipeline.run("Artificial intelligence and machine learning are transforming technology.")
    
    print("Pipeline Example Result:")
    print(result)
    print()


async def custom_configuration_example():
    """Example with custom configuration."""
    # Create a custom wrapper
    wrapper = NestAI(
        config={
            "default_provider": "openai",
            "default_model": "gpt-3.5-turbo",
            "api_keys": {
                "openai": os.environ.get("OPENAI_API_KEY", "your-api-key")
            }
        },
        monitoring={
            "log_level": "INFO",
            "metrics_dir": "metrics",
            "audit_dir": "audit"
        }
    )
    
    # Make a request
    response = await wrapper.process_request(
        "What are the ethical considerations of AI?",
        system="You are an AI ethics expert.",
        temperature=0.7,
        max_tokens=200
    )
    
    print("Custom Configuration Example Response:")
    print(response[:100] + "..." if len(response) > 100 else response)
    print()


async def main():
    """Run all examples."""
    # Simple example
    simple_example()
    
    # Async example
    await async_example()
    
    # Conversation example
    await conversation_example()
    
    # Batch example
    await batch_example()
    
    # Pipeline example
    await pipeline_example()
    
    # Custom configuration example
    await custom_configuration_example()


if __name__ == "__main__":
    asyncio.run(main())

