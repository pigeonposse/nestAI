"""
NestAI - A simplified interface for working with AI models.
"""

from nestai.core.wrapper import (
    NestAI,
    nest,
    async_nest,
    set_api_key,
    create_conversation,
    create_batch,
    create_pipeline
)

__version__ = "1.0.0"
__all__ = [
    "NestAI",
    "nest",
    "async_nest",
    "set_api_key",
    "create_conversation",
    "create_batch",
    "create_pipeline"
]

