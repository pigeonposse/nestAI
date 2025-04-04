"""
Semantic caching for NestAI.
"""

import os
import json
import hashlib
import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from datetime import datetime


class SemanticCache:
    """
    Cache with semantic similarity matching.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl: int = 3600,
        similarity_threshold: float = 0.9,
        max_memory_entries: int = 1000,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the semantic cache.
        
        Args:
            cache_dir: Directory for storing cache
            ttl: Time-to-live in seconds
            similarity_threshold: Threshold for semantic similarity
            max_memory_entries: Maximum number of entries in memory
            embedding_model: Model for generating embeddings
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.max_memory_entries = max_memory_entries
        self.embedding_model = embedding_model
        
        # In-memory cache
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Create cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def generate_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        provider: str,
        preset: Optional[str],
        options: Dict[str, Any]
    ) -> str:
        """
        Generate a cache key.
        
        Args:
            prompt: The prompt
            system_prompt: The system prompt
            model: The model
            provider: The provider
            preset: The preset
            options: Additional options
            
        Returns:
            The cache key
        """
        # Create a string representation of the inputs
        key_parts = [
            prompt,
            system_prompt or "",
            model,
            provider,
            preset or ""
        ]
        
        # Add relevant options
        relevant_options = {}
        for key in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
            if key in options:
                relevant_options[key] = options[key]
        
        if relevant_options:
            key_parts.append(json.dumps(relevant_options, sort_keys=True))
        
        # Join and hash
        key_string = "||".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """
        Get a response from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached response, or None if not found
        """
        # Check memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            # Check if entry is expired
            if time.time() - entry["timestamp"] > self.ttl:
                del self.memory_cache[key]
                return None
            
            return entry["response"]
        
        # Check disk cache if enabled
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        entry = json.load(f)
                    
                    # Check if entry is expired
                    if time.time() - entry["timestamp"] > self.ttl:
                        os.remove(cache_file)
                        return None
                    
                    # Add to memory cache
                    self.memory_cache[key] = entry
                    self._prune_memory_cache()
                    
                    return entry["response"]
                except Exception:
                    # If there's an error reading the cache, ignore it
                    pass
        
        return None
    
    def set(
        self,
        key: str,
        prompt: str,
        system_prompt: Optional[str],
        response: str
    ) -> None:
        """
        Set a response in the cache.
        
        Args:
            key: The cache key
            prompt: The prompt
            system_prompt: The system prompt
            response: The response
        """
        # Create cache entry
        entry = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response,
            "timestamp": time.time()
        }
        
        # Store in memory cache
        self.memory_cache[key] = entry
        self._prune_memory_cache()
        
        # Store in disk cache if enabled
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            
            try:
                with open(cache_file, "w") as f:
                    json.dump(entry, f)
            except Exception:
                # If there's an error writing to the cache, ignore it
                pass
    
    def clear(self) -> None:
        """
        Clear the cache.
        """
        # Clear memory cache
        self.memory_cache = {}
        
        # Clear disk cache if enabled
        if self.cache_dir and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except Exception:
                        # If there's an error removing a file, ignore it
                        pass
    
    def _prune_memory_cache(self) -> None:
        """
        Prune the memory cache if it exceeds the maximum size.
        """
        if len(self.memory_cache) <= self.max_memory_entries:
            return
        
        # Remove oldest entries
        entries = sorted(self.memory_cache.items(), key=lambda x: x[1]["timestamp"])
        entries_to_remove = entries[:len(entries) - self.max_memory_entries]
        
        for key, _ in entries_to_remove:
            del self.memory_cache[key]
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding for text.
        
        Args:
            text: The text
            
        Returns:
            The embedding
        """
        # This is a placeholder - in a real implementation, this would use an embedding model
        # For now, we'll use a simple hash-based approach
        
        # Convert text to a hash
        hash_value = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a numpy array of floats
        embedding = np.array([float(int(c, 16)) for c in hash_value]) / 15.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            The similarity score
        """
        return np.dot(embedding1, embedding2)
    
    async def _find_semantic_match(self, prompt: str) -> Tuple[Optional[str], bool]:
        """
        Find a semantic match for a prompt.
        
        Args:
            prompt: The prompt
            
        Returns:
            A tuple of (cached_response, is_semantic_match) or (None, False) if not found
        """
        # Get embedding for the prompt
        prompt_embedding = await self._get_embedding(prompt)
        
        best_match = None
        best_similarity = 0.0
        
        # Check memory cache
        for entry in self.memory_cache.values():
            # Skip expired entries
            if time.time() - entry["timestamp"] > self.ttl:
                continue
            
            # Get embedding for the cached prompt
            cached_prompt = entry["prompt"]
            cached_embedding = await self._get_embedding(cached_prompt)
            
            # Calculate similarity
            similarity = self._calculate_similarity(prompt_embedding, cached_embedding)
            
            # Check if it's a good match
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_match = entry["response"]
                best_similarity = similarity
        
        if best_match:
            return best_match, True
        
        return None, False

