"""
SYSTEM_PROMPT Module

Centralized system prompt management for the Mimi chatbot.

This module provides:
- PromptLoader: Dynamically loads prompts from markdown files
- PromptRegistry: Central registry for accessing prompts
- Organized prompt structure in core/ subdirectories

Usage:
    # Method 1: Using global registry (recommended)
    from SYSTEM_PROMPT.registry import prompt_registry
    
    agent_prompt = prompt_registry.get_agent("mimi_school")
    emotion_prompt = prompt_registry.get_function("emotion_extraction")
    
    # Method 2: Using registry functions
    from SYSTEM_PROMPT.registry import get_agent_prompt, get_function_prompt
    
    prompt = get_agent_prompt("mimi_school")
    
    # Method 3: Getting registry instance
    from SYSTEM_PROMPT.registry import get_registry
    
    registry = get_registry()
    info = registry.get_info()
"""

from .loader import PromptLoader, get_loader
from .registry import PromptRegistry, get_registry, prompt_registry
from .registry import (
    get_agent_prompt,
    get_function_prompt,
    get_supporting_prompt,
    list_all_prompts
)

__version__ = "1.0.0"
__author__ = "THCSGIATHANH_KHKT"
__description__ = "Centralized system prompt management for Mimi chatbot"

__all__ = [
    # Loader
    "PromptLoader",
    "get_loader",
    
    # Registry
    "PromptRegistry",
    "get_registry",
    "prompt_registry",
    
    # Convenience functions
    "get_agent_prompt",
    "get_function_prompt",
    "get_supporting_prompt",
    "list_all_prompts",
]

# Module initialization logging
import logging

logger = logging.getLogger(__name__)
logger.debug(f"✅ SYSTEM_PROMPT module initialized (v{__version__})")
