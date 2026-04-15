"""
SYSTEM_PROMPT Loader

Dynamically loads prompts from markdown files in the SYSTEM_PROMPT module.
Handles caching and formatting for efficient prompt retrieval.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PromptLoader:
    """Load prompts from markdown files in SYSTEM_PROMPT/core/"""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize loader with base path to SYSTEM_PROMPT module.
        
        Args:
            base_path: Path to SYSTEM_PROMPT directory. If None, auto-detects.
        """
        if base_path is None:
            # Auto-detect: find SYSTEM_PROMPT directory relative to this file
            self.base_path = Path(__file__).parent
        else:
            self.base_path = Path(base_path)
            
        self.core_path = self.base_path / "core"
        self._cache: Dict[str, str] = {}  # Cache loaded prompts
        
        logger.info(f"✅ PromptLoader initialized at: {self.base_path}")
        
    def load_agent(self, agent_name: str) -> str:
        """
        Load agent prompt from core/agents/{agent_name}.md
        
        Args:
            agent_name: Name of agent (e.g., "mimi_school")
            
        Returns:
            Prompt content as string
            
        Raises:
            FileNotFoundError: If prompt file not found
        """
        prompt_path = self.core_path / "agents" / f"{agent_name}.md"
        return self._load_file(prompt_path, f"agent:{agent_name}")
    
    def load_function(self, function_name: str) -> str:
        """
        Load function prompt from core/functions/{function_name}.md
        
        Args:
            function_name: Name of function (e.g., "emotion_extraction")
            
        Returns:
            Prompt content as string
            
        Raises:
            FileNotFoundError: If prompt file not found
        """
        prompt_path = self.core_path / "functions" / f"{function_name}.md"
        return self._load_file(prompt_path, f"function:{function_name}")
    
    def load_supporting(self, prompt_name: str) -> str:
        """
        Load supporting prompt from core/supporting/{prompt_name}.md
        
        Args:
            prompt_name: Name of supporting prompt
            
        Returns:
            Prompt content as string
            
        Raises:
            FileNotFoundError: If prompt file not found
        """
        prompt_path = self.core_path / "supporting" / f"{prompt_name}.md"
        return self._load_file(prompt_path, f"supporting:{prompt_name}")
    
    def _load_file(self, file_path: Path, cache_key: str) -> str:
        """
        Load file content with caching.
        
        Args:
            file_path: Full path to prompt file
            cache_key: Key for caching (category:name)
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"📦 Cache hit for {cache_key}")
            return self._cache[cache_key]
        
        # Load from file
        if not file_path.exists():
            raise FileNotFoundError(
                f"❌ Prompt file not found: {file_path}\n"
                f"Expected location: {file_path.absolute()}"
            )
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Cache it
            self._cache[cache_key] = content
            logger.info(f"✅ Loaded prompt: {cache_key}")
            return content
            
        except Exception as e:
            logger.error(f"❌ Error loading prompt {cache_key}: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()
        logger.info("✅ Prompt cache cleared")
    
    def list_prompts(self) -> Dict[str, list]:
        """
        List all available prompts by category.
        
        Returns:
            Dictionary with categories as keys and prompt names as values
        """
        prompts = {
            "agents": [],
            "functions": [],
            "supporting": []
        }
        
        if (self.core_path / "agents").exists():
            prompts["agents"] = [
                f.stem for f in (self.core_path / "agents").glob("*.md")
            ]
        
        if (self.core_path / "functions").exists():
            prompts["functions"] = [
                f.stem for f in (self.core_path / "functions").glob("*.md")
            ]
        
        if (self.core_path / "supporting").exists():
            prompts["supporting"] = [
                f.stem for f in (self.core_path / "supporting").glob("*.md")
            ]
        
        return prompts


# Global loader instance
_loader: Optional[PromptLoader] = None


def get_loader(base_path: Optional[str] = None) -> PromptLoader:
    """
    Get or create global PromptLoader instance.
    
    Args:
        base_path: Path to SYSTEM_PROMPT directory (used only on first call)
        
    Returns:
        PromptLoader instance
    """
    global _loader
    if _loader is None:
        _loader = PromptLoader(base_path=base_path)
    return _loader
