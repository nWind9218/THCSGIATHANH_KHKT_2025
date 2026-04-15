"""
SYSTEM_PROMPT Registry

Central registry for accessing all system prompts throughout the application.
Provides convenient methods to get prompts by category and name.
"""

import logging
from typing import Optional, Dict, List
from .loader import get_loader, PromptLoader

logger = logging.getLogger(__name__)


class PromptRegistry:
    """
    Central registry for accessing system prompts.
    
    Usage:
        from SYSTEM_PROMPT.registry import prompt_registry
        
        # Get agent prompt
        prompt = prompt_registry.get_agent("mimi_school")
        
        # Get function prompt
        emotion_prompt = prompt_registry.get_function("emotion_extraction")
        
        # Get supporting prompt
        guidance = prompt_registry.get_supporting("response_guidance")
    """
    
    def __init__(self, loader: Optional[PromptLoader] = None):
        """
        Initialize registry with loader.
        
        Args:
            loader: PromptLoader instance (default: creates new one)
        """
        self.loader = loader or get_loader()
    
    # ========== AGENT PROMPTS ==========
    
    def get_agent(self, agent_name: str) -> str:
        """
        Get agent system prompt by name.
        
        Args:
            agent_name: Agent identifier (e.g., "mimi_school")
            
        Returns:
            Full prompt content
            
        Example:
            prompt = prompt_registry.get_agent("mimi_school")
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=user_input)
            ]
        """
        try:
            return self.loader.load_agent(agent_name)
        except FileNotFoundError as e:
            logger.error(f"❌ Agent prompt not found: {agent_name}")
            raise
    
    # ========== FUNCTION PROMPTS ==========
    
    def get_function(self, function_name: str) -> str:
        """
        Get function-specific prompt by name.
        
        Args:
            function_name: Function identifier
            Available functions:
                - "emotion_extraction": Extract emotion from message
                - "rag_selection": Select best answer from RAG
                - "response_refinement": Make response kid-friendly
                - "bot_planning": Plan response strategy
                - "summary_update": Update conversation summary
            
        Returns:
            Full prompt content
            
        Example:
            emotion_prompt = prompt_registry.get_function("emotion_extraction")
            formatted = emotion_prompt.format(
                messages=user_message,
                prev_intense=prev_intensity,
                conv_summary=summary
            )
        """
        try:
            return self.loader.load_function(function_name)
        except FileNotFoundError as e:
            logger.error(f"❌ Function prompt not found: {function_name}")
            raise
    
    # ========== SUPPORTING PROMPTS ==========
    
    def get_supporting(self, prompt_name: str) -> str:
        """
        Get supporting/guidelines prompt by name.
        
        Args:
            prompt_name: Supporting prompt identifier
            Available prompts:
                - "response_guidance": Response structure and guidelines
                - "summary_update": Conversation summary guidelines
            
        Returns:
            Full prompt content
        """
        try:
            return self.loader.load_supporting(prompt_name)
        except FileNotFoundError as e:
            logger.error(f"❌ Supporting prompt not found: {prompt_name}")
            raise
    
    # ========== BATCH OPERATIONS ==========
    
    def get_all_agents(self) -> Dict[str, str]:
        """
        Load all agent prompts at once.
        
        Returns:
            Dictionary mapping agent names to prompts
        """
        prompts = {}
        available = self.loader.list_prompts()["agents"]
        
        for agent_name in available:
            try:
                prompts[agent_name] = self.get_agent(agent_name)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load agent '{agent_name}': {e}")
        
        return prompts
    
    def get_all_functions(self) -> Dict[str, str]:
        """
        Load all function prompts at once.
        
        Returns:
            Dictionary mapping function names to prompts
        """
        prompts = {}
        available = self.loader.list_prompts()["functions"]
        
        for func_name in available:
            try:
                prompts[func_name] = self.get_function(func_name)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load function '{func_name}': {e}")
        
        return prompts
    
    def get_all_supporting(self) -> Dict[str, str]:
        """
        Load all supporting prompts at once.
        
        Returns:
            Dictionary mapping prompt names to content
        """
        prompts = {}
        available = self.loader.list_prompts()["supporting"]
        
        for prompt_name in available:
            try:
                prompts[prompt_name] = self.get_supporting(prompt_name)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load supporting '{prompt_name}': {e}")
        
        return prompts
    
    # ========== UTILITY METHODS ==========
    
    def list_prompts(self) -> Dict[str, List[str]]:
        """
        List all available prompts by category.
        
        Returns:
            Dictionary with categories and available prompt names
            
        Example:
            prompts = prompt_registry.list_prompts()
            # {
            #     "agents": ["mimi_school"],
            #     "functions": ["emotion_extraction", "rag_selection", ...],
            #     "supporting": ["response_guidance", "summary_update"]
            # }
        """
        return self.loader.list_prompts()
    
    def clear_cache(self):
        """Clear the prompt cache to force reload from disk."""
        self.loader.clear_cache()
        logger.info("✅ Prompt cache cleared")
    
    def validate_all(self) -> bool:
        """
        Validate that all prompts can be loaded.
        
        Returns:
            True if all prompts load successfully
        """
        try:
            prompts = self.list_prompts()
            
            # Try loading each
            for agent in prompts["agents"]:
                self.get_agent(agent)
            
            for func in prompts["functions"]:
                self.get_function(func)
            
            for supp in prompts["supporting"]:
                self.get_supporting(supp)
            
            logger.info("✅ All prompts validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prompt validation failed: {e}")
            return False
    
    def get_info(self) -> Dict:
        """
        Get registry information (for debugging).
        
        Returns:
            Dictionary with registry metadata
        """
        available = self.list_prompts()
        return {
            "loader_path": str(self.loader.base_path),
            "cache_size": len(self.loader._cache),
            "available_prompts": {
                "agents": len(available["agents"]),
                "functions": len(available["functions"]),
                "supporting": len(available["supporting"])
            },
            "prompt_names": available
        }


# Global registry instance
_registry: Optional[PromptRegistry] = None


def get_registry(loader: Optional[PromptLoader] = None) -> PromptRegistry:
    """
    Get or create global PromptRegistry instance.
    
    Args:
        loader: Custom loader (used only on first call)
        
    Returns:
        PromptRegistry instance
        
    Example:
        from SYSTEM_PROMPT.registry import get_registry
        registry = get_registry()
        prompt = registry.get_agent("mimi_school")
    """
    global _registry
    if _registry is None:
        _registry = PromptRegistry(loader=loader)
    return _registry


# Convenience instance for direct import
prompt_registry = PromptRegistry()


# ========== HELPER FUNCTIONS (For convenience) ==========

def get_agent_prompt(agent_name: str) -> str:
    """Convenience function to get agent prompt."""
    return prompt_registry.get_agent(agent_name)


def get_function_prompt(function_name: str) -> str:
    """Convenience function to get function prompt."""
    return prompt_registry.get_function(function_name)


def get_supporting_prompt(prompt_name: str) -> str:
    """Convenience function to get supporting prompt."""
    return prompt_registry.get_supporting(prompt_name)


def list_all_prompts() -> Dict[str, List[str]]:
    """Convenience function to list all prompts."""
    return prompt_registry.list_prompts()


# ========== USAGE EXAMPLES ==========

"""
# Example 1: Get agent prompt for LLM
from SYSTEM_PROMPT.registry import prompt_registry
from langchain_core.messages import SystemMessage, HumanMessage

agent_prompt = prompt_registry.get_agent("mimi_school")
messages = [
    SystemMessage(content=agent_prompt),
    HumanMessage(content=user_input)
]
response = llm.ainvoke(messages)


# Example 2: Get function prompt with formatting
emotion_template = prompt_registry.get_function("emotion_extraction")
formatted_prompt = emotion_template.format(
    messages=user_message,
    prev_intense=intensity_level,
    conv_summary=conversation_summary
)


# Example 3: Batch load all prompts
all_agents = prompt_registry.get_all_agents()
all_functions = prompt_registry.get_all_functions()


# Example 4: Validate registry
is_valid = prompt_registry.validate_all()
if is_valid:
    print("✅ All prompts loaded successfully")


# Example 5: Get registry info (debugging)
info = prompt_registry.get_info()
print(info)
# Output:
# {
#   'loader_path': '/path/to/SYSTEM_PROMPT',
#   'cache_size': 8,
#   'available_prompts': {
#     'agents': 1,
#     'functions': 4,
#     'supporting': 2
#   },
#   'prompt_names': {...}
# }
"""
