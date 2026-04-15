"""Quick validation script to check for import errors."""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_imports():
    """Check all major imports."""
    errors = []

    # Test memory layer
    try:
        from memory import (
            load_history, save_history, load_topic, save_topic,
            load_takeover_flag, set_takeover_flag, publish_admin_alert,
            search_psychology_kb, search_user_memory_kb, upsert_user_memory_chunk,
            notify_human_admin, update_user_longterm_style, infer_ocean_increment,
        )
        logger.info("✅ Memory layer imports OK")
    except Exception as e:
        errors.append(f"Memory layer: {e}")
        logger.error(f"❌ Memory layer import failed: {e}")

    # Test prompts layer
    try:
        from prompts import (
            get_safety_check_prompt,
            get_intent_routing_prompt,
            get_simple_response_system_prompt,
            get_simple_response_user_prompt,
            get_out_of_scope_response,
            get_clarification_prompt,
            get_deep_reasoning_system_prompt,
            get_deep_reasoning_user_prompt,
            get_info_gap_assessment_prompt,
            get_memory_gate_prompt,
        )
        logger.info("✅ Prompts layer imports OK")
    except Exception as e:
        errors.append(f"Prompts layer: {e}")
        logger.error(f"❌ Prompts layer import failed: {e}")

    # Test utils.embeddings
    try:
        from utils.embeddings import embed_text, vector_literal, get_embeddings
        logger.info("✅ Utils.embeddings imports OK")
    except Exception as e:
        errors.append(f"Utils.embeddings: {e}")
        logger.error(f"❌ Utils.embeddings import failed: {e}")

    # Test graph.tools
    try:
        from graph.tools import get_llm, latest_user_message
        logger.info("✅ Graph.tools imports OK")
    except Exception as e:
        errors.append(f"Graph.tools: {e}")
        logger.error(f"❌ Graph.tools import failed: {e}")

    # Test graph.nodes
    try:
        from graph import nodes
        logger.info("✅ Graph.nodes imports OK")
    except Exception as e:
        errors.append(f"Graph.nodes: {e}")
        logger.error(f"❌ Graph.nodes import failed: {e}")

    # Test API layer
    try:
        from api import main
        logger.info("✅ API main imports OK")
    except Exception as e:
        errors.append(f"API main: {e}")
        logger.error(f"❌ API main import failed: {e}")

    if errors:
        logger.error(f"\n❌ Found {len(errors)} import errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    logger.info("\n✅ All imports validated successfully!")
    return True


if __name__ == "__main__":
    success = validate_imports()
    sys.exit(0 if success else 1)
