from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.edges import (
    route_after_info_gap,
    route_after_intent,
    route_after_load_memory,
    route_after_safety,
)
from graph.nodes import (
    clarification_node,
    deep_reasoning_node,
    handoff_human_node,
    info_gap_assessor_node,
    intent_routing_node,
    load_memory_node,
    out_of_scope_node,
    safety_gateway_node,
    save_memory_node,
    simple_response_node,
)
from graph.state import CounselingState


def build_counseling_graph():
    graph = StateGraph(CounselingState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("safety_gateway", safety_gateway_node)
    graph.add_node("handoff_human", handoff_human_node)
    graph.add_node("intent_routing", intent_routing_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("simple_response", simple_response_node)
    graph.add_node("info_gap_assessor", info_gap_assessor_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("deep_reasoning", deep_reasoning_node)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge(START, "load_memory")

    graph.add_conditional_edges(
        "load_memory",
        route_after_load_memory,
        {"safety_gateway": "safety_gateway", "end": END},
    )

    graph.add_conditional_edges(
        "safety_gateway",
        route_after_safety,
        {"handoff_human": "handoff_human", "intent_routing": "intent_routing"},
    )

    graph.add_conditional_edges(
        "intent_routing",
        route_after_intent,
        {
            "out_of_scope": "out_of_scope",
            "simple_response": "simple_response",
            "info_gap_assessor": "info_gap_assessor",
        },
    )

    graph.add_conditional_edges(
        "info_gap_assessor",
        route_after_info_gap,
        {"clarification": "clarification", "deep_reasoning": "deep_reasoning"},
    )

    graph.add_edge("out_of_scope", "save_memory")
    graph.add_edge("simple_response", "save_memory")
    graph.add_edge("clarification", "save_memory")
    graph.add_edge("deep_reasoning", "save_memory")
    graph.add_edge("handoff_human", END)
    graph.add_edge("save_memory", END)

    return graph.compile()
