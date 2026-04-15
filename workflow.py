from __future__ import annotations

from graph.workflow import build_counseling_graph


def workflow():
    """Single active workflow for MVP counseling chat."""
    return build_counseling_graph()
