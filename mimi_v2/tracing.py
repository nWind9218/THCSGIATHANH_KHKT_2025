"""Minimal per-request timing trace (LangSmith-style waterfall) with no external service.

A Trace records sequential spans relative to the moment the request started. It never
stores message text; attributes are numbers and short codes (tokens, error codes).
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


class Trace:
    def __init__(self):
        self.t0 = perf_counter()
        self.spans: list[dict] = []

    def add(self, name: str, start: float, end: float, **attrs) -> dict:
        span = {"name": name, "start": round(start - self.t0, 3), "duration": round(end - start, 3),
                "attrs": {k: v for k, v in attrs.items() if v is not None}}
        self.spans.append(span)
        return span

    @contextmanager
    def span(self, name: str, **attrs):
        start = perf_counter()
        extra: dict = {}
        try:
            yield extra
        except BaseException as exc:
            extra["error"] = type(exc).__name__
            raise
        finally:
            self.add(name, start, perf_counter(), **{**attrs, **extra})

    def finish(self) -> dict:
        total = round(perf_counter() - self.t0, 3)
        spans = sorted(self.spans, key=lambda s: s["start"])
        measured, covered_to = 0.0, 0.0
        for span in spans:  # union of intervals, so concurrent steps are counted once
            start, end = span["start"], span["start"] + span["duration"]
            measured += max(0.0, end - max(start, covered_to))
            covered_to = max(covered_to, end)
        other = round(max(total - measured, 0.0), 3)
        return {"total": total, "unaccounted": other, "spans": spans}


def model_attrs(usage: dict | None) -> dict:
    """Token counts that explain model latency (output and reasoning tokens dominate)."""
    usage = usage or {}
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    return {"input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
            "reasoning_tokens": details.get("reasoning_tokens") if isinstance(details, dict) else None}


def summarize(trace: dict) -> dict[str, float]:
    """{span name: seconds}; repeated names (e.g. a retry) are summed."""
    result: dict[str, float] = {}
    for span in trace.get("spans", []):
        result[span["name"]] = round(result.get(span["name"], 0.0) + span["duration"], 3)
    return result


def waterfall(trace: dict, width: int = 28) -> str:
    """Markdown table: step, start, duration, share of total, bar, attributes."""
    total = trace.get("total") or 0.0
    if not trace.get("spans") or total <= 0:
        return "_Chưa có dữ liệu thời gian._"
    rows = ["| Bước | Bắt đầu | Thời gian | % | Biểu đồ | Chi tiết |", "|---|---:|---:|---:|---|---|"]
    slowest = max(trace["spans"], key=lambda s: s["duration"])
    for span in trace["spans"]:
        left = min(int(span["start"] / total * width), width - 1)
        length = max(1, round(span["duration"] / total * width))
        bar = "·" * left + "█" * min(length, width - left)
        attrs = ", ".join(f"{k}={v}" for k, v in span["attrs"].items())
        name = f"**{span['name']}** ⚠" if span is slowest else span["name"]
        rows.append(f"| {name} | {span['start']:.2f}s | {span['duration']:.2f}s | {span['duration'] / total * 100:.0f}% | `{bar}` | {attrs} |")
    rows.append(f"| _không đo được (điều phối)_ | | {trace.get('unaccounted', 0):.2f}s | "
                f"{trace.get('unaccounted', 0) / total * 100:.0f}% | | |")
    rows.append(f"| **Tổng** | | **{total:.2f}s** | 100% | | |")
    return "\n".join(rows)
