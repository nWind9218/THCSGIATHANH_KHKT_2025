import json
import time
from types import SimpleNamespace

import httpx
import pytest

import api.facebook_webhook as webhook
from mimi_v2.core import CounselingEngine
from mimi_v2.messenger_format import to_messenger
from mimi_v2.store import InMemoryStore
from tests.test_v2_core import FakeModel, assessment

# (input, expected). Raw strings so the LaTeX reads as the model would write it.
MATH_CASES = [
    # delimiters
    (r"Giải $x^2 - 5x + 6 = 0$ nha", "Giải x² - 5x + 6 = 0 nha"),
    (r"$$x = 2$$", "x = 2"),
    (r"Ta có \(a + b\) và \[a - b\]", "Ta có a + b và\na - b"),
    (r"Mimi làm \$5 và \$10", "Mimi làm $5 và $10"),
    # fractions
    (r"\frac{1}{2}", "1/2"),
    (r"\dfrac{a}{b} + \tfrac{3}{4}", "a/b + 3/4"),
    (r"\frac{a+b}{c-d}", "(a+b)/(c-d)"),
    (r"\frac{x+1}{2}", "(x+1)/2"),
    (r"\frac{\sqrt{x+1}}{2}", "√(x+1)/2"),
    (r"\frac{-b \pm \sqrt{\Delta}}{2a}", "(-b ± √Δ)/(2a)"),
    (r"\frac{1}{\frac{a}{b}}", "1/(a/b)"),
    (r"\frac{x^{2}}{3}", "x²/3"),
    (r"\frac12", "1/2"),
    # roots
    (r"\sqrt{2}", "√2"),
    (r"\sqrt{x}", "√x"),
    (r"\sqrt{x+1}", "√(x+1)"),
    (r"\sqrt{b^2 - 4ac}", "√(b² - 4ac)"),
    (r"\sqrt[3]{27}", "∛27"),
    (r"\sqrt[3]{x+1}", "∛(x+1)"),
    (r"\sqrt[4]{x}", "⁴√(x)"),
    (r"\sqrt[n]{x}", "ⁿ√(x)"),
    # super / subscripts
    (r"x^2", "x²"),
    (r"x^{10}", "x¹⁰"),
    (r"x^{-1}", "x⁻¹"),
    (r"x^{n}", "xⁿ"),
    (r"x^{n+1}", "xⁿ⁺¹"),
    (r"x^2 + y^3", "x² + y³"),
    (r"e^{-x}", "e^(-x)"),
    (r"a^{b+c}", "a^(b+c)"),
    (r"x^k", "x^k"),
    (r"2^10", "2¹⁰"),
    (r"x^{2}+2x+1=0", "x²+2x+1=0"),
    (r"\Delta = b^2 - 4ac", "Δ = b² - 4ac"),
    (r"x_1, x_2", "x₁, x₂"),
    (r"x_{10}", "x₁₀"),
    (r"a_n", "aₙ"),
    (r"x_{i+1}", "xᵢ₊₁"),
    (r"x_{k+1}", "x_(k+1)"),
    (r"H_2O và CO_2", "H₂O và CO₂"),
    (r"\text{CH}_4", "CH₄"),
    (r"\alpha_1", "α₁"),
    (r"90^\circ", "90°"),
    (r"90^{\circ}", "90°"),
    (r"\angle A = 60\circ", "∠A = 60°"),
    (r"\text{cm}^2", "cm²"),
    # symbols
    (r"2 \times 3 \cdot 4 \div 5", "2 × 3 · 4 ÷ 5"),
    (r"a \pm b \mp c", "a ± b ∓ c"),
    (r"a \le b \leq c \ge d \geq e", "a ≤ b ≤ c ≥ d ≥ e"),
    (r"a \neq b \ne c \approx d", "a ≠ b ≠ c ≈ d"),
    (r"\infty \pi \alpha \beta \gamma \Delta \delta \theta \lambda \sigma", "∞ π α β γ Δ δ θ λ σ"),
    (r"A \Rightarrow B \Leftrightarrow C \rightarrow D \to E", "A ⇒ B ⇔ C → D → E"),
    (r"x \in A, y \notin B, A \subset B, A \cup B, A \cap B, \emptyset, \varnothing", "x ∈ A, y ∉ B, A ⊂ B, A ∪ B, A ∩ B, ∅, ∅"),
    (r"\triangle ABC \perp d \parallel e", "△ABC ⊥ d ∥ e"),
    (r"50\%", "50%"),
    (r"\left( a + b \right)", "( a + b )"),
    (r"\left\{ 1, 2 \right\}", "{ 1, 2 }"),
    (r"\{1; 2\}", "{1; 2}"),
    (r"\text{diện tích} = \mathrm{cm}", "diện tích = cm"),
    (r"\mathbf{v} \mathbb{R}", "v ℝ"),
    (r"a\quad b\, c\; d\! e", "a b c d e"),
    (r"1, 2, \cdots, n \ldots \dots", "1, 2, …, n … …"),
    (r"\sin x + \foo", "sin x + foo"),
    # environments
    ("\\begin{align*}\nx^2 - 5x + 6 &= 0 \\\\\n(x-2)(x-3) &= 0\n\\end{align*}",
     "x² - 5x + 6 = 0\n(x-2)(x-3) = 0"),
    (r"\begin{aligned} a &= 1 \\ b &= 2 \end{aligned}", "a = 1\nb = 2"),
    (r"\begin{cases} x = 1 \\ y = 2 \end{cases}", "x = 1\ny = 2"),
    (r"$$\begin{array}{cc} 1 & 2 \\ 3 & 4 \end{array}$$", "1 2\n3 4"),
    (r"Q\&A", "Q&A"),
]


@pytest.mark.parametrize("raw, expected", MATH_CASES)
def test_latex_becomes_unicode_plain_text(raw, expected):
    assert to_messenger(raw) == expected
    assert to_messenger(expected) == expected


MARKDOWN_CASES = [
    ("**Bước 1:** tính Δ", "Bước 1: tính Δ"),
    ("__đậm__ và *nghiêng* và _nghiêng_ và ~~gạch~~", "đậm và nghiêng và nghiêng và gạch"),
    ("***rất đậm***", "rất đậm"),
    ("dùng `x + 1` nha", "dùng x + 1 nha"),
    ("```python\nprint(1)\n```", "print(1)"),
    ("```\nx = 1\ny = 2\n```\nxong", "x = 1\ny = 2\nxong"),
    ("# Tiêu đề\nnội dung", "Tiêu đề\nnội dung"),
    ("### Bước 2 ###", "Bước 2"),
    ("- một\n* hai\n  - lồng", "• một\n• hai\n  • lồng"),
    ("1. một\n2. hai", "1. một\n2. hai"),
    ("trên\n\n---\n\ndưới", "trên\n\ndưới"),
    ("| Tên | Điểm |\n|---|:---:|\n| An | 9 |\n| Bình | 8 |", "Tên – Điểm\nAn – 9\nBình – 8"),
    ("|a|b|\n|-|-|\n|1|2|", "a – b\n1 – 2"),
    ("xem [trang này](https://example.com/a_b) nhé", "xem trang này (https://example.com/a_b) nhé"),
    ("xem [https://example.com](https://example.com)", "xem https://example.com"),
    ("**Cách làm:**\n- Chuyển vế: x^2 = 4\n- Lấy √ hai vế\n**Đáp số:** x = ±2",
     "Cách làm:\n• Chuyển vế: x² = 4\n• Lấy √ hai vế\nĐáp số: x = ±2"),
    ("a\n\n\n\n\nb   \nc", "a\n\nb\nc"),
]


@pytest.mark.parametrize("raw, expected", MARKDOWN_CASES)
def test_markdown_becomes_plain_text(raw, expected):
    assert to_messenger(raw) == expected
    assert to_messenger(expected) == expected


UNTOUCHED = [
    "Chào cậu, hôm nay cậu thế nào? 😆",
    "Gọi thầy cô qua 0962186108 hoặc 0915266338 nha, tổng đài 111 🫶",
    "Số +84 912-345-678 và 1️⃣ 2️⃣ 3️⃣ ✨",
    "Chia đôi là 1/2, được 50% điểm, 3/4 cái bánh, ngày 12/10/2025",
    "Phép tính 2 * 3 * 4 = 24 và a*b*c",
    "Tên file mimi_v2_final và snake_case_name nha",
    "^_^ T_T -_- :) <3",
    "Mua giá 5$ rồi 10$ nữa",
    "email a_1@example.com và https://example.com/x_1_y?z=1*2",
    "• gạch đầu dòng sẵn\n  • lồng sẵn",
    "x² + 2x + 1 = 0\n√(x+1)/2 ≥ 3 ± π°",
    "(a+b)/(c−d) và |x| + |y| và [1; 2]",
    "Bước 1: tính Δ = b² − 4ac\n👉 Đáp số: x = 2",
    "",
]


@pytest.mark.parametrize("text", UNTOUCHED)
def test_ordinary_text_is_unchanged(text):
    assert to_messenger(text) == text


def test_vietnamese_diacritics_and_emoji_are_preserved():
    text = "Ừ ơi, “quá trời” 🥺🫶 ✔️ — đỉnh nóc ✨"
    assert to_messenger(f"**{text}**") == text


def test_idempotent_on_a_messy_reply():
    raw = ("## Giải phương trình\n\n"
           r"Ta có $\Delta = b^2 - 4ac = 25 - 24 = 1$, nên $x_{1,2} = \frac{5 \pm \sqrt{1}}{2}$." "\n\n"
           "| x | y |\n|---|---|\n| 2 | 0 |\n\n"
           "- **Nghiệm:** $x_1 = 3$, $x_2 = 2$\n"
           "* Thử lại: `3^2 - 5*3 + 6 = 0`\n\n\n\n"
           r"$$\begin{align*} f(x) &= x^2 \\ g(x) &= \sqrt[3]{x} \end{align*}$$")
    once = to_messenger(raw)
    assert "$" not in once and "\\" not in once and "**" not in once and "##" not in once
    assert to_messenger(once) == once


def test_pure_and_fast():
    raw = (r"$\frac{\sqrt{x+1}}{2}$ " * 60) + "**x**\n" * 20
    start = time.perf_counter()
    for _ in range(20):
        to_messenger(raw)
    assert (time.perf_counter() - start) / 20 < 0.05
    assert to_messenger(None) == ""
    assert to_messenger("   ") == ""


def test_malformed_latex_never_raises():
    for raw in (r"\frac{1}{", r"\sqrt[3", r"x^{", r"x_{", r"\begin{align", r"\left", "\\", "$", "$$", "{{{", r"\frac", "x^", "x_",
                "a ^ b", "$$ unclosed", r"\text{abc"):
        out = to_messenger(raw)
        assert to_messenger(out) == out


# --- wiring ---------------------------------------------------------------------------

class LatexModel(FakeModel):
    reply = (r"Okela, mình giải nha ✨" "\n"
             r"**Bước 1:** $\Delta = b^2 - 4ac = 1$" "\n"
             r"**Bước 2:** $x = \frac{5 \pm \sqrt{1}}{2}$" "\n"
             r"- $x_1 = 3$, $x_2 = 2$")

    async def complete(self, messages, purpose):
        result = await super().complete(messages, purpose)
        if purpose == "reply":
            return SimpleNamespace(text=self.reply, usage=result.usage, metadata=result.metadata)
        return result


@pytest.mark.asyncio
async def test_core_reply_is_converted_to_plain_text():
    model = LatexModel([assessment(need="explain", intent="support", topic="school")])
    result = await CounselingEngine(model, InMemoryStore()).respond("a", "giải giúp t pt")
    expected = ("Okela, mình giải nha ✨\nBước 1: Δ = b² - 4ac = 1\nBước 2: x = (5 ± √1)/2\n• x₁ = 3, x₂ = 2")
    assert result["response_parts"][-1] == expected
    assert expected in result["response_parts"]
    assert result["messages"][-1]["content"] == expected
    joined = "\n".join(result["response_parts"])
    assert "$" not in joined and "\\" not in joined and "**" not in joined
    reply_prompt = [m for p, m in model.calls if p == "reply"][-1][0]["content"]
    assert "ĐỊNH DẠNG" in reply_prompt


@pytest.mark.asyncio
async def test_core_fixed_messages_are_not_reformatted():
    from mimi_v2 import prompts
    model = FakeModel(fail="reply")
    result = await CounselingEngine(model, InMemoryStore()).respond("a", "hello")
    assert result["response_parts"][-1] == prompts.REPLY_FAILURE_MESSAGE


@pytest.mark.asyncio
async def test_webhook_sends_plain_text(monkeypatch):
    sent = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            sent.append(kwargs["json"]["message"]["text"])
            return httpx.Response(200, request=httpx.Request("POST", url), json={"message_id": "ok"})

    monkeypatch.setattr(webhook, "FACEBOOK_PAGE_ACCESS_TOKEN", "token")
    monkeypatch.setattr(webhook.httpx, "AsyncClient", FakeClient)
    await webhook._send_messenger_message("1", r"**Đáp số:** $x = \frac{1}{2}$, $y = \sqrt{x+1}$, $z^2$")
    assert sent == ["Đáp số: x = 1/2, y = √(x+1), z²"]
    assert all("$" not in text and "\\frac" not in text for text in sent)


@pytest.mark.asyncio
async def test_webhook_chunking_applies_after_formatting(monkeypatch):
    sent = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            sent.append(kwargs["json"]["message"]["text"])
            return httpx.Response(200, request=httpx.Request("POST", url), json={})

    monkeypatch.setattr(webhook, "FACEBOOK_PAGE_ACCESS_TOKEN", "token")
    monkeypatch.setattr(webhook.httpx, "AsyncClient", FakeClient)
    await webhook._send_messenger_message("1", ("**dòng** $x^2$\n" * 400))
    assert len(sent) > 1 and all(len(text) <= 2000 for text in sent)
    assert all("*" not in text and "$" not in text and "^" not in text for text in sent)


def test_gradio_preview_is_plain_text():
    gr = pytest.importorskip("gradio")
    chat = gr.Chatbot(type="messages", render_markdown=False, latex_delimiters=[])
    assert chat.render_markdown is False and chat.latex_delimiters == []
    import inspect
    from mimi_v2 import gradio_prod
    source = inspect.getsource(gradio_prod)
    assert "render_markdown=False" in source and "latex_delimiters=[]" in source
