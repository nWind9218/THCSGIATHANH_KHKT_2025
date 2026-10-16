"""Turn LaTeX / Markdown left in a model reply into plain text that reads well on Messenger.

Messenger's Send API shows text messages verbatim: no Markdown, no LaTeX. The prompt
tells the model to write Unicode math, and this module is the deterministic safety net
for whatever slips through. `to_messenger` is pure, stdlib only, fast and idempotent
(`to_messenger(to_messenger(x)) == to_messenger(x)`).
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- tables

_SUP = {**{d: s for d, s in zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")},
        "+": "⁺", "-": "⁻", "−": "⁻", "=": "⁼", "(": "⁽", ")": "⁾", "n": "ⁿ", "i": "ⁱ"}
_SUB = {**{d: s for d, s in zip("0123456789", "₀₁₂₃₄₅₆₇₈₉")},
        "+": "₊", "-": "₋", "−": "₋", "=": "₌", "(": "₍", ")": "₎",
        "a": "ₐ", "e": "ₑ", "o": "ₒ", "x": "ₓ", "i": "ᵢ", "j": "ⱼ", "n": "ₙ", "m": "ₘ"}
_SCRIPT_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓᵢⱼₙₘ"

_SYMBOLS = {
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓", "le": "≤", "leq": "≤", "leqslant": "≤",
    "ge": "≥", "geq": "≥", "geqslant": "≥", "neq": "≠", "ne": "≠", "approx": "≈", "infty": "∞",
    "pi": "π", "alpha": "α", "beta": "β", "gamma": "γ", "Delta": "Δ", "delta": "δ", "theta": "θ",
    "lambda": "λ", "sigma": "σ", "Rightarrow": "⇒", "Leftrightarrow": "⇔", "rightarrow": "→", "to": "→",
    "in": "∈", "notin": "∉", "subset": "⊂", "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅",
    "angle": "∠", "triangle": "△", "perp": "⊥", "parallel": "∥", "circ": "°", "degree": "°",
    "cdots": "…", "ldots": "…", "dots": "…", "prime": "′",
    # a few more that models like to use
    "implies": "⇒", "iff": "⇔", "leftarrow": "←", "Leftarrow": "⇐", "sum": "∑", "int": "∫", "forall": "∀",
    "exists": "∃", "equiv": "≡", "sim": "∼", "cong": "≅", "mid": "|", "wedge": "∧", "vee": "∨",
    "land": "∧", "lor": "∨", "neg": "¬", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇",
    "setminus": "∖", "Pi": "Π", "Sigma": "Σ", "Omega": "Ω", "omega": "ω", "mu": "μ", "phi": "φ",
    "varphi": "φ", "epsilon": "ε", "varepsilon": "ε", "rho": "ρ", "tau": "τ", "Theta": "Θ", "Lambda": "Λ",
    "nabla": "∇", "partial": "∂", "propto": "∝", "ll": "≪", "gg": "≫", "langle": "⟨", "rangle": "⟩",
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "ast": "∗", "star": "⋆", "bullet": "•",
    "quad": " ", "qquad": " ", "enspace": " ", "thinspace": " ", "hspace": " ",
}
_DROP = {"displaystyle", "textstyle", "scriptstyle", "limits", "nolimits", "big", "Big", "bigg", "Bigg",
         "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "hline", "centering", "nonumber", "notag", "noindent",
         "newline", "par"}
_CONTENT = {"text", "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathscr", "mathfrak", "textbf",
            "textit", "textrm", "textsf", "texttt", "emph", "boldsymbol", "bm", "operatorname", "mbox", "hbox",
            "overline", "underline", "widehat", "widetilde", "hat", "tilde", "bar", "vec", "overrightarrow",
            "overleftarrow", "dot", "ddot", "cancel", "boxed", "fbox", "phantom", "mathop", "mathrel"}
_BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ"}
_TIGHT = {"angle", "triangle"}
_FRAC = {"frac", "dfrac", "tfrac", "cfrac"}

_SPECIAL = re.compile(r"[\\^_]")
_COMMAND_NAME = re.compile(r"[A-Za-z]+")
_PUNCT_SPACE = {",": " ", ";": " ", ":": " ", " ": " ", "!": "", "-": ""}

# Operators / separators that make a piece of an expression "compound".
_OPS = r"\s+\-−×÷·/()=<>±,;:|"
_SIMPLE = re.compile(rf"[−-]?(?:\d+[.,]\d+|[^{_OPS}]+)")
_GLYPH = re.compile(rf"(?:\d+(?:[.,]\d+)?|[^{_OPS}√∛∜])[{_SCRIPT_CHARS}]*")
_CALLISH_PREFIX = re.compile(rf"[\w√∛∜{_SCRIPT_CHARS}]+")


# --------------------------------------------------------------------------- helpers

def _match_paren(s: str, start: int) -> int:
    """Index of the ')' closing the '(' at `start`, or -1."""
    depth = 0
    for k in range(start, len(s)):
        if s[k] == "(":
            depth += 1
        elif s[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    return -1


def _wrapped(s: str) -> bool:
    return len(s) >= 2 and s[0] == "(" and _match_paren(s, 0) == len(s) - 1


def _callish(s: str) -> bool:
    """`√(x+1)`, `sin(x)`, `f(x)`: a prefix followed by one parenthesised group that ends the string."""
    k = s.find("(")
    return k > 0 and _CALLISH_PREFIX.fullmatch(s[:k]) is not None and _match_paren(s, k) == len(s) - 1


def _atomic(s: str) -> bool:
    return bool(_SIMPLE.fullmatch(s)) or _wrapped(s) or _callish(s)


def _paren(s: str, atomic: bool) -> str:
    return s if atomic else f"({s})"


def _fmt_frac(num: str, den: str) -> str:
    num, den = num.strip(), den.strip()
    if not den:
        return num
    if not num:
        return "/" + den
    # Denominators are held to a stricter standard: "1/2a" would read as (1/2)a.
    den_ok = bool(_GLYPH.fullmatch(den)) or _wrapped(den) or _callish(den) or (
        den[0] == "√" and bool(_GLYPH.fullmatch(den[1:])))
    return f"{_paren(num, _atomic(num))}/{_paren(den, den_ok)}"


def _fmt_sqrt(index: str | None, body: str) -> str:
    body = body.strip()
    atom = bool(_GLYPH.fullmatch(body)) or _wrapped(body) or _callish(body)
    if index is None or index.strip() == "2":
        return "√" + _paren(body, atom)
    index = index.strip()
    if index == "3":
        return "∛" + _paren(body, atom)
    if index and all(c in _SUP for c in index):
        prefix = "".join(_SUP[c] for c in index)
    else:
        prefix = index
    return prefix + "√" + (body if _wrapped(body) else f"({body})")


def _to_script(text: str, table: dict[str, str]) -> str | None:
    compact = "".join(text.split())
    if compact and all(c in table for c in compact):
        return "".join(table[c] for c in compact)
    return None


def _brace(s: str, start: int) -> tuple[str, int]:
    """Inner text of the {...} group opening at `start` and the index after it (nesting and \\{ aware)."""
    depth, k, n = 0, start, len(s)
    while k < n:
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:k], k + 1
        k += 1
    return s[start + 1:], n


def _arg(s: str, k: int) -> tuple[str, int]:
    """One LaTeX argument starting at/after `k`: a {group}, a \\command or a single character."""
    n = len(s)
    while k < n and s[k] in " \t":
        k += 1
    if k >= n:
        return "", n
    if s[k] == "{":
        return _brace(s, k)
    if s[k] == "\\":
        m = _COMMAND_NAME.match(s, k + 1)
        end = m.end() if m else min(k + 2, n)
        return s[k:end], end
    return s[k], k + 1


# --------------------------------------------------------------------------- LaTeX scanner

def _convert(s: str) -> str:
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        m = _SPECIAL.search(s, i)
        if not m:
            out.append(s[i:])
            break
        j = m.start()
        if j > i:
            out.append(s[i:j])
        c = s[j]
        if c == "\\":
            i = _command(s, j, out)
        elif c == "^":
            i = _caret(s, j, out)
        else:
            i = _underscore(s, j, out)
    return "".join(out)


def _command(s: str, i: int, out: list[str]) -> int:
    n = len(s)
    j = i + 1
    if j >= n:
        return j
    c = s[j]
    if not (c.isascii() and c.isalpha()):
        if c == "\\":
            k = j + 1
            if k < n and s[k] == "[":  # \\[2pt] spacing argument
                end = s.find("]", k)
                if end != -1 and end - k <= 8:
                    k = end + 1
            out.append("\n")
            return k
        if c in _PUNCT_SPACE:
            out.append(_PUNCT_SPACE[c])
        elif c == "|":
            out.append("‖")
        else:
            out.append(c)  # \{ \} \% \# \& \_ ... -> the character itself
        return j + 1
    name = _COMMAND_NAME.match(s, j).group(0)
    k = j + len(name)
    if name in _FRAC:
        a, k = _arg(s, k)
        b, k = _arg(s, k)
        out.append(_fmt_frac(_convert(a), _convert(b)))
    elif name == "sqrt":
        index = None
        p = k
        while p < n and s[p] in " \t":
            p += 1
        if p < n and s[p] == "[":
            end = s.find("]", p)
            if end != -1:
                index, k = _convert(s[p + 1:end]), end + 1
        a, k = _arg(s, k)
        out.append(_fmt_sqrt(index, _convert(a)))
    elif name in ("left", "right"):
        if k < n and s[k] == ".":
            k += 1
    elif name == "mathbb":
        a, k = _arg(s, k)
        out.append(_BLACKBOARD.get(a.strip(), _convert(a)))
    elif name in _CONTENT:
        a, k = _arg(s, k)
        out.append(_convert(a))
    elif name in _DROP:
        pass
    elif name in _SYMBOLS:
        out.append(_SYMBOLS[name])
        if name in _TIGHT:  # "\angle A" -> "∠A": LaTeX swallows the space after a control word
            while k < n and s[k] == " ":
                k += 1
    else:
        out.append(name)  # unknown command: keep the word, lose the backslash (\sin x -> sin x)
    return k


def _caret(s: str, i: int, out: list[str]) -> int:
    n, j = len(s), i + 1
    if j >= n:
        out.append("^")
        return j
    c = s[j]
    wrapped = False
    if c == "{":
        raw, k = _brace(s, j)
        wrapped = True
    elif c == "\\":
        raw, k = _arg(s, j)
    elif c == "(":
        end = _match_paren(s, j)
        if end == -1:
            out.append("^")
            return j
        raw, k, wrapped = s[j + 1:end], end + 1, True
    else:
        m = _CARET_TOKEN.match(s, j)
        if not m:
            out.append("^")
            return j
        raw, k = m.group(0), m.end()
    conv = _convert(raw)
    if conv in ("°", "′", "″"):
        out.append(conv)
        return k
    script = _to_script(conv, _SUP)
    if script is not None:
        out.append(script)
    elif _wrapped(conv):
        out.append("^" + conv)
    elif wrapped or len(conv) > 1:
        out.append(f"^({conv})")
    else:
        out.append("^" + conv)
    return k


_SUB_BASE_WORD = re.compile(r"[^\W_]+\Z")
_CARET_TOKEN = re.compile(r"[+\-−]?\d+|[A-Za-z]")
_DIGITS = re.compile(r"\d+")


def _underscore(s: str, i: int, out: list[str]) -> int:
    n, j = len(s), i + 1
    if j >= n:
        out.append("_")
        return j
    if s[j] == "{":
        raw, k = _brace(s, j)
        conv = _convert(raw)
        script = _to_script(conv, _SUB)
        if script is not None:
            out.append(script)
        elif _wrapped(conv):
            out.append("_" + conv)
        elif len(conv) > 1:
            out.append(f"_({conv})")
        else:
            out.append("_" + conv)
        return k
    tail = "".join(out[-3:])
    base_ok_digit = bool(tail) and (tail[-1] in ")}" or (
        _SUB_BASE_WORD.search(tail) is not None and len(_SUB_BASE_WORD.search(tail).group(0)) <= 2))
    m = _DIGITS.match(s, j)
    if m and base_ok_digit:
        out.append("".join(_SUB[d] for d in m.group(0)))
        return m.end()
    letter = s[j]
    if letter in _SUB and letter.isalpha():
        word = _SUB_BASE_WORD.search(tail)
        single_base = word is not None and len(word.group(0)) == 1 and word.group(0).isalpha()
        after_ok = j + 1 >= n or not (s[j + 1].isalnum() or s[j + 1] == "_")
        if single_base and after_ok and word.group(0).lower() != letter:
            out.append(_SUB[letter])
            return j + 1
    out.append("_")
    return j


# --------------------------------------------------------------------------- stages

_PLACE_DOLLAR = "\ue003"
_PLACE_AMP = "\ue004"
_FENCE = re.compile(r"```[^\n`]*\n?(.*?)```", re.S)
_FENCE_LINE = re.compile(r"^[ \t]*```.*$", re.M)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"!?\[([^\]\n]*)\]\(\s*((?:https?://|www\.|mailto:)[^)\s]+)\s*\)")
_PROTECT = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_BRACKET_ENV = re.compile(r"\\\[(.*?)\\\]", re.S)
_PAREN_ENV = re.compile(r"\\\((.*?)\\\)", re.S)
_BEGIN_END = re.compile(r"\\begin\{([A-Za-z*]+)\}(.*?)\\end\{\1\}", re.S)
_STRAY_ENV = re.compile(r"\\(?:begin|end)\{[^{}]*\}")
_DISPLAY = re.compile(r"\$\$(.*?)\$\$", re.S)
_INLINE = re.compile(r"(?<![\w$])\$(?=[^\s$])([^$\n]*?[^\s$])\$(?!\d)")


def _protect_urls(s: str) -> tuple[str, list[str]]:
    stash: list[str] = []

    def keep(m: re.Match) -> str:
        url = m.group(0)
        trail = ""
        while url and url[-1] in ".,;:!?":
            trail, url = url[-1] + trail, url[:-1]
        stash.append(url)
        return f"\ue001{len(stash) - 1}\ue002{trail}"

    return _PROTECT.sub(keep, s), stash


def _restore_urls(s: str, stash: list[str]) -> str:
    return re.sub("\ue001(\\d+)\ue002", lambda m: stash[int(m.group(1))], s)


def _environment(m: re.Match) -> str:
    env, body = m.group(1), m.group(2)
    if env.rstrip("*") in ("array", "tabular"):
        body = re.sub(r"^\s*\{[^{}]*\}", "", body)
    body = body.replace("\\&", _PLACE_AMP)
    body = re.sub(r"\\\\(?:\[[^\]\n]*\])?", "\n", body)
    body = re.sub(r"[ \t]*&[ \t]*", " ", body)
    lines = [ln.strip() for ln in body.split("\n")]
    return "\n" + "\n".join(ln for ln in lines if ln).replace(_PLACE_AMP, "&") + "\n"


def _strip_math_delimiters(s: str) -> str:
    s = _BRACKET_ENV.sub(lambda m: "\n" + m.group(1).strip() + "\n", s)
    s = _PAREN_ENV.sub(lambda m: m.group(1).strip(), s)
    s = _BEGIN_END.sub(_environment, s)
    s = _STRAY_ENV.sub("", s)
    s = _DISPLAY.sub(lambda m: "\n" + m.group(1).strip() + "\n", s)
    return _INLINE.sub(lambda m: m.group(1).strip(), s)


_HR = re.compile(r"^\s{0,3}([-*_=])(?:\s*\1){2,}\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")
_TABLE_ROW_SPACED = re.compile(r"^\s*\|\s.*\s\|\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)(?:\s+#+)?\s*$")
_BULLET = re.compile(r"^(\s*)[-*]\s+(\S.*)$")
_BOLD = re.compile(r"(?<![\w*])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![\w*])")
_BOLD_US = re.compile(r"(?<![\w_])__(?=\S)(.+?)(?<=\S)__(?![\w_])")
_ITALIC = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])")
_ITALIC_US = re.compile(r"(?<![\w_])_(?=[^\W_])(.+?)(?<=[^\s_])_(?![\w_])")
_STRIKE = re.compile(r"(?<![\w~])~~(?=\S)(.+?)(?<=\S)~~(?![\w~])")


def _is_table_sep(line: str) -> bool:
    return "|" in line and _TABLE_SEP.match(line) is not None and "-" in line


def _inline_markdown(line: str) -> str:
    line = _BOLD.sub(r"\1", line)
    line = _BOLD_US.sub(r"\1", line)
    line = _ITALIC.sub(r"\1", line)
    line = _ITALIC_US.sub(r"\1", line)
    return _STRIKE.sub(r"\1", line)


def _markdown(s: str) -> str:
    lines = s.split("\n")
    seps = [_is_table_sep(ln) for ln in lines]
    out: list[str] = []
    for idx, line in enumerate(lines):
        if seps[idx]:
            continue
        if _HR.match(line):
            out.append("")
            continue
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2 and (
                _TABLE_ROW_SPACED.match(line) or (idx > 0 and seps[idx - 1]) or (idx + 1 < len(lines) and seps[idx + 1])):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            line = " – ".join(c for c in cells if c)
        else:
            heading = _HEADING.match(line)
            if heading:
                line = heading.group(1)
            else:
                bullet = _BULLET.match(line)
                if bullet:
                    line = f"{bullet.group(1)}• {bullet.group(2)}"
        out.append(_inline_markdown(line))
    return "\n".join(out)


def _clean_whitespace(s: str) -> str:
    s = "\n".join(re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", ln.rstrip()) for ln in s.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", s).strip()


_QUICK = re.compile(r"[\\^_$`*#|~\[>=-]")


def to_messenger(text: str) -> str:
    """Plain-text, Unicode-math rendering of `text` (LaTeX and Markdown removed). Idempotent."""
    if not isinstance(text, str) or not text:
        return "" if not isinstance(text, str) else text
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    if _QUICK.search(s):
        s = s.replace("\\$", _PLACE_DOLLAR)
        s = _FENCE.sub(lambda m: m.group(1).strip("\n"), s)
        s = _FENCE_LINE.sub("", s)
        s = _INLINE_CODE.sub(r"\1", s)
        s = _LINK.sub(lambda m: m.group(2) if m.group(1).strip() in ("", m.group(2)) else f"{m.group(1).strip()} ({m.group(2)})", s)
        s, stash = _protect_urls(s)
        s = _strip_math_delimiters(s)
        s = _convert(s)
        s = _markdown(s)
        s = _restore_urls(s, stash).replace(_PLACE_DOLLAR, "$")
    return _clean_whitespace(s)
