"""Deterministic Vietnamese teencode helpers: text normalisation, keyword safety scan, reply register.

Pure functions, stdlib only, no LLM. Used by the core to (1) pick the speaking register from how the student
writes and (2) raise (never lower) the safety risk via a keyword floor plus hints for the LLM safety classifier.

Trade-offs (principle: "thà nhắc thừa còn hơn bỏ sót"):
- tier1 = explicit statements (tự tử, rạch tay, treo cổ, nhảy lầu, bị bố đánh đập, bị sờ...). The scan is context-blind
  (it also matches "em không có ý nghĩ làm đau mình"), so core.py treats a tier-1 hit as a TRIGGER for a deeper
  multi-round safety review (negation, jokes, quotes, intent), not as a verdict; only if that review cannot run does a
  tier-1 hit fail closed to handoff. Hand-written negation rules are deliberately not attempted here. Known ambiguous bare forms
  without diacritics ("tu tu" = từ từ, "treo co" = treo cờ, "tu hai" = từ hai, "that co" = thật có, "nhay cau" =
  diving) are guarded by context rules; text with explicit diacritics "từ từ"/"treo cờ"/"từ hai" is never a hit.
- tier2 = vague signals, hint only. Clear jokes ("chết cười", "xỉu ngang", "mệt chết", "muốn chết vì bài tập")
  are removed before matching; want_die is dropped when it is the only vague signal and homework/game context is
  present. "lần cuối", "tạm biệt mọi người", "ngủ luôn không dậy" are kept even though they are often jokes.
- Normalisation expands ambiguous teencode greedily ("mun" = mụn is read as "muốn", "may" = mày/may is read as "bạn").
  That can only add matches around risky phrases, not remove them.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

Register = Literal["dam", "vua", "nghiem_tuc"]
RISK_ORDER = {"safe": 0, "uncertain": 1, "urgent": 2}


def max_risk(a: str, b: str) -> str:
    return a if RISK_ORDER.get(a, -1) >= RISK_ORDER.get(b, -1) else b


# ---------------------------------------------------------------- normalisation

_GROUPS: dict[str, str] = {
    "khong": "k ko kh hok hong khum khom hk hem koo",
    "muon": "mun",
    "gi": "j",
    "vay": "z v zay dzay",
    "duoc": "dc",
    "noi": "ns",
    "biet": "bt bik bit",
    "nguoi yeu": "ny",
    "nguoi yeu cu": "nyc",
    "cung": "cx cg",
    "di": "ik",
    "toi": "t tao tui",
    "ban": "m may",
    "ok": "oke okela oki okie okey oce okla",
    "truoc": "trc",
    "nhu the nao": "ntn",
    "bay gio": "bh bjo bgio",
    "anh em": "ae",
    "chong": "ck",
    "vo": "vk",
    "nhung ma": "nma nhma",
    "voi": "vs",
    "roi": "r",
    "sao": "s",
    "qua": "wa qa",
    "thich": "thik thix",
    "luon": "lun",
    "luc": "luk",
    "lam": "lm",
    "lam gi": "lmj",
    "dien thoai": "dt",
    "bo me": "bme",
    "hoc sinh": "hs",
    "giao vien": "gv",
    "tra loi": "tl",
    "bao nhieu": "bn bnhieu",
    "minh": "mik mk",
    "hom nay": "hnay",
    "hom qua": "hqua",
    "vui": "zui dzui",
    "dang": "dg dag",
    "cam on": "cmon camon",
    "xin loi": "sr",
    "nua": "nx",
    "u": "uk uh ukm",
    "thoi": "thui",
    "chua": "chx",
    "nguoi": "ng",
    "moi nguoi": "mn mng",
    "vi du": "vd",
    "gio": "gjo",
    "that": "tht",
    "khong biet": "kbiet",
}
_DICT: dict[str, str] = {k: v for v, keys in _GROUPS.items() for k in keys.split()}
_NOT_TEEN = {"may", "bit"}
_SLANG = {"vl", "vcl", "dm", "dmm", "dkm", "cl", "clm", "vkl", "cmnr", "hjhj", "hihi", "hehe", "huhu", "hix", "hic", "kk", "zz", "hh"}

_REPEAT = re.compile(r"([a-z])\1{2,}")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_APOS = re.compile(r"['’`]")


def _fold(text: str) -> str:
    t = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c)).replace("đ", "d")


def _analyze(text: str) -> tuple[str, int, int]:
    """Return (normalized, teencode_token_count, token_count) in one pass."""
    t = _NON_ALNUM.sub(" ", _APOS.sub("", _fold(text or "")))
    out: list[str] = []
    teen = 0
    toks = t.split()
    for tok in toks:
        if len(tok) >= 3 and len(set(tok)) == 1 and tok.isalpha():
            tok, stretched = tok[0] * 2, True
        else:
            new = _REPEAT.sub(r"\1", tok)
            stretched = new != tok
            tok = new
        exp = _DICT.get(tok)
        if stretched or tok in _SLANG or (exp is not None and exp != tok and tok not in _NOT_TEEN):
            teen += 1
        out.append(exp or tok)
    return " ".join(out), teen, len(toks)


def normalize(text: str) -> str:
    return _analyze(text)[0]


# ---------------------------------------------------------------- safety scan

@dataclass(frozen=True)
class Scan:
    tier1: tuple[str, ...]
    tier2: tuple[str, ...]
    normalized: str


def _rx(*parts: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9])")


_WHO = (
    r"(?:bo me|bo|me|ba me|ba|cha|chu|bac|di|duong|ke|anh em|anh trai|anh|chi gai|chi|ban trai|ban gai|ban hoc|"
    r"ban|thay giao|co giao|thay|co|ong noi|ba noi|ong|noi|ngoai|chong|vo|nguoi yeu|nguoi la|nguoi|dam ban|"
    r"nhom ban|may dua|moi nguoi)"
)
_FAM = r"(?:bo me|bo|me|ba|cha|chu|bac|di|duong|ke|anh|chi)"
_NOT_HIT = r"(?! (?:nhau|game|thuc|gia|mat|bai|lua|cap|dau|bom|truot|rot|rang|bong|co|ca|bac|chu|may|tu|lac|nhom|trong|giay|ngoai))"
_OBJ = r"(?:em|con|toi|minh|chau)"
_NOT_WHO = "".join(f"(?<!{w} )" for w in ("ban", "no", "anh", "chi", "me", "bo", "ai"))

_TIER1: dict[str, re.Pattern[str]] = {
    "suicide": _rx(
        r"tu sat", r"tsat", r"tusat", r"kms", r"kill my ?self", r"unalive", r"un alive", r"suicid\w*",
        r"ket lieu (?:ban than|cuoc doi|doi minh|doi)", r"ket thuc (?:cuoc doi|cuoc song|doi minh|doi toi|ban than)",
        r"tu ket thuc", r"end my life", r"end it all",
    ),
    "self_harm_cut": _rx(
        r"rach (?:co )?tay(?! ao)", r"cat (?:co )?tay(?! ao)", r"tu hai(?! (?:gio|ngay|nam|tuan|den|toi|tieng|phut|thang|ban|cai|lan|mon|tiet|tram|"
        r"muoi|nghin|ngan|trieu|chieu|sang|khoa|cu|ba|bon|nguoi|doi|nua|xe|con|chiec|vien|dua|em|anh|chi|day|viec))",
        r"tu lam dau", _NOT_WHO + r"lam dau (?:ban than|minh|chinh minh)", r"tu (?:cat|rach) (?:tay|minh|da)",
    ),
    "hanging": _rx(
        r"treo co(?! (?:to quoc|do|vang|lop|truong|tet|ngay le|len cot|ngoai|cho|nha))",
        r"that co(?! (?:the|ly|gi|phai|duoc|khong|nhieu|ai|ba|hoi|y|van|le|du|mot|su|tinh|dep|chai|ao|dong|tay|nghia|moi|can|cach))",
        r"tu treo",
    ),
    "jump": _rx(
        r"nhay (?:lau|cau|song|ho|xuong (?:song|duong|lau|cau|dat)|tu (?:tang|lau|cau|nha|ban cong|tren|mai))", r"gieo minh",
    ),
    "overdose": _rx(
        r"uong (?:het |ca |nhieu )?(?:vi |lo |hop |nam |\d+ vien )?thuoc (?:ngu|tru sau|chuot|doc)",
        r"uong (?:ca|het) (?:vi|lo|hop) thuoc", r"uong (?:\d+|nhieu) vien thuoc(?: ngu)?",
        r"qua lieu", r"uong thuoc (?:de )?chet",
    ),
    "abuse_physical": _rx(
        rf"bi {_WHO} danh{_NOT_HIT}",
        rf"bi {_WHO} (?:dap|dam|tat|xich|nhot|troi|giam|bop co|bao luc|bao hanh)",
        r"bi (?:danh dap|dap|xich|nhot|troi|tra tan|bao luc|bao hanh|hanh ha|bop co)",
        r"bi danh (?:dap|den|vao|tray|hoai|lien|tan|nhieu|chay mau|bam|hoi)",
        r"bi danh$",
        rf"{_FAM}(?: me)? (?:danh dap|danh|dap|tat|bop co|dam) {_OBJ}",
    ),
    "abuse_sexual": _rx(
        rf"bi (?:{_WHO} )?so (?:mo|soang|vung|nguc|mong|dui|nguoi|vao|lung|tay|chan)",
        rf"bi (?:{_WHO} )?so$",
        rf"bi (?:{_WHO} )?(?:xam hai|xam pham|hiep|cuong bao|quay roi|lam chuyen do|lam chuyen xau|lam chuyen nguoi lon|"
        r"ep quan he|ep lam chuyen|dung cham (?:vao )?(?:nguc|mong|vung kin|cho kin))",
        r"ep (?:toi |em |minh |con )?(?:quan he|lam chuyen do|ngu chung|lam tinh|xem phim nguoi lon|xem anh nong)",
        r"bat (?:toi |em |minh |con )?(?:cho )?(?:xem|coi|nhin|so) (?:vung kin|cho kin|anh nong|phim nguoi lon|bo phan)",
    ),
    "stranger_grooming": _rx(
        r"nguoi la.{0,25}(?:xin|doi|gui|bao gui|bat gui).{0,15}(?:anh|hinh|clip)",
        r"(?:xin|doi) anh (?:nong|khong mac|nhay cam|rieng tu|khoa than|ho hang|lo)",
        r"(?:doi|ru|hen|bao|muon) (?:toi |em |minh )?gap (?:rieng|mot minh|ngoai doi|o khach san|o nha nghi|tai nha)",
        r"nguoi la.{0,20}(?:ru|hen|doi|bao|muon).{0,12}(?:gap|di choi|di cung|den nha|ra ngoai)",
        r"nguoi la.{0,40}(?:bi mat|dung noi)",
        r"(?:quen|ket ban) (?:qua|tren) (?:mang|game|facebook|fb|zalo|tiktok|insta)\b.{0,40}(?:gap|ru|hen).{0,10}(?:rieng|mot minh|ngoai)",
    ),
}
_TU_TU = re.compile(r"(?<![a-z0-9])tu tu(?![a-z0-9])")
_TU_TU_INTENT = {"muon", "dinh", "toan", "nghi", "that"}
_TU_TU_SLOW_NEXT = {"da", "thoi", "nao", "nhe", "nha", "dung", "chut", "nua", "roi", "doi", "ha", "di", "se", "ma", "hay", "cho",
                    "noi", "lam", "ky", "giai", "xem", "tinh", "kiem", "nghi", "an", "doc", "viet", "chay", "tung", "cung"}
_TU_TU_SLOW_PREV = {"di", "noi", "lam", "an", "chay", "doc", "nghi", "tinh", "viet", "bao", "doi", "tu", "cu", "cu tu", "hay", "tung"}

_TIER2: dict[str, re.Pattern[str]] = {
    "want_die": _rx(
        r"muon (?:duoc )?chet", r"chet di cho (?:xong|roi|do kho|do khoc)", r"song lam gi", r"song de lam (?:gi|chi)",
        r"song tiep lam gi", r"khong (?:con )?muon (?:con )?(?:song|ton tai)", r"chan song", r"uoc (?:gi|chi) (?:minh |toi |em )?chet",
        r"neu (?:toi|em|minh|con) chet",
    ),
    "disappear": _rx(
        r"muon (?:minh )?(?:bien mat|khong ton tai|bien khoi)", r"bien mat khoi (?:the gioi|cuoc song|day|doi nay)",
        r"ngu (?:luon|mai) khong (?:day|thuc|tinh)", r"muon ngu (?:mai |luon )?khong (?:day|thuc)",
        r"nghi game cuoc doi", r"out game cuoc doi", r"out game doi", r"roi khoi the gioi", r"di mai khong ve",
        r"ra di (?:mai mai|vinh vien)", r"uoc gi (?:minh |toi |em )?bien mat",
    ),
    "last_time": _rx(r"lan cuoi cung", r"lan cuoi"),
    "burden": _rx(
        r"ganh nang", r"lam phien (?:moi nguoi|ba me|bo me|gia dinh|cac ban|ca nha|nguoi khac)", r"ton com",
        r"lam kho (?:ba me|bo me|gia dinh|moi nguoi)",
    ),
    "nobody_cares": _rx(
        r"khong ai (?:can|quan tam|thuong|yeu|nho|ben canh|o canh|thiet)", r"chang ai (?:can|quan tam|thuong|yeu)",
        r"(?:co|neu) (?:toi|em|minh)? ?hay khong(?: co (?:toi|em|minh))? cung (?:vay|chang sao|nhu nhau)",
        r"co hay khong cung vay", r"ai cung ghet (?:toi|em|minh)", r"co don",
    ),
    "goodbye": _rx(
        r"tam biet (?:moi nguoi|tat ca|het)", r"vinh biet", r"xin loi (?:moi nguoi|tat ca|vi tat ca)",
        r"cam on vi tat ca", r"dung nho toi nua", r"loi cuoi cung", r"nhan cuoi cung",
    ),
    "hopeless": _rx(
        r"het hy vong", r"mat het hy vong", r"khong con hy vong", r"bat luc", r"met moi voi moi thu", r"khong con y nghia",
        r"vo vong", r"tuyet vong", r"khong co loi thoat", r"khong con lam gi duoc", r"be tac", r"vo dung",
        r"khong thay tuong lai", r"khong xung dang",
    ),
    "give_away": _rx(
        r"(?:tang|cho|chia|tang lai|cho lai) (?:het|toan bo|tat ca) (?:do|do dac|do choi|sach vo|quan ao|vat dung|thu cung|dien thoai)",
        r"de lai (?:do|do dac|do choi|sach|quan ao|dien thoai|may tinh) (?:cho|lai cho)",
    ),
    "bullied": _rx(r"bat nat",r"bi co lap", r"bi tay chay", r"bi che cuoi", r"bi nguoi ta ghet"),
}
_JOKE_SPAN = re.compile(
    r"(?<![a-z0-9])(?:chet cuoi|cuoi chet|xiu ngang|(?:met|doi|nong|lanh) chet|"
    r"muon chet (?:(?:vi|boi|do|khi|luc|ma) )?(?:bai|deadline|dl|kiem tra|ktra|thi|hoc|diem|game|nong|doi|met|buon ngu|lanh|xau ho|ngai|ve sinh|nhieu))(?![a-z0-9])"
)
_JOKE_CTX = _rx(
    r"game", r"choi game", r"rank", r"boss", r"lag", r"deadline", r"dl", r"bai tap", r"bai kiem tra", r"bai thi", r"bai van",
    r"bai ve nha", r"bai vo", r"bai on", r"on bai", r"lam bai", r"nop bai", r"hoc bai", r"bai toan", r"bai ly", r"bai hoa",
    r"bai nhieu", r"bai qua nhieu", r"kiem tra", r"ktra", r"thi cu", r"on thi", r"ky thi", r"thi hoc ki", r"diem so", r"diem thap",
)
_MASKS = (("từ từ", " chamcham "), ("từ hai", " tuhaiok "), ("treo cờ", " treocook "), ("tự vẫn", " tự sát "))


def _tu_tu_hit(norm: str) -> bool:
    toks = norm.split()
    for i, t in enumerate(toks):
        if t != "tu" or i + 1 >= len(toks) or toks[i + 1] != "tu":
            continue
        prev = toks[i - 1] if i else ""
        nxt = toks[i + 2] if i + 2 < len(toks) else ""
        if prev in _TU_TU_INTENT or (prev not in _TU_TU_SLOW_PREV and nxt not in _TU_TU_SLOW_NEXT):
            return True
    return False


@lru_cache(maxsize=1024)
def _scan(text: str) -> Scan:
    low = unicodedata.normalize("NFC", text.lower())
    masked = low
    for src, dst in _MASKS:
        if src in masked:
            masked = masked.replace(src, dst)
    norm = normalize(text)
    m = norm if masked is low else normalize(masked)
    t1 = {code for code, rx in _TIER1.items() if rx.search(m)}
    if _tu_tu_hit(m):
        t1.add("suicide")
    t2m = _JOKE_SPAN.sub(" ", m)
    t2 = {code for code, rx in _TIER2.items() if rx.search(t2m)}
    if t2 == {"want_die"} and not t1 and _JOKE_CTX.search(m):
        t2.clear()
    return Scan(tuple(sorted(t1)), tuple(sorted(t2)), norm)


def scan(text: str) -> Scan:
    return _scan(text if isinstance(text, str) else "")


# ---------------------------------------------------------------- register

_HEAVY = _rx(
    r"qua doi", r"bi bat nat", r"ly hon", r"ly di", r"tram cam", r"dam tang", r"tang le", r"bao luc", r"xam hai", r"roi loan",
    r"hoang loan", r"lo au", r"bi co lap", r"da mat (?:bo|me|ba|ong|noi|ngoai|nguoi than|em|anh|chi|ban)",
    r"vua mat (?:bo|me|ba|ong|noi|ngoai|nguoi than|em|anh|chi|ban)",
    r"mat (?:bo|me|ba|cha|ong|ba noi|nguoi than|thu cung|con cho|con meo)(?! (?:ngu|tien|dien))",
    r"(?:bo|me|ba|cha|ong|noi|ngoai|em|anh|chi|ban|bac|chu|di|co)(?: minh| em| toi)? (?:da |vua |moi )?mat(?: roi| thang| nam| tuan| hom| ngay| vao| truoc| khi|$)",
    r"bo me (?:chia tay|cai nhau|bo nhau)",
)
_VN_WORDS = {
    "toi", "khong", "la", "va", "cua", "nhung", "minh", "em", "nay", "qua", "duoc", "co", "roi", "that", "hom", "met", "bi", "muon",
    "khi", "con", "ban", "cho", "trong", "di", "lam", "hoc", "nen", "cung", "sao", "dang", "nhieu", "gi", "ma", "vay", "nua", "lai",
    "ve", "voi", "den", "tu", "nao", "bo", "me", "thay", "biet", "noi", "nhu", "thi", "mot", "nhe", "nha",
}
_WORD = re.compile(r"[^\W\d_]+")


def detect_register(messages: list[dict]) -> Register:
    users = [str(m.get("content") or "") for m in (messages or []) if isinstance(m, dict) and m.get("role") == "user"][-3:]
    users = [u for u in users if u.strip()]
    if not users:
        return "vua"
    teen = ntok = 0
    heavy = False
    for u in users:
        s = _scan(u)
        if s.tier1 or _HEAVY.search(s.normalized):
            heavy = True
        _, t, n = _analyze(u)
        teen, ntok = teen + t, ntok + n
    if heavy:
        return "nghiem_tuc"
    if teen >= 2 or (ntok and teen / ntok >= 0.15):
        return "dam"
    words = _WORD.findall(unicodedata.normalize("NFC", " ".join(users).lower()))
    if len(words) >= 4:
        accented = sum(1 for w in words if any(ord(c) > 127 for c in w))
        if accented / len(words) < 0.25 and sum(1 for w in set(normalize(" ".join(users)).split()) if w in _VN_WORDS) >= 2:
            return "dam"
    last = _WORD.findall(unicodedata.normalize("NFC", users[-1].lower()))
    if len(last) >= 40 and teen == 0 and sum(1 for w in last if any(ord(c) > 127 for c in w)) / len(last) >= 0.4:
        return "nghiem_tuc"
    return "vua"
