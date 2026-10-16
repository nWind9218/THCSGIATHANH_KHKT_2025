import dataclasses
import time

import pytest

from mimi_v2.teencode import RISK_ORDER, Scan, detect_register, max_risk, normalize, scan


def _msgs(*texts: str) -> list[dict]:
    return [{"role": "user", "content": t} for t in texts]


def test_normalize_basic_and_diacritics():
    assert normalize("Hôm nay ĐI HỌC mệt quá!!!") == "hom nay di hoc met qua"
    assert normalize("") == ""
    assert normalize(None) == ""  # type: ignore[arg-type]


def test_normalize_collapses_repeats():
    assert normalize("chếttttt") == "chet"
    assert normalize("quáaaaa   điiii") == "qua di"
    assert normalize("kkkk") == "kk"  # laughter is not "khong"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("k", "khong"), ("ko", "khong"), ("kh", "khong"), ("hok", "khong"), ("hông", "khong"), ("khum", "khong"),
        ("mún", "muon"), ("mun", "muon"), ("j", "gi"), ("z", "vay"), ("v", "vay"), ("zay", "vay"), ("dzậy", "vay"),
        ("dc", "duoc"), ("đc", "duoc"), ("ns", "noi"), ("bt", "biet"), ("ny", "nguoi yeu"), ("cx", "cung"), ("cg", "cung"),
        ("ik", "di"), ("t", "toi"), ("tui", "toi"), ("tao", "toi"), ("m", "ban"), ("mày", "ban"), ("oke", "ok"),
        ("okela", "ok"), ("oki", "ok"), ("trc", "truoc"), ("ntn", "nhu the nao"), ("bh", "bay gio"), ("bjo", "bay gio"),
        ("ae", "anh em"), ("ck", "chong"), ("vk", "vo"), ("nma", "nhung ma"), ("nhma", "nhung ma"), ("vs", "voi"),
        ("r", "roi"), ("s", "sao"), ("mn", "moi nguoi"),
    ],
)
def test_normalize_teencode_dictionary(raw, expected):
    assert normalize(raw) == expected


def test_normalize_sentences():
    assert normalize("t ko bít j z") == "toi khong biet gi vay"
    assert normalize("ns j z, hnay mệt vs bt") == "noi gi vay hom nay met voi biet"
    assert normalize("tu_tu") == "tu tu"
    assert normalize("Don't") == "dont"


def test_normalize_does_not_touch_plain_words():
    assert normalize("không muốn gì") == "khong muon gi"
    assert normalize("Bài kiểm tra") == "bai kiem tra"


@pytest.mark.parametrize(
    "text, code",
    [
        ("t mun tu tu", "suicide"),
        ("tui mún tự tử", "suicide"),
        ("kms", "suicide"),
        ("em muốn tự sát", "suicide"),
        ("tsat luon cho r", "suicide"),
        ("em muốn kill myself", "suicide"),
        ("em định kết liễu bản thân", "suicide"),
        ("muốn rạch tay", "self_harm_cut"),
        ("muon rach tay", "self_harm_cut"),
        ("e cắt cổ tay hôm qua", "self_harm_cut"),
        ("em tự làm đau bản thân", "self_harm_cut"),
        ("em muốn treo cổ", "hanging"),
        ("muon that co", "hanging"),
        ("em muốn nhảy lầu", "jump"),
        ("nhay cau cho xong", "jump"),
        ("em uống cả vỉ thuốc ngủ", "overdose"),
        ("uong thuoc ngu qua lieu", "overdose"),
        ("tối qua bị bố đánh đập", "abuse_physical"),
        ("hom qua bi me danh", "abuse_physical"),
        ("bị đánh đến chảy máu", "abuse_physical"),
        ("bố đánh em", "abuse_physical"),
        ("t bi ny danh", "abuse_physical"),
        ("em bị chú sờ vùng kín", "abuse_sexual"),
        ("bi so mo", "abuse_sexual"),
        ("em bị xâm hại", "abuse_sexual"),
        ("anh ấy ép em quan hệ", "abuse_sexual"),
        ("có người lạ xin ảnh nóng của em", "stranger_grooming"),
        ("nguoi la ru gap rieng", "stranger_grooming"),
        ("anh ấy rủ em gặp riêng", "stranger_grooming"),
    ],
)
def test_tier1_hits(text, code):
    assert code in scan(text).tier1, (text, scan(text))


@pytest.mark.parametrize(
    "text",
    [
        "chết cười với ông này", "xỉu ngang luôn", "muốn đấm bài kiểm tra", "game này chết hoài", "đánh game tới khuya",
        "đánh nhau trong game", "mệt chết đi được", "nóng chết mất", "bị đánh thức lúc 5h sáng", "bị đánh giá thấp",
        "bị bạn đánh bại trong game", "bị người yêu đá", "nói từ từ thôi", "tu tu da", "di tu tu thoi", "treo cờ ngày lễ",
        "từ hai giờ chiều", "tu hai gio chieu", "that co the", "rách tay áo rồi", "ba đánh nhau với chú",
        "hôm nay mình hơi buồn", "kiểm tra xong rồi", "bị cô đánh giá cao",
    ],
)
def test_no_tier1_on_jokes_and_benign(text):
    s = scan(text)
    assert s.tier1 == (), (text, s)


@pytest.mark.parametrize(
    "text, code",
    [
        ("em muốn chết", "want_die"),
        ("ko muon song nua", "want_die"),
        ("chết đi cho xong", "want_die"),
        ("sống làm gì", "want_die"),
        ("muốn biến mất", "disappear"),
        ("ngủ luôn ko dậy", "disappear"),
        ("nghỉ game cuộc đời", "disappear"),
        ("đây là lần cuối", "last_time"),
        ("e là gánh nặng", "burden"),
        ("lam phien moi nguoi", "burden"),
        ("không ai cần em", "nobody_cares"),
        ("có hay khong cung vay", "nobody_cares"),
        ("tạm biệt mn", "goodbye"),
        ("vĩnh biệt", "goodbye"),
        ("em hết hy vọng rồi", "hopeless"),
        ("mệt mỏi với mọi thứ", "hopeless"),
        ("tặng hết đồ chơi cho em", "give_away"),
        ("tụi nó bắt nạt mình", "bullied"),
    ],
)
def test_tier2_hits(text, code):
    s = scan(text)
    assert code in s.tier2, (text, s)


def test_tier2_joke_exclusion():
    assert "want_die" not in scan("muốn chết vì bài tập nhiều quá").tier2
    assert "want_die" not in scan("deadline dí muốn chết").tier2
    assert "want_die" not in scan("mệt chết đi được").tier2
    assert "want_die" in scan("muốn chết thật sự").tier2
    # joke context never hides want_die when other signals are present
    assert "want_die" in scan("không ai cần em, bài kiểm tra xong em muốn chết").tier2


def test_tier2_does_not_fire_on_plain_chat():
    for t in ["hôm nay đi học vui lắm", "mình cãi nhau với bạn thân", "mai có bài kiểm tra", "cảm ơn Mimi nha"]:
        s = scan(t)
        assert s.tier1 == () and s.tier2 == (), (t, s)


def test_scan_combined_and_sorted():
    s = scan("t mún tự tử, hết hy vọng rồi, tạm biệt mn")
    assert s.tier1 == ("suicide",)
    assert s.tier2 == tuple(sorted(s.tier2)) and set(s.tier2) >= {"hopeless", "goodbye"}
    assert s.normalized == normalize("t mún tự tử, hết hy vọng rồi, tạm biệt mn")


def test_scan_is_frozen_and_hashable():
    s = scan("kms")
    assert isinstance(s, Scan)
    assert hash(s) == hash(scan("kms"))
    assert {s: 1}[scan("kms")] == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.tier1 = ()  # type: ignore[misc]
    assert scan("") == Scan((), (), "")
    assert scan(None) == Scan((), (), "")  # type: ignore[arg-type]


def test_max_risk_ordering():
    assert RISK_ORDER == {"safe": 0, "uncertain": 1, "urgent": 2}
    assert max_risk("safe", "uncertain") == "uncertain"
    assert max_risk("uncertain", "safe") == "uncertain"
    assert max_risk("urgent", "uncertain") == "urgent"
    assert max_risk("safe", "safe") == "safe"
    assert max_risk("uncertain", "urgent") == "urgent"


def test_detect_register_dam():
    assert detect_register(_msgs("k ko hok j z")) == "dam"
    assert detect_register(_msgs("hnay t mệt vs bt luôn r")) == "dam"
    assert detect_register(_msgs("hom nay minh met qua, khong biet lam sao")) == "dam"
    assert detect_register(_msgs("chán quáaaaa kkk")) == "dam"


def test_detect_register_vua():
    assert detect_register([]) == "vua"
    assert detect_register(_msgs("")) == "vua"
    assert detect_register(_msgs("Hôm nay mình hơi mệt vì bài kiểm tra")) == "vua"
    assert detect_register([{"role": "assistant", "content": "k ko hok j z"}]) == "vua"


def test_detect_register_nghiem_tuc():
    long_text = (
        "Dạo này mình cảm thấy rất áp lực vì việc học, mỗi tối mình đều ngồi nghĩ về chuyện của gia đình và các bạn trong lớp, "
        "mình không biết nên bắt đầu kể từ đâu, nhưng mình muốn Mimi lắng nghe mình một lúc, vì mình thấy nặng lòng quá."
    )
    assert detect_register(_msgs(long_text)) == "nghiem_tuc"
    assert detect_register(_msgs("Ông nội mình vừa qua đời")) == "nghiem_tuc"
    assert detect_register(_msgs("mẹ mình mất tháng trước")) == "nghiem_tuc"
    assert detect_register(_msgs("bố mẹ mình ly hôn rồi")) == "nghiem_tuc"
    assert detect_register(_msgs("mình bị bắt nạt ở lớp")) == "nghiem_tuc"
    # heavy topic wins over teencode style
    assert detect_register(_msgs("t bi bo danh dap r hu hu")) == "nghiem_tuc"
    assert detect_register(_msgs("t mun kms")) == "nghiem_tuc"


def test_detect_register_uses_only_last_three_user_messages():
    msgs = _msgs("k ko hok j z") + [{"role": "assistant", "content": "ok"}] + _msgs(
        "Mình ổn", "Hôm nay đi học bình thường", "Mình ăn cơm với gia đình", "Rồi mình đi ngủ sớm"
    )
    assert detect_register(msgs) == "vua"


def test_performance_sanity():
    texts = [f"t mún tự tử lần {i}, bị bố đánh đập, ko ai cần, game này chết hoài hnay {i}" for i in range(1000)]
    start = time.perf_counter()
    for t in texts:
        scan(t)
    assert time.perf_counter() - start < 1.0
    start = time.perf_counter()
    for t in texts[:200]:
        detect_register(_msgs(t))
    assert time.perf_counter() - start < 1.0
