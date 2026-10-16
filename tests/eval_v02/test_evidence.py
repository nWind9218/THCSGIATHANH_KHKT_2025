from copy import deepcopy

import pytest

from mimi_eval.contracts import ContractError, text_hash
from mimi_eval.evidence import check_ref
from mimi_eval.validator import evaluate
from conftest import make_ref, rebind


def test_code_point_offsets_preserve_emoji_vietnamese_combining_marks(sample):
    c,_=sample
    text="🙂 Em buồn e\u0301 quá."
    text=text.replace("\\u0301", "\u0301")
    s=c["candidate_response"]
    s.update(text=text,sha256=text_hash(text))
    start=text.index("Em"); end=text.index(" quá")
    r=make_ref(s);r.update(start=start,end=end,quote=text[start:end])
    check_ref(c,r,{s["message_id"]})
    assert start==2  # UTF-16 would count the emoji as two units and return 3.
    bad=deepcopy(r);bad["start"]+=1
    with pytest.raises(ContractError):check_ref(c,bad)
    bad=deepcopy(r);bad["quote"]=r["quote"].replace("e\u0301","é")
    with pytest.raises(ContractError):check_ref(c,bad)


@pytest.mark.parametrize("change", [
    {"quote":"fabricated quote"}, {"role":"student"}, {"source_sha256":"0"*64},
    {"start":-1}, {"end":99999}, {"message_id":"unknown-source"},
])
def test_invalid_source_quote_span(sample,change):
    c,o=sample
    o["observed_actions"][0]["evidence_refs"][0].update(change)
    assert evaluate(c,o)["verdict"]=="INVALID"


def test_action_cannot_quote_student_even_if_literal(sample):
    c,o=sample
    o["observed_actions"][0]["evidence_refs"]=[make_ref(c["transcript"][-1])]
    assert evaluate(c,o)["verdict"]=="INVALID"


def test_kb_cannot_establish_student_boundary(sample):
    c,o=sample
    kb={"message_id":"kb1","role":"tool","text":"chỉ nghe","sha256":text_hash("chỉ nghe")}
    c["retrieval_evidence"]=[kb];rebind(c,o)
    o["label_evidence"][0]["evidence_refs"]=[make_ref(kb)]
    assert evaluate(c,o)["verdict"]=="INVALID"


def test_shared_multi_action_span_is_valid(suite):
    e=next(e for e in suite["cases"] if e["fixture_id"]=="T12-C1")
    a=e["scripted_observation"]["observed_actions"]
    assert a[0]["evidence_refs"]==a[1]["evidence_refs"]
    assert evaluate(e["input"],e["scripted_observation"])["verdict"]=="FAIL"
