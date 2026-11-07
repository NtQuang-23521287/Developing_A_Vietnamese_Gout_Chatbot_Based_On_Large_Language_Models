from gout_llmops.training.preference import labels_for, mine_pair, prompt_messages, select_prompts


def cand(idx, reward, text="câu trả lời", fatal=False, risks=(0, 0, 0), truncated=False, status="ok"):
    h, o, s = risks
    return {"idx": idx, "text": text, "reward": reward, "fatal": fatal, "truncated": truncated, "status": status,
            "raw": {"hallucination_level": h, "omission_risk": o, "safety_risk": s}}


def row(cands, risk_level="moderate", teacher="mẫu an toàn"):
    return {"id": "x", "risk_level": risk_level, "teacher_answers": [teacher], "candidates": cands}


def test_best_and_worst_become_pair():
    pair, reason = mine_pair(row([cand(0, 0.80), cand(1, 0.55), cand(2, 0.20)]))
    assert reason == "ok" and pair["chosen_idx"] == 0 and pair["rejected_idx"] == 2
    assert abs(pair["margin"] - 0.60) < 1e-9


def test_small_margin_is_dropped():
    pair, reason = mine_pair(row([cand(0, 0.70), cand(1, 0.62)]))
    assert pair is None and reason == "small_margin"


def test_chosen_must_be_clean_of_risk_and_not_fatal_or_truncated():
    risky = cand(0, 0.90, risks=(0, 2, 0))       # omission 2 -> không được làm chosen
    fatal = cand(1, 0.85, fatal=True)
    cut = cand(2, 0.88, truncated=True)
    ok = cand(3, 0.60)
    bad = cand(4, 0.10)
    pair, _ = mine_pair(row([risky, fatal, cut, ok, bad]))
    assert pair["chosen_idx"] == 3 and pair["rejected_idx"] == 4


def test_tie_prefers_shorter_chosen():
    pair, _ = mine_pair(row([cand(0, 0.8, text="dài " * 50), cand(1, 0.8, text="ngắn"), cand(2, 0.1)]))
    assert pair["chosen_idx"] == 1


def test_teacher_fallback_only_for_high_risk():
    bad = [cand(0, 0.30, risks=(0, 3, 2)), cand(1, 0.10, risks=(1, 3, 3))]
    pair, reason = mine_pair(row(bad, risk_level="critical"))
    assert reason == "ok_teacher_fallback" and pair["chosen"] == "mẫu an toàn" and pair["rejected_idx"] == 1
    pair, reason = mine_pair(row(bad, risk_level="moderate"))
    assert pair is None and reason == "no_eligible_chosen"
    pair, reason = mine_pair(row(bad, risk_level="critical"), {"teacher_fallback": False})
    assert pair is None


def test_unscored_candidates_ignored():
    pair, reason = mine_pair(row([cand(0, 0.9, status="judge_error"), cand(1, 0.2, status="judge_error")]))
    assert pair is None and reason == "no_scored_candidate"


def test_labels_and_prompt_format():
    assert labels_for("red_flag") == ("red_flag", "critical", "urgent_referral")
    assert labels_for("unknown_cat")[0] == "clinical_reasoning"
    msgs = prompt_messages("SYS", ["q1", "q2", "q3"], ["a1", "a2", "a3"],
                           [{"chunk_id": "c1", "text": "ngữ cảnh", "metadata": {}}])
    assert msgs[0]["role"] == "system" and "NGỮ CẢNH RAG" in msgs[0]["content"] and "ngữ cảnh" in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:]] == ["q1", "a1", "q2", "a2", "q3"]


def test_select_prompts_keeps_safety_first_and_is_deterministic():
    rows = [{"id": f"r{i:03d}", "category": "clinical_reasoning", "turns": ["q"]} for i in range(50)]
    rows += [{"id": f"s{i}", "category": "safety_trap", "turns": ["q"]} for i in range(5)]
    picked = select_prompts(rows, 10, seed=1)
    assert len(picked) == 10 and sum(r["category"] == "safety_trap" for r in picked) == 5
    assert [r["id"] for r in picked] == [r["id"] for r in select_prompts(rows, 10, seed=1)]
