import pytest

from gout_llmops.evaluation.retrieval_metrics import (
    hit_at_k,
    mcnemar_exact,
    paired_bootstrap,
    recall_at_k,
    reciprocal_rank,
    wilson_ci,
)

RANKED = ["c1", "c2", "c3", "c4", "c5"]


def test_hit_and_recall_at_k():
    assert hit_at_k(RANKED, {"c3"}, 3) == 1.0
    assert hit_at_k(RANKED, {"c4"}, 3) == 0.0
    # 2 trong 4 chunk bằng chứng nằm trong top-3: chia cho min(4, 3) = 3
    assert recall_at_k(RANKED, {"c1", "c3", "c4", "c9"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, {"c2"}, 3) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 3)


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"c2", "c4"}) == 0.5
    assert reciprocal_rank(RANKED, {"c4"}, k=3) == 0.0
    assert reciprocal_rank(RANKED, {"zz"}) == 0.0


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(60, 104)  # p = 0,577 -> khoảng [0,481; 0,667]
    assert 0.47 < lo < 0.49 and 0.66 < hi < 0.68
    assert wilson_ci(8, 8)[1] == 1.0
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_paired_bootstrap_and_mcnemar():
    a = [1.0] * 30 + [0.0] * 10
    b = [1.0] * 20 + [0.0] * 20
    diff, lo, hi = paired_bootstrap(a, b, n=2000)
    assert diff == pytest.approx(0.25)
    assert 0 < lo <= diff <= hi
    # 10 câu chỉ a đúng, 0 câu chỉ b đúng -> p = 2 / 2^10
    assert mcnemar_exact(a, b) == pytest.approx(2 / 1024)
    assert mcnemar_exact(a, a) == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
