"""Chỉ số truy xuất so với chunk bằng chứng (evidence_chunk_ids) của testset.

Module thuần (không GPU, không gọi API) để unit test. Dùng cho scripts/benchmark_retrieval.py.

  Hit@k     = 1 nếu có ít nhất 1 chunk bằng chứng trong top-k
  Recall@k  = |bằng chứng ∩ top-k| / min(|bằng chứng|, k)   (chia cho min để top-k vẫn đạt được 1.0)
  RR@k      = 1 / hạng của chunk bằng chứng đầu tiên trong top-k (0 nếu không có)

Nhãn bằng chứng của testset v03 gán bằng luật/LLM và các tài liệu hướng dẫn trùng nội dung nhau, nên một đoạn
không có trong nhãn vẫn có thể chứa thông tin cần: các chỉ số ở đây là cận dưới của chất lượng truy xuất.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return 1.0 if any(c in gold for c in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        raise ValueError("gold rỗng")
    return len(gold & set(ranked[:k])) / min(len(gold), k)


def reciprocal_rank(ranked: Sequence[str], gold: set[str], k: int | None = None) -> float:
    for i, c in enumerate(ranked[:k] if k else ranked, start=1):
        if c in gold:
            return 1.0 / i
    return 0.0


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho một tỉ lệ (vd. Hit@3)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 5000, seed: int = 2026
                     ) -> tuple[float, float, float]:
    """Chênh lệch trung bình a − b trên cùng tập câu hỏi, kèm CI95 bootstrap ghép cặp."""
    if len(a) != len(b) or not a:
        raise ValueError("a, b phải cùng độ dài và khác rỗng")
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b, strict=True)]
    m = len(d)
    stats = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return (sum(d) / m, stats[int(0.025 * n)], stats[int(0.975 * n) - 1])


def mcnemar_exact(a: Sequence[float], b: Sequence[float]) -> float:
    """p-value hai phía của kiểm định McNemar chính xác cho hai chỉ số nhị phân ghép cặp (vd. Hit@3)."""
    if len(a) != len(b):
        raise ValueError("a, b phải cùng độ dài")
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
