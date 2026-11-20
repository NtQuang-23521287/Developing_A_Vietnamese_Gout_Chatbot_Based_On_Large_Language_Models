import json
from pathlib import Path

import pytest

from gout_llmops.core.config import load_yaml
from gout_llmops.rag.retriever import check_index_embedding
from gout_llmops.rag.segment import embedding_identity, get_segmenter

ROOT = Path(__file__).resolve().parents[1]


def test_default_segmenter_keeps_text():
    for name in (None, "", "none"):
        assert get_segmenter(name)("Bệnh nhân bị viêm khớp") == "Bệnh nhân bị viêm khớp"
    with pytest.raises(ValueError):
        get_segmenter("underthesea")


def test_pyvi_segmenter():
    pytest.importorskip("pyvi")
    assert get_segmenter("pyvi")("Bệnh nhân bị viêm khớp") == "Bệnh_nhân bị viêm khớp"


def test_embedding_identity_defaults():
    assert embedding_identity({"model_id": "m", "revision": "r"}) == {
        "model_id": "m", "revision": "r", "normalize_embeddings": True, "word_segmentation": "none"}


def test_index_must_match_embedding(tmp_path):
    emb = {"model_id": "m", "revision": "r", "normalize_embeddings": True}
    check_index_embedding(tmp_path, emb)  # không có manifest: bỏ qua
    (tmp_path / "manifest.json").write_text(json.dumps({"embedding": emb}), encoding="utf-8")
    check_index_embedding(tmp_path, {**emb, "word_segmentation": "none"})
    for bad in ({"word_segmentation": "pyvi"}, {"model_id": "khac"}, {"revision": "r2"}):
        with pytest.raises(RuntimeError):
            check_index_embedding(tmp_path, {**emb, **bad})


@pytest.mark.parametrize("config", ["configs/rag/rag_v1.yaml", "configs/rag/rag_v2_bkai.yaml"])
def test_committed_index_matches_config(config):
    cfg = load_yaml(ROOT / config)
    check_index_embedding(ROOT / cfg["output_dir"], cfg["embedding"])
