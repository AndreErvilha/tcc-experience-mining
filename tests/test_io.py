"""Testes do módulo de leitura e gravação (pkl legado e formato portátil)."""
import json
from pathlib import Path

import numpy as np
import pytest

from experience_mining.io import (
    carregar_agrupamento,
    carregar_agrupamento_pkl,
    carregar_agrupamento_portatil,
    carregar_representacao,
    carregar_representacao_pkl,
    carregar_representacao_portatil,
    portar_v1_para_portatil,
    salvar_agrupamento_portatil,
    salvar_representacao_portatil,
    validar_formato_chave,
)
from experience_mining.nucleo.jornadas import MODELOS

ROOT = Path(__file__).resolve().parent.parent


def test_ler_todos_pkl_da_v1():
    rep_nomes = ["tfidf_vectors.pkl", "minilm_embeddings.pkl", "gemini_embeddings.pkl"]
    ids_ref = None
    for arq in rep_nomes:
        d = carregar_representacao_pkl(ROOT / "data" / "v1.0" / arq)
        assert len(d["session_ids"]) == 516
        assert d["X"].shape[0] == 516
        assert not np.isnan(d["X"]).any()
        if ids_ref is None:
            ids_ref = list(d["session_ids"])
        else:
            assert list(d["session_ids"]) == ids_ref

    for m in MODELOS:
        cl = carregar_agrupamento_pkl(ROOT / "clusters" / "v1.0" / f"{m}.pkl")
        assert len(cl["cluster_labels"]) == 516
        assert list(cl["session_ids"]) == ids_ref


def test_roundtrip_formato_portatil_sem_rotulos(tmp_path):
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ids = ["SESS_001", "SESS_002"]

    rep_dir = tmp_path / "rep_tfidf"
    salvar_representacao_portatil(
        rep_dir,
        X,
        ids,
        meta={"vectorizer_type": "TF-IDF", "primary_labels": ["vazou"], "experiences": ["success"]},
    )
    meta = json.loads((rep_dir / "meta.json").read_text(encoding="utf-8"))
    assert "primary_labels" not in meta
    assert "experiences" not in meta
    assert "labels" not in meta

    lida = carregar_representacao_portatil(rep_dir)
    assert np.array_equal(lida["X"], X)
    assert lida["session_ids"] == ids
    assert lida["vectorizer_type"] == "TF-IDF"

    cl_dir = tmp_path / "cl_kmeans"
    labels = np.array([0, 1])
    salvar_agrupamento_portatil(
        cl_dir,
        ids,
        labels,
        params={"algorithm": "K-Means", "k": 2, "primary_labels": ["vazou"]},
    )
    params = json.loads((cl_dir / "params.json").read_text(encoding="utf-8"))
    assert "primary_labels" not in params
    assert "labels" not in params

    cl_lido = carregar_agrupamento_portatil(cl_dir)
    assert cl_lido["session_ids"] == ids
    assert np.array_equal(cl_lido["cluster_labels"], labels)


def test_contrato_invalido_falha(tmp_path):
    bad_dir = tmp_path / "bad_rep"
    with pytest.raises(ValueError):
        salvar_representacao_portatil(bad_dir, np.zeros((2, 3)), ["SESS_001"])

    # Formato divergente da chave (_pkl_path apontando para .npy)
    npy_file = tmp_path / "vec.npy"
    np.save(npy_file, {"X": np.zeros((2, 2)), "session_ids": ["S1", "S2"]})
    with pytest.raises(ValueError, match="Formato divergente"):
        validar_formato_chave("representations.tfidf.tfidf_pkl_path", npy_file)

    # Agrupamento com IDs trocados deve falhar
    cl_dir = tmp_path / "cl_trocado"
    salvar_agrupamento_portatil(cl_dir, ["SESS_002", "SESS_001"], [0, 1])
    with pytest.raises(ValueError, match="divergem"):
        carregar_agrupamento(cl_dir, expected_session_ids=["SESS_001", "SESS_002"])


def test_porte_v1_para_portatil_100_por_cento_igual(tmp_path):
    dst_data = tmp_path / "data" / "v1.0"
    dst_clusters = tmp_path / "clusters" / "v1.0"
    resumo = portar_v1_para_portatil(ROOT, dst_data, dst_clusters)

    assert set(resumo["representacoes"]) == {"tfidf", "minilm", "gemini"}
    assert len(resumo["agrupamentos"]) == 6

    tf_orig = carregar_representacao_pkl(ROOT / "data" / "v1.0" / "tfidf_vectors.pkl")
    tf_port = carregar_representacao(dst_data / "tfidf")
    assert np.array_equal(tf_orig["X"], tf_port["X"])
    assert list(tf_orig["feature_names"]) == list(tf_port["feature_names"])

    km_orig = carregar_agrupamento_pkl(ROOT / "clusters" / "v1.0" / "kmeans_tf_idf.pkl")
    km_port = carregar_agrupamento(dst_clusters / "kmeans_tf_idf", expected_session_ids=tf_port["session_ids"])
    assert np.array_equal(km_orig["cluster_labels"], km_port["cluster_labels"])
