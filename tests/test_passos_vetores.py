"""Testes dos passos de representação e agrupamento sem parâmetros CLI."""
from pathlib import Path

import numpy as np
import pandas as pd

from experience_mining.io import carregar_agrupamento_pkl, carregar_representacao_pkl
from experience_mining.passos.agrupamento import AGRUPADORES
from experience_mining.passos.representacao import REPRESENTACOES

ROOT = Path(__file__).resolve().parent.parent


def test_registros_por_nome_completos():
    assert set(REPRESENTACOES) == {"tfidf", "minilm", "gemini"}
    assert set(AGRUPADORES) == {"kmeans", "hdbscan"}


def test_tfidf_e_agrupadores_sem_cli(tmp_path):
    sessions_df = pd.read_csv(ROOT / "data" / "v1.1" / "sessions_represented.csv")
    ref_tfidf = carregar_representacao_pkl(ROOT / "data" / "v1.0" / "tfidf_vectors.pkl")

    tfidf_out = tmp_path / "tfidf.pkl"
    gerado = REPRESENTACOES["tfidf"](sessions_df, output_path=tfidf_out)
    assert tfidf_out.exists()
    assert gerado["X"].shape == ref_tfidf["X"].shape
    assert np.allclose(gerado["X"], ref_tfidf["X"])

    km_out = tmp_path / "kmeans.pkl"
    km_res = AGRUPADORES["kmeans"](gerado, output_path=km_out, auto_k=True, random_state=42)
    assert km_out.exists()
    km_lido = carregar_agrupamento_pkl(km_out)
    assert len(km_lido["cluster_labels"]) == len(sessions_df)
    assert km_res["k"] >= 2

    hd_out = tmp_path / "hdbscan.pkl"
    hd_res = AGRUPADORES["hdbscan"](gerado, output_path=hd_out, auto_grid=True)
    assert hd_out.exists()
    hd_lido = carregar_agrupamento_pkl(hd_out)
    assert len(hd_lido["cluster_labels"]) == len(sessions_df)
    assert hd_res["n_clusters"] >= 2
