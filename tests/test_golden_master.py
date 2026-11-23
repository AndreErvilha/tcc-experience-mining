"""Testes de regressão (golden master) contra tests/referencia/manifesto.json."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from experience_mining.configuracao import carregar_config_json
from experience_mining.orquestracao import Pipeline
from tests.gerar_referencia import (
    ROOT,
    SAIDA as MANIFESTO_PATH,
    inspecionar_pnml,
    sha256_texto_normalizado,
)


def carregar_manifesto() -> dict:
    if not MANIFESTO_PATH.exists():
        pytest.fail(f"Manifesto de referencia nao encontrado: {MANIFESTO_PATH}")
    return json.loads(MANIFESTO_PATH.read_text(encoding="utf-8"))


def resolver_caminho(rel_posix: str, raiz_results: Path, caminho_sessions_csv: Path) -> Path:
    if rel_posix == "data/sessions_represented.csv":
        return Path(caminho_sessions_csv)
    if rel_posix.startswith("results/"):
        return Path(raiz_results) / rel_posix[len("results/"):]
    raise ValueError(f"Caminho fora do escopo esperado: {rel_posix}")


def eh_artefato_petri(rel_posix: str) -> bool:
    return rel_posix.endswith(("dfg.png", "rede_petri.png", "rede_petri.pnml", "petri_parametros.csv"))


def comparar_saida_com_referencia(
    raiz_results: Path,
    caminho_sessions_csv: Path,
    manifesto: dict,
    grupo: str = "pipeline",
    incluir_petri: bool = True,
    apenas_arquivos: set[str] | None = None,
) -> None:
    sec = manifesto[grupo]

    for rel, esperado in sec["csv_json_sha256"].items():
        if not incluir_petri and eh_artefato_petri(rel):
            continue
        if apenas_arquivos is not None and rel not in apenas_arquivos:
            continue
        p = resolver_caminho(rel, raiz_results, caminho_sessions_csv)
        assert p.exists(), f"Arquivo ausente: {rel} ({p})"
        obtido = sha256_texto_normalizado(p)
        assert obtido == esperado, f"Divergencia em {rel}: esperado {esperado}, obtido {obtido}"

    for rel, info in sec["png_info"].items():
        if not incluir_petri and eh_artefato_petri(rel):
            continue
        if apenas_arquivos is not None and rel not in apenas_arquivos:
            continue
        p = resolver_caminho(rel, raiz_results, caminho_sessions_csv)
        assert p.exists(), f"PNG ausente: {rel} ({p})"
        assert p.stat().st_size > 0, f"PNG vazio: {rel}"
        arr = np.ascontiguousarray(mpimg.imread(str(p)))
        assert arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[1] > 0, f"PNG invalido: {rel}"
        if info["pixel_estavel"]:
            assert (int(arr.shape[0]), int(arr.shape[1])) == (info["altura"], info["largura"]), (
                f"Dimensoes diferentes em {rel}: esperado {(info['altura'], info['largura'])}, obtido {arr.shape[:2]}"
            )
            obtido_px = hashlib.sha256(arr.tobytes()).hexdigest()
            assert obtido_px == info["pixel_sha256"], f"Pixels diferentes em {rel}"

    for rel, info in sec["pnml_info"].items():
        if not incluir_petri and eh_artefato_petri(rel):
            continue
        if apenas_arquivos is not None and rel not in apenas_arquivos:
            continue
        p = resolver_caminho(rel, raiz_results, caminho_sessions_csv)
        assert p.exists(), f"PNML ausente: {rel} ({p})"
        obtido = inspecionar_pnml(p)
        assert obtido == info, f"PNML divergente em {rel}: esperado {info}, obtido {obtido}"


def test_referencia_passa_sobre_saida_atual():
    manifesto = carregar_manifesto()
    comparar_saida_com_referencia(
        ROOT / "results",
        ROOT / "data" / "v1.1" / "sessions_represented.csv",
        manifesto,
        grupo="pipeline",
        incluir_petri=True,
    )
    comparar_saida_com_referencia(
        ROOT / "results",
        ROOT / "data" / "v1.1" / "sessions_represented.csv",
        manifesto,
        grupo="analise_v1_1",
    )


def test_golden_master_detecta_divergencia(tmp_path):
    manifesto = carregar_manifesto()
    res_dir = tmp_path / "results"
    (res_dir / "metrics").mkdir(parents=True)
    shutil.copy2(ROOT / "results" / "summary_metrics.csv", res_dir / "summary_metrics.csv")
    shutil.copy2(
        ROOT / "results" / "metrics" / "metrics_hdbscan_tf_idf.json",
        res_dir / "metrics" / "metrics_hdbscan_tf_idf.json",
    )

    sumario = res_dir / "summary_metrics.csv"
    conteudo = sumario.read_text(encoding="utf-8")
    sumario.write_text(conteudo.replace("0.7268", "0.9999", 1), encoding="utf-8")

    alvo = {"results/summary_metrics.csv", "results/metrics/metrics_hdbscan_tf_idf.json"}
    with pytest.raises(AssertionError, match="summary_metrics.csv"):
        comparar_saida_com_referencia(
            res_dir,
            ROOT / "data" / "v1.1" / "sessions_represented.csv",
            manifesto,
            grupo="pipeline",
            apenas_arquivos=alvo,
        )


def _vincular_ou_copiar(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def test_analise_v1_1_em_pasta_temporaria(tmp_path):
    manifesto = carregar_manifesto()
    (tmp_path / "analises" / "v1.1").mkdir(parents=True)
    shutil.copy2(
        ROOT / "analises" / "v1.1" / "comparacao_v1.0_v1.1.py",
        tmp_path / "analises" / "v1.1" / "comparacao_v1.0_v1.1.py",
    )
    if (ROOT / "src").exists():
        shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    if (ROOT / "experience_mining").exists():
        shutil.copytree(
            ROOT / "experience_mining",
            tmp_path / "experience_mining",
            ignore=shutil.ignore_patterns("__pycache__"),
        )

    (tmp_path / "data" / "v1.1").mkdir(parents=True)
    _vincular_ou_copiar(
        ROOT / "data" / "v1.1" / "sessions_represented.csv",
        tmp_path / "data" / "v1.1" / "sessions_represented.csv",
    )
    (tmp_path / "clusters" / "v1.0").mkdir(parents=True)
    for p in (ROOT / "clusters" / "v1.0").glob("*.pkl"):
        _vincular_ou_copiar(p, tmp_path / "clusters" / "v1.0" / p.name)
    _vincular_ou_copiar(
        ROOT / "dataset" / "processed" / "total-rotulado.csv",
        tmp_path / "dataset" / "processed" / "total-rotulado.csv",
    )

    res = subprocess.run(
        [sys.executable, "analises/v1.1/comparacao_v1.0_v1.1.py"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Analise v1.1 falhou:\n{res.stdout}\n{res.stderr}"

    comparar_saida_com_referencia(
        tmp_path / "results",
        tmp_path / "data" / "v1.1" / "sessions_represented.csv",
        manifesto,
        grupo="analise_v1_1",
    )


def test_pipeline_em_pasta_temporaria(tmp_path):
    manifesto = carregar_manifesto()
    cfg = carregar_config_json(ROOT / "configs" / "v1.1.json", root=ROOT, validar_contratos=True)
    out = tmp_path / "results"
    cfg.output_dir = out
    cfg.eventos.sessions_csv_path = tmp_path / "sessions_represented.csv"
    cfg.metricas.output_metrics_dir = out / "metrics"
    cfg.metricas.output_dados_dir = out / "dados"
    cfg.metricas.summary_csv_path = out / "summary_metrics.csv"
    cfg.shap.output_jornadas_dir = out / "jornadas_por_grupo"
    cfg.shap.output_figures_dir = out / "figures"
    cfg.shap.output_dados_dir = out / "dados"
    cfg.jornadas_grupo.output_jornadas_dir = out / "jornadas_por_grupo"
    cfg.pm4py.output_jornadas_dir = out / "jornadas_por_grupo"
    cfg.figuras.output_figures_dir = out / "figures"

    Pipeline(cfg).executar()

    comparar_saida_com_referencia(
        out,
        cfg.eventos.sessions_csv_path,
        manifesto,
        grupo="pipeline",
        incluir_petri=True,
    )
    assert (out / "jornadas_por_grupo" / "hdbscan_tf_idf" / "grupos" / "grupo_00" / "dfg.dfg").exists()
