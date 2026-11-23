"""Passos de cálculo e gravação de métricas, tabelas de jornada e sumário."""
from pathlib import Path

import pandas as pd

from experience_mining.configuracao import ConfigAgrupamentos, ConfigMetricas, ConfigRepresentacoes
from experience_mining.io import (
    carregar_agrupamento,
    carregar_representacao,
    ler_json,
    salvar_csv,
    salvar_json,
)
from experience_mining.nucleo.jornadas import MODELOS, REPRESENTACOES_PADRAO
from experience_mining.nucleo.metricas import (
    avaliar_agrupamento,
    calcular_cobertura_jornadas,
    calcular_contagem_jornadas,
    consolidar_sumario,
)

CHAVES_BASE = ("algorithm", "vectorizer", "n_samples", "k_discovered", "noise_count", "noise_ratio")
MAPA_METRICAS_INCLUDE = {
    "silhouette": ("silhouette_score",),
    "ari": ("ari_all",),
    "ari_clean": ("ari_clean",),
    "nmi": ("nmi_all",),
    "nmi_clean": ("nmi_clean",),
    "homogeneity": ("homogeneity",),
    "completeness": ("completeness",),
    "v_measure": ("v_measure",),
    "purity_macro_experience": ("purity_real_experience", "cluster_purity_details"),
}


def _resolver_caminho_cluster(clusters: ConfigAgrupamentos | Path, modelo: str) -> Path:
    if isinstance(clusters, ConfigAgrupamentos):
        return clusters.caminho_modelo(modelo)
    return Path(clusters) / f"{modelo}.pkl"


def _carregar_X_fallback(reps: ConfigRepresentacoes | None, modelo: str):
    if reps is None:
        return None
    suf = modelo.split("_", 1)[1]
    rep_path = reps.caminho_por_sufixo(suf)
    return carregar_representacao(rep_path)["X"] if Path(rep_path).exists() else None


def _filtrar_metricas(metricas: dict, include: list[str] | None) -> dict:
    if include is None:
        return metricas
    permitidas = set(CHAVES_BASE)
    for item in include:
        permitidas.update(MAPA_METRICAS_INCLUDE.get(item, ()))
    return {k: v for k, v in metricas.items() if k in permitidas}


def executar_metricas(
    secao: ConfigMetricas,
    clusters: ConfigAgrupamentos | Path,
    sessions_csv: Path,
    reps: ConfigRepresentacoes | None = None,
) -> None:
    if not secao.enabled:
        return
    rot = pd.read_csv(sessions_csv).set_index("cd_sessao")
    for nome, _, rotulo in REPRESENTACOES_PADRAO:
        for algo in ("kmeans", "hdbscan"):
            modelo = f"{algo}_{nome}"
            caminho = _resolver_caminho_cluster(clusters, modelo)
            if not caminho.exists():
                continue
            print(f"\n[*] Metricas {algo.upper()} {rotulo}")
            X_fb = _carregar_X_fallback(reps, modelo)
            data = carregar_agrupamento(caminho, X_fallback=X_fb)
            ids = list(data["session_ids"])
            data["primary_labels"] = rot.loc[ids, "primary_label"].tolist()
            data["experiences"] = rot.loc[ids, "experience"].tolist()
            metrics = _filtrar_metricas(avaliar_agrupamento(data), secao.include)
            out = Path(secao.output_metrics_dir) / f"metrics_{modelo}.json"
            salvar_json(metrics, out)
            print(f"[+] Salvo em: {out}")


def executar_tabelas_jornadas(
    secao: ConfigMetricas,
    clusters: ConfigAgrupamentos | Path,
    sessions_csv: Path,
) -> None:
    if not secao.enabled:
        return
    s = pd.read_csv(sessions_csv)
    contagem = calcular_contagem_jornadas(s)
    salvar_csv(contagem, Path(secao.output_dados_dir) / "jornadas_contagem.csv", ptbr=True)
    print(contagem.to_string(index=False))

    if secao.include is not None and "coverage_by_journey" not in secao.include:
        return

    clusters_por_modelo = {}
    modelos_ativos = []
    for m in MODELOS:
        caminho = _resolver_caminho_cluster(clusters, m)
        if caminho.exists():
            clusters_por_modelo[m] = carregar_agrupamento(caminho)
            modelos_ativos.append(m)
    if not modelos_ativos:
        return
    cobertura = calcular_cobertura_jornadas(s, clusters_por_modelo, modelos=modelos_ativos)
    salvar_csv(cobertura, Path(secao.output_metrics_dir) / "cobertura_por_jornada.csv", ptbr=True)
    print(cobertura.to_string(index=False))


def executar_sumario_metricas(secao: ConfigMetricas) -> None:
    if not secao.enabled:
        return
    metrics_files = sorted(Path(secao.output_metrics_dir).glob("metrics_*.json"))
    itens = [
        (mf.stem.replace("metrics_", "").upper().replace("_", " "), ler_json(mf))
        for mf in metrics_files
    ]
    df = consolidar_sumario(itens)
    if not df.empty:
        salvar_csv(df, secao.summary_csv_path, ptbr=False)
        print(f"[+] Sumario exportado para: {secao.summary_csv_path}")
        print("\n" + df.to_string(index=False))
    else:
        print("[!] Nenhum arquivo de metrica encontrado para consolidar.")
