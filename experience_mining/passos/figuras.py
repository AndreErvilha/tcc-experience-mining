"""Passo de geração das figuras consolidadas de resultados do TCC."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from experience_mining.configuracao import ConfigAgrupamentos, ConfigFiguras, ConfigRepresentacoes
from experience_mining.io import carregar_agrupamento, carregar_representacao, ler_json
from experience_mining.nucleo.jornadas import MODELOS_COMPARATIVO, NOMES_MODELOS


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


def _comparativo(metrics_dir: Path, figures_dir: Path) -> None:
    modelos_disponiveis = [k for k in MODELOS_COMPARATIVO if (Path(metrics_dir) / f"metrics_{k}.json").exists()]
    if not modelos_disponiveis:
        return
    m = {k: ler_json(Path(metrics_dir) / f"metrics_{k}.json") for k in modelos_disponiveis}
    series = [
        ("Silhouette", "silhouette_score", "#4C7FAF"),
        ("ARI Sem Ruído", "ari_clean", "#F29A3F"),
        ("V-Measure", "v_measure", "#68A95E"),
        ("Pureza Real", "purity_real_experience", "#E3686B"),
    ]
    x = np.arange(len(modelos_disponiveis))
    w = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (nome, chave, cor) in enumerate(series):
        ax.bar(
            x + (i - 1.5) * w,
            [m[k].get(chave) or 0 for k in modelos_disponiveis],
            w,
            label=nome,
            color=cor,
        )
    ax.set_xticks(x, modelos_disponiveis, rotation=15, ha="right", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score das Métricas (0 a 1)", fontweight="bold")
    ax.set_xlabel("Experimentos Realizados (Tríade: TF-IDF, MiniLM, Gemini)", fontweight="bold")
    ax.set_title(
        "Comparativo Consolidado de Desempenho: TF-IDF vs. all-MiniLM vs. Gemini Embedding",
        fontweight="bold",
        fontsize=14,
        pad=14,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(figures_dir) / "comparison_metrics.png", dpi=150)
    plt.close(fig)


def _matriz_e_tsne(
    clusters: ConfigAgrupamentos | Path,
    sessions_csv: Path,
    figures_dir: Path,
    tsne_seed: int = 42,
    reps: ConfigRepresentacoes | None = None,
    gerar_matriz: bool = True,
    gerar_tsne: bool = True,
    highlight_model: str | None = "hdbscan_tf_idf",
) -> None:
    if highlight_model is None:
        return
    hd_path = _resolver_caminho_cluster(clusters, highlight_model)
    if not hd_path.exists():
        return
    cl = carregar_agrupamento(hd_path, X_fallback=_carregar_X_fallback(reps, highlight_model))
    lab = np.asarray(cl["cluster_labels"])
    nome_m = NOMES_MODELOS.get(highlight_model, highlight_model)

    if gerar_matriz:
        rot = pd.read_csv(sessions_csv).set_index("cd_sessao").loc[list(cl["session_ids"]), "primary_label"].values
        m = lab != -1
        tab = pd.crosstab(pd.Series(lab[m], name="grupo"), pd.Series(rot[m], name="rótulo"), normalize="index") * 100
        tab.index = [f"Cluster {k}" for k in tab.index]
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            tab,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            linewidths=1,
            linecolor="white",
            cbar_kws={"label": "% da Sessão no Cluster"},
            ax=ax,
            vmin=0,
            vmax=100,
        )
        titulo_matriz = (
            "Matriz de Associação (%): Clusters Descobertos (HDBSCAN) vs. Rótulos Reais de Jornada"
            if highlight_model == "hdbscan_tf_idf"
            else f"Matriz de Associação (%): Clusters Descobertos ({nome_m}) vs. Rótulos Reais de Jornada"
        )
        ax.set_title(titulo_matriz, fontweight="bold", fontsize=14, pad=12)
        ax.set_xlabel("Rótulo Real da Jornada de Interesse", fontweight="bold")
        ax.set_ylabel("Cluster Descoberto pelo Algoritmo", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(Path(figures_dir) / "matriz_associacao_clusters_labels.png", dpi=150)
        plt.close(fig)

    if gerar_tsne:
        X = cl["X"]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        Y = TSNE(n_components=2, random_state=tsne_seed, init="pca", perplexity=30).fit_transform(X)
        fig, ax = plt.subplots(figsize=(12, 8))
        ru = lab == -1
        ax.scatter(Y[ru, 0], Y[ru, 1], marker="x", s=18, c="#9BBBE0", alpha=0.6, linewidths=0.8, label="Ruído (-1)")
        cores = plt.get_cmap("tab20")
        for k in sorted(set(lab) - {-1}):
            s = lab == k
            ax.scatter(
                Y[s, 0],
                Y[s, 1],
                s=60,
                color=cores(k % 20),
                edgecolor="black",
                linewidth=0.4,
                label=f"Cluster {k}",
            )
        titulo_tsne = (
            "Visualização t-SNE 2D dos Clusters Comportamentais (Espaço Semântico HDBSCAN + TF-IDF)"
            if highlight_model == "hdbscan_tf_idf"
            else f"Visualização t-SNE 2D dos Clusters Comportamentais ({nome_m})"
        )
        ax.set_title(titulo_tsne, fontweight="bold", fontsize=14, pad=12)
        ax.set_xlabel("Dimensão t-SNE 1", fontweight="bold")
        ax.set_ylabel("Dimensão t-SNE 2", fontweight="bold")
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(Path(figures_dir) / "tsne_clusters_visualizacao.png", dpi=150)
        plt.close(fig)


def executar_figuras_tcc(
    secao: ConfigFiguras,
    metrics_dir: Path,
    clusters: ConfigAgrupamentos | Path,
    sessions_csv: Path,
    reps: ConfigRepresentacoes | None = None,
) -> None:
    if not secao.enabled:
        return
    inc = secao.include
    quer_comp = inc is None or "comparison_metrics" in inc
    quer_mat = inc is None or "association_matrix" in inc
    quer_tsne = inc is None or "tsne" in inc
    if not (quer_comp or quer_mat or quer_tsne):
        return

    fig_dir = Path(secao.output_figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if quer_comp:
        _comparativo(metrics_dir, fig_dir)
    if (quer_mat or quer_tsne) and secao.highlight_model is not None:
        _matriz_e_tsne(
            clusters,
            sessions_csv,
            fig_dir,
            tsne_seed=secao.tsne_seed,
            reps=reps,
            gerar_matriz=quer_mat,
            gerar_tsne=quer_tsne,
            highlight_model=secao.highlight_model,
        )
    print(f"[+] Figuras em {fig_dir}")
