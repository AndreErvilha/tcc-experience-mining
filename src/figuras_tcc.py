"""Figuras de resultados do TCC (antes geradas fora do repositório, sem script).

  results/figures/comparison_metrics.png                 Silhouette, ARI (sem ruído), V-Measure e pureza dos 6 modelos
  results/figures/matriz_associacao_clusters_labels.png  % de cada rótulo em cada grupo do HDBSCAN + TF-IDF
  results/figures/tsne_clusters_visualizacao.png         t-SNE 2D dos grupos do HDBSCAN + TF-IDF

Mesmas bibliotecas e estilo das figuras do TCC (matplotlib, seaborn, scikit-learn). Entradas:
results/metrics/metrics_<modelo>.json, clusters/hdbscan_tf_idf.pkl e data/sessions_represented.csv (rótulo).
Uso: python src/figuras_tcc.py
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

OUT = "results/figures/"
MODELOS = ["hdbscan_gemini", "hdbscan_minilm", "hdbscan_tf_idf", "kmeans_gemini", "kmeans_minilm", "kmeans_tf_idf"]


def comparativo():
    m = {k: json.load(open(f"results/metrics/metrics_{k}.json", encoding="utf-8")) for k in MODELOS}
    series = [("Silhouette", "silhouette_score", "#4C7FAF"), ("ARI Sem Ruído", "ari_clean", "#F29A3F"),
              ("V-Measure", "v_measure", "#68A95E"), ("Pureza Real", "purity_real_experience", "#E3686B")]
    x = np.arange(len(MODELOS))
    w = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (nome, chave, cor) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, [m[k][chave] or 0 for k in MODELOS], w, label=nome, color=cor)
    ax.set_xticks(x, MODELOS, rotation=15, ha="right", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score das Métricas (0 a 1)", fontweight="bold")
    ax.set_xlabel("Experimentos Realizados (Tríade: TF-IDF, MiniLM, Gemini)", fontweight="bold")
    ax.set_title("Comparativo Consolidado de Desempenho: TF-IDF vs. all-MiniLM vs. Gemini Embedding",
                 fontweight="bold", fontsize=14, pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT + "comparison_metrics.png", dpi=150)
    plt.close(fig)


def matriz_e_tsne():
    cl = pickle.load(open("clusters/hdbscan_tf_idf.pkl", "rb"))
    lab = np.asarray(cl["cluster_labels"])
    rot = pd.read_csv("data/sessions_represented.csv").set_index("cd_sessao").loc[list(cl["session_ids"]), "primary_label"].values
    m = lab != -1
    tab = pd.crosstab(pd.Series(lab[m], name="grupo"), pd.Series(rot[m], name="rótulo"), normalize="index") * 100
    tab.index = [f"Cluster {k}" for k in tab.index]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(tab, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=1, linecolor="white",
                cbar_kws={"label": "% da Sessão no Cluster"}, ax=ax, vmin=0, vmax=100)
    ax.set_title("Matriz de Associação (%): Clusters Descobertos (HDBSCAN) vs. Rótulos Reais de Jornada",
                 fontweight="bold", fontsize=14, pad=12)
    ax.set_xlabel("Rótulo Real da Jornada de Interesse", fontweight="bold")
    ax.set_ylabel("Cluster Descoberto pelo Algoritmo", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT + "matriz_associacao_clusters_labels.png", dpi=150)
    plt.close(fig)

    X = cl["X"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    Y = TSNE(n_components=2, random_state=42, init="pca", perplexity=30).fit_transform(X)
    fig, ax = plt.subplots(figsize=(12, 8))
    ru = lab == -1
    ax.scatter(Y[ru, 0], Y[ru, 1], marker="x", s=18, c="#9BBBE0", alpha=0.6, linewidths=0.8, label="Ruído (-1)")
    cores = plt.get_cmap("tab20")
    for k in sorted(set(lab) - {-1}):
        s = lab == k
        ax.scatter(Y[s, 0], Y[s, 1], s=60, color=cores(k % 20), edgecolor="black", linewidth=0.4, label=f"Cluster {k}")
    ax.set_title("Visualização t-SNE 2D dos Clusters Comportamentais (Espaço Semântico HDBSCAN + TF-IDF)",
                 fontweight="bold", fontsize=14, pad=12)
    ax.set_xlabel("Dimensão t-SNE 1", fontweight="bold")
    ax.set_ylabel("Dimensão t-SNE 2", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + "tsne_clusters_visualizacao.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    comparativo()
    matriz_e_tsne()
    print("[+] Figuras em", OUT)
