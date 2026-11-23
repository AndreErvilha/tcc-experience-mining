"""Jornadas por grupo: as sessões de cada grupo de cada agrupamento e a matriz grupos × rótulo.

Para cada modelo (kmeans_|hdbscan_ x tf_idf|minilm|gemini), em results/jornadas_por_grupo/<modelo>/:
  resumo_grupos.csv                    uma linha por grupo (tamanho, jornada dominante, % de sucesso, composição)
  matriz_grupos[_KdeN].png             matriz grupos × rótulo (formato da figura do TCC); célula = % do grupo
                                       (% do rótulo); conversões, abandonos e neutro por último; ruído na última
                                       linha; dividida em partes quando há muitos grupos. Colunas pelo id da
                                       jornada (001, 002...), sucesso ao lado do atrito
  espaco_semantico_tsne.png            t-SNE 2D da representação do modelo, colorido por grupo
  espaco_semantico_tsne_3d.png         o mesmo em 3D, de dois ângulos
  grupos/grupo_NN/sessoes.csv          as sessões do grupo (grupos/ruido/ = grupo -1 do HDBSCAN)
  grupos/grupo_NN/dfg.png, rede_petri.*  DFG e Rede de Petri do grupo (src/petri_por_grupo.py, run_all.py --petri)
  grupos/grupo_NN/beeswarm.png         SHAP do grupo (src/shap_figures.py)
e, na raiz da pasta, resumo_grupos.csv com os 6 modelos.

Entradas: clusters/<modelo>.pkl e data/sessions_represented.csv (sequência, flags is_* e rótulo).
Uso: python src/jornadas_por_grupo.py [--labels data/sessions_represented.csv]
"""
import argparse
import math
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from jornadas_tabelas import JORNADAS, codigo, colunas_completas, tipo_rotulo as tipo

MODELOS = {"kmeans_tf_idf": "K-Means + TF-IDF", "kmeans_minilm": "K-Means + MiniLM",
           "kmeans_gemini": "K-Means + Gemini", "hdbscan_tf_idf": "HDBSCAN + TF-IDF",
           "hdbscan_minilm": "HDBSCAN + MiniLM", "hdbscan_gemini": "HDBSCAN + Gemini"}
OUT = "results/jornadas_por_grupo/"
LINHAS_POR_IMAGEM = 15
TINTA, TINTA2 = "#1f2328", "#59636e"
CMAP = {"conv": LinearSegmentedColormap.from_list("c", ["#ffffff", "#2a6fbb"]),
        "aban": LinearSegmentedColormap.from_list("a", ["#ffffff", "#c8372d"]),
        "neutro": LinearSegmentedColormap.from_list("n", ["#ffffff", "#6b6b67"])}
CSV = dict(sep=";", index=False, encoding="utf-8-sig")


def pasta_grupo(m, g):
    return os.path.join(OUT, m, "grupos", "ruido" if g == -1 else f"grupo_{g:02d}")


def resumo(m, d):
    linhas = []
    for g, x in d.groupby("grupo"):
        vc = x.primary_label.value_counts()
        n = len(x)
        linhas.append({"modelo": m, "grupo": g, "grupo_nome": x.grupo_nome.iloc[0], "sessoes": n,
                       "jornada_dominante": vc.index[0], "pct_jornada_dominante": round(vc.iloc[0] / n, 3),
                       "pct_sucesso": round((x.experience == "success").mean(), 3),
                       "n_jornadas_distintas": len(vc),
                       "composicao": "; ".join(f"{j} {c} ({c / n:.0%})" for j, c in vc.items()),
                       "mediana_eventos": int(np.median(x.num_events))})
    return pd.DataFrame(linhas)


LEGENDA = "\n".join("  ·  ".join(f"{j} {n}" for j, n, *_ in JORNADAS[i:i + 6]) for i in (0, 6))


def matriz(d, nome, m):
    """Mesmo formato da figura do TCC (seaborn, YlGnBu). Célula: % do grupo (% do rótulo que está no grupo).
    Colunas: conversões, abandonos e neutro por último; ruído (HDBSCAN) na última linha."""
    import seaborn as sns
    tab = pd.crosstab(d.grupo, d.primary_label)
    cols = colunas_completas(list(tab.columns))
    tab = tab.reindex(columns=cols, fill_value=0)
    linhas = [g for g in tab.index if g != -1] + ([-1] if -1 in tab.index else [])
    tab = tab.loc[linhas, cols]
    n_grupo, n_rot = tab.sum(axis=1), tab.sum(axis=0)
    pg, pj = tab.div(n_grupo, axis=0) * 100, tab.div(n_rot.replace(0, np.nan), axis=1).fillna(0) * 100
    partes = max(1, math.ceil(len(linhas) / LINHAS_POR_IMAGEM))
    por_parte = math.ceil(len(linhas) / partes)
    for p in range(partes):
        sel = linhas[p * por_parte:(p + 1) * por_parte]
        txt = np.where(pg.loc[sel].values > 0,
                       np.char.add(np.char.add(np.char.mod("%.1f", pg.loc[sel].values), "\n("),
                                   np.char.add(np.char.mod("%.1f", pj.loc[sel].values), ")")), "0.0")
        fig, ax = plt.subplots(figsize=(1.0 + 0.78 * len(cols), 2.8 + 0.55 * len(sel)))
        sns.heatmap(pg.loc[sel], annot=txt, fmt="", cmap="YlGnBu", vmin=0, vmax=100, linewidths=1,
                    linecolor="white", cbar_kws={"label": "% da Sessão no Cluster"}, annot_kws={"fontsize": 8}, ax=ax)
        ax.set_yticklabels([("Ruído" if g == -1 else f"Cluster {g}") + f" (n={n_grupo[g]})" for g in sel], rotation=0)
        ax.set_xticklabels([f"{codigo(c)} (N={n_rot[c]})" for c in cols], rotation=45, ha="right")
        for t, c in zip(ax.get_xticklabels(), cols):
            t.set_color(["#1f5fa3", "#b3302a", "#555555"][tipo(c)] if n_rot[c] else "#b0b0b0")
        if -1 in sel:
            ax.axhline(sel.index(-1), color="black", lw=2)
        sufixo = f" ({p + 1}/{partes})" if partes > 1 else ""
        ax.set_title(f"Matriz de Associação (%): Clusters Descobertos ({nome}) vs. Rótulos de Jornada{sufixo}",
                     fontweight="bold", fontsize=13, pad=12)
        ax.set_xlabel("Jornada da Tabela 1 revisada (azul: sucesso · vermelho: atrito · cinza: neutro)", fontweight="bold")
        ax.set_ylabel("Cluster Descoberto pelo Algoritmo", fontweight="bold")
        fig.text(0.01, 0.005, "Célula: % das sessões do cluster com o rótulo e, entre parênteses, % das sessões do rótulo que "
                 "estão no cluster. n = sessões do cluster; N = sessões com o rótulo (N=0 em cinza claro: nenhuma sessão); '+' = concluiu as duas jornadas sem formar uma composta.\n"
                 + LEGENDA, fontsize=7.5, color="#555555", ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.6 / fig.get_figheight(), 1, 1))
        arq = "matriz_grupos.png" if partes == 1 else f"matriz_grupos_{p + 1}de{partes}.png"
        fig.savefig(os.path.join(OUT, m, arq), dpi=150)
        plt.close(fig)


def tsne(cl, nome, m):
    """Mapa do espaço semântico (t-SNE 2D da representação do modelo), no formato da figura do TCC
    (src/figuras_tcc.py): ruído em x, grupos coloridos e o número de cada grupo no centro dele."""
    from sklearn.manifold import TSNE
    lab = np.asarray(cl["cluster_labels"])
    X = cl["X"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    Y = TSNE(n_components=2, random_state=42, init="pca", perplexity=30).fit_transform(X)
    grupos = sorted(set(lab) - {-1})
    cores = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ru = lab == -1
    if ru.any():
        ax.scatter(Y[ru, 0], Y[ru, 1], marker="x", s=18, c="#9BBBE0", alpha=0.6, linewidths=0.8,
                   label=f"Ruído (-1) (n={int(ru.sum())})")
    for k in grupos:
        sel = lab == k
        ax.scatter(Y[sel, 0], Y[sel, 1], s=60, color=cores[k % len(cores)], edgecolor="black", linewidth=0.4,
                   label=f"Cluster {k} (n={int(sel.sum())})")
        cx, cy = np.median(Y[sel], axis=0)
        ax.annotate(str(k), (cx, cy), ha="center", va="center", fontsize=9, fontweight="bold", color=TINTA,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title(f"Visualização t-SNE 2D dos Clusters Comportamentais (Espaço Semântico {nome})",
                 fontweight="bold", fontsize=14, pad=12)
    ax.set_xlabel("Dimensão t-SNE 1", fontweight="bold")
    ax.set_ylabel("Dimensão t-SNE 2", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8 if len(grupos) > 20 else 9,
              ncol=2 if len(grupos) > 24 else 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, m, "espaco_semantico_tsne.png"), dpi=150)
    plt.close(fig)


def tsne_3d(cl, nome, m):
    """Mesmo mapa em 3 dimensões (t-SNE 3D, matplotlib), visto de dois ângulos lado a lado."""
    from sklearn.manifold import TSNE
    lab = np.asarray(cl["cluster_labels"])
    X = cl["X"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    Y = TSNE(n_components=3, random_state=42, init="pca", perplexity=30).fit_transform(X)
    grupos = sorted(set(lab) - {-1})
    cores = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    fig = plt.figure(figsize=(18, 8.5))
    for n, (elev, azim) in enumerate([(22, -60), (22, 30)], start=1):
        ax = fig.add_subplot(1, 2, n, projection="3d")
        ru = lab == -1
        if ru.any():
            ax.scatter(Y[ru, 0], Y[ru, 1], Y[ru, 2], marker="x", s=14, c="#9BBBE0", alpha=0.5, linewidths=0.8,
                       label=f"Ruído (-1) (n={int(ru.sum())})")
        for k in grupos:
            sel = lab == k
            ax.scatter(Y[sel, 0], Y[sel, 1], Y[sel, 2], s=36, color=cores[k % len(cores)], edgecolor="black",
                       linewidth=0.3, depthshade=True, label=f"Cluster {k} (n={int(sel.sum())})")
            c = np.median(Y[sel], axis=0)
            ax.text(c[0], c[1], c[2], str(k), fontsize=8, fontweight="bold", color=TINTA)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("t-SNE 1", fontweight="bold")
        ax.set_ylabel("t-SNE 2", fontweight="bold")
        ax.set_zlabel("t-SNE 3", fontweight="bold")
        ax.set_title(f"Ângulo {n} (elevação {elev}°, azimute {azim}°)", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="center right", fontsize=8, ncol=2 if len(grupos) > 24 else 1)
    fig.suptitle(f"Visualização t-SNE 3D dos Clusters Comportamentais (Espaço Semântico {nome})",
                 fontweight="bold", fontsize=14)
    fig.subplots_adjust(left=0.0, right=0.84 if len(grupos) <= 24 else 0.78, wspace=0.02)
    fig.savefig(os.path.join(OUT, m, "espaco_semantico_tsne_3d.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/sessions_represented.csv")
    a = ap.parse_args()
    s = pd.read_csv("data/sessions_represented.csv")[["cd_sessao", "num_events", "text_representation"]]
    rot = pd.read_csv(a.labels)
    if "ordem_conclusao" not in rot.columns:
        rot["ordem_conclusao"] = ""
    flags = [c for c in rot.columns if c.startswith("is_")]
    rot["flags_ativas"] = rot[flags].apply(lambda r: " | ".join(c[3:] for c in flags if r[c] == 1), axis=1)
    base = s.merge(rot[["cd_sessao", "primary_label", "experience", "ordem_conclusao", "flags_ativas"]], on="cd_sessao")
    cols = ["grupo", "grupo_nome", "cd_sessao", "primary_label", "experience", "ordem_conclusao",
            "num_events", "flags_ativas", "text_representation"]
    todos = []
    for m, nome in MODELOS.items():
        cl = pickle.load(open(f"clusters/{m}.pkl", "rb"))
        lab = pd.DataFrame({"cd_sessao": cl["session_ids"], "grupo": np.asarray(cl["cluster_labels"])})
        d = lab.merge(base, on="cd_sessao", how="left")
        assert d.primary_label.notna().all(), f"sessões sem rótulo em {m}"
        d["grupo_nome"] = np.where(d.grupo == -1, "ruído", "grupo " + d.grupo.astype(str))
        for g, x in d.groupby("grupo"):
            os.makedirs(pasta_grupo(m, g), exist_ok=True)
            x[cols].to_csv(os.path.join(pasta_grupo(m, g), "sessoes.csv"), **CSV)
        r = resumo(m, d)
        r.to_csv(os.path.join(OUT, m, "resumo_grupos.csv"), **CSV)
        todos.append(r)
        matriz(d, nome, m)
        tsne(cl, nome, m)
        tsne_3d(cl, nome, m)
        print(f"[+] {m}: {d.grupo.nunique()} grupos")
    pd.concat(todos).to_csv(OUT + "resumo_grupos.csv", **CSV)


if __name__ == "__main__":
    main()
