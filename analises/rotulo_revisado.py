"""Rótulo da versão entregue x rótulo pela Tabela 1: o que muda na avaliação.

Rótulo entregue (v1): flags de dataset/processed/total-rotulado.csv (dataset/gerar_massas.py, só o evento de
conversão) e rótulo principal = primeira flag de sucesso na ordem das colunas (src/events2vector.py até a v1).
Rótulo revisado (v1.1): data/sessions_represented.csv, gerado a partir do dataset/processed/total-rotulado-v1.1.csv
(dataset/gerar_massas.py --saida total-rotulado-v1.1.csv, Tabela 1 revisada) pelo src/events2vector.py.
Os agrupamentos (clusters/*.pkl) são os mesmos. Na figura da migração, os rótulos aparecem pelo id da jornada
de cada versão (na v1.0, openfinance_pago era a 005 e copia_pix_paga a 010; na v1.1 são a 012 e a 011).
Conferido em 22/11/2026: o "antes" calculado aqui reproduz exatamente os results/metrics/*.json entregues.

Saídas: results/metrics/comparacao_v1.0_v1.1.csv, results/figures/comparacao_v1.0_v1.1.png e
results/dados/{sessoes_rotulo_revisado.csv, migracao_rotulos.csv/.png}.
Uso (da raiz do repositório): python analises/rotulo_revisado.py
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jornadas_tabelas import IDS_V10, IDS_V11, codigo, colunas_completas, tipo_rotulo  # noqa: E402
from sklearn.metrics import (adjusted_rand_score, homogeneity_completeness_v_measure,
                             normalized_mutual_info_score)

OUT = "results/dados/"
MODELOS = {"kmeans_tf_idf": "K-Means + TF-IDF", "kmeans_minilm": "K-Means + MiniLM",
           "kmeans_gemini": "K-Means + Gemini", "hdbscan_tf_idf": "HDBSCAN + TF-IDF",
           "hdbscan_minilm": "HDBSCAN + MiniLM", "hdbscan_gemini": "HDBSCAN + Gemini"}
CURTO = {"kmeans_tf_idf": "KM\nTF-IDF", "kmeans_minilm": "KM\nMiniLM", "kmeans_gemini": "KM\nGemini",
         "hdbscan_tf_idf": "HD\nTF-IDF", "hdbscan_minilm": "HD\nMiniLM", "hdbscan_gemini": "HD\nGemini"}

V1 = "dataset/processed/total-rotulado.csv"       # versão entregue
REVISADO = "data/sessions_represented.csv"        # Tabela 1 revisada (gerar_massas.py + events2vector.py)


def rotulo_v1(csv):
    """Rótulo e experience da versão entregue, com a lógica do events2vector.py da v1."""
    ev = pd.read_csv(csv)
    flags = [c for c in ev.columns if c.startswith("is_")]
    g = ev.groupby("cd_sessao")[flags].max()
    def prim(r):
        ativos = [c[3:] for c in flags if r[c] == 1]
        suc = [a for a in ativos if "sem_converter" not in a]
        return suc[0] if suc else (ativos[0] if ativos else "neutro")
    def exp(r):
        if any(r[c] == 1 for c in flags if "sem_converter" not in c):
            return "success"
        return "attrition" if any(r[c] == 1 for c in flags if "sem_converter" in c) else "none"
    return pd.DataFrame({"primary_label": g.apply(prim, axis=1), "experience": g.apply(exp, axis=1)})


def metricas(lab, y, exp):
    lab, y, exp = np.asarray(lab), np.asarray(y, dtype=object), np.asarray(exp, dtype=object)
    m = lab != -1
    h, c, v = homogeneity_completeness_v_measure(y[m], lab[m])
    # pureza por jornada: fração do grupo que é a jornada dominante (média ponderada, sem ruído)
    dom = sum(pd.Series(y[m & (lab == k)]).value_counts().iloc[0] for k in set(lab[m]))
    # pureza sucesso x abandono (mesma fórmula do src/eval_metrics.py)
    num = den = 0
    for k in set(lab[m]):
        e = exp[lab == k]
        s, n = (e == "success").sum(), np.isin(e, ["attrition", "error"]).sum()
        if s + n:
            num += max(s, n)
            den += s + n
    return {"ari_clean": adjusted_rand_score(y[m], lab[m]), "nmi_clean": normalized_mutual_info_score(y[m], lab[m]),
            "homogeneity": h, "completeness": c, "v_measure": v, "pureza_jornada": dom / m.sum(),
            "purity_real_experience": num / den if den else np.nan}


# ---------------------------------------------------------------- figuras
AZUL, CINZA, TINTA, TINTA2 = "#2a6fbb", "#b9bec7", "#1f2328", "#59636e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d0d7de", "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False,
                     "axes.spines.right": False})
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cfe0f3", AZUL, "#123d6b"])


def fig_metricas(df):
    painel = [("pureza_jornada", "Pureza por jornada"), ("v_measure", "V-measure"),
              ("ari_clean", "ARI (sem ruído)"), ("purity_real_experience", "Pureza sucesso × abandono")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    x = np.arange(len(MODELOS))
    for ax, (col, tit) in zip(axs, painel):
        a = df.set_index("modelo").loc[list(MODELOS), f"{col}_antes"].values
        d = df.set_index("modelo").loc[list(MODELOS), f"{col}_depois"].values
        ax.bar(x - 0.2, a, 0.38, color=CINZA, label="Rótulo v1.0 (entregue)")
        ax.bar(x + 0.2, d, 0.38, color=AZUL, label="Rótulo v1.1 (Tabela 1 revisada)")
        for i, (va, vd) in enumerate(zip(a, d)):
            ax.text(i + 0.2, vd + 0.015, f"{vd:.2f}", ha="center", va="bottom", fontsize=7, color=TINTA)
        ax.set_title(tit, fontsize=10, color=TINTA, loc="left")
        ax.set_xticks(x, [CURTO[m] for m in MODELOS], fontsize=7.5)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", color="#eef1f4", lw=0.8)
        ax.set_axisbelow(True)
    axs[0].legend(frameon=False, loc="upper left", fontsize=8, ncol=2, bbox_to_anchor=(0, 1.18))
    fig.text(0.01, -0.06, "KM = K-Means · HD = HDBSCAN. Métricas sem as sessões de ruído; os agrupamentos são os mesmos do TCC, só o rótulo muda.",
             fontsize=8, color=TINTA2, ha="left")
    fig.suptitle("Avaliação dos agrupamentos: rótulo v1.0 × rótulo v1.1",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TINTA, y=1.06)
    fig.savefig("results/figures/comparacao_v1.0_v1.1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def heatmap(tab, caminho, titulo, rodape, rotulo_linhas=None, pct_por_linha=True):
    val = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100 if pct_por_linha else tab
    fig_w, fig_h = 2.6 + 0.62 * tab.shape[1], 1.6 + 0.36 * tab.shape[0]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(val.values, cmap=SEQ, vmin=0, vmax=100 if pct_por_linha else val.values.max(), aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            n = tab.values[i, j]
            if n:
                v = val.values[i, j]
                escuro = v > (55 if pct_por_linha else val.values.max() * 0.55)
                ax.text(j, i, f"{n}", ha="center", va="center", fontsize=7.5, color="white" if escuro else TINTA)
    ax.set_xticks(range(tab.shape[1]), [f"{codigo(c, IDS_V11)} (N={int(tab[c].sum())})" for c in tab.columns],
                  rotation=90, fontsize=7.5)
    ax.set_yticks(range(tab.shape[0]), rotulo_linhas or [f"{codigo(r, IDS_V10)} (n={int(tab.loc[r].sum())})" for r in tab.index],
                  fontsize=8)
    for t, c in zip(ax.get_xticklabels(), tab.columns):
        t.set_color(["#1f5fa3", "#b3302a", "#555555"][tipo_rotulo(c)] if tab[c].sum() else "#b0b0b0")
    for t, r in zip(ax.get_yticklabels(), tab.index):
        t.set_color(["#1f5fa3", "#b3302a", "#555555"][tipo_rotulo(r)] if tab.loc[r].sum() else "#b0b0b0")
    ax.set_xticks(np.arange(-.5, tab.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, tab.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold", color=TINTA, pad=24)
    ax.text(0, 1.005, rodape, transform=ax.transAxes, fontsize=7.5, color=TINTA2, ha="left", va="bottom")
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rev = pd.read_csv(REVISADO)[["cd_sessao", "primary_label", "experience", "num_events"]].rename(
        columns={"primary_label": "primary_label_revisado", "experience": "experience_revisado"})
    orig = rotulo_v1(V1).reset_index()
    df = rev[["cd_sessao"]].merge(orig.rename(columns={"primary_label": "primary_label_original", "experience": "experience_original"}), on="cd_sessao").merge(rev, on="cd_sessao")
    assert len(df) == 516
    df.to_csv(OUT + "sessoes_rotulo_revisado.csv", sep=";", index=False, encoding="utf-8-sig")

    mig = pd.crosstab(df.primary_label_original, df.primary_label_revisado)
    mig = mig.reindex(index=colunas_completas(list(mig.index), "v1.0"), columns=colunas_completas(list(mig.columns)),
                      fill_value=0)
    mig.to_csv(OUT + "migracao_rotulos.csv", sep=";", encoding="utf-8-sig")
    heatmap(mig, OUT + "migracao_rotulos.png", "Rótulo v1.0 (linhas) × rótulo v1.1 (colunas)",
            "Linhas: v1.0 (10 jornadas) · colunas: v1.1 (12 jornadas; 001+008 = concluiu as duas sem formar composta) · "
            "número = sessões; cor = % da linha; cinza claro = nenhuma sessão.\n"
            f"Mudaram de rótulo: {(df.primary_label_original != df.primary_label_revisado).sum()} de {len(df)} sessões.")

    mets = []
    for m, nome in MODELOS.items():
        cl = pickle.load(open(f"clusters/{m}.pkl", "rb"))
        assert list(cl["session_ids"]) == list(df.cd_sessao), f"ordem das sessões difere em {m}"
        lab = np.asarray(cl["cluster_labels"])
        antes = metricas(lab, df.primary_label_original, df.experience_original)
        depois = metricas(lab, df.primary_label_revisado, df.experience_revisado)
        mets.append({"modelo": m, "nome": nome, "grupos": len(set(lab) - {-1}), "ruido_pct": round((lab == -1).mean() * 100, 1),
                     **{f"{k}_antes": round(v, 4) for k, v in antes.items()},
                     **{f"{k}_depois": round(v, 4) for k, v in depois.items()}})
    met = pd.DataFrame(mets)
    met.to_csv("results/metrics/comparacao_v1.0_v1.1.csv", sep=";", index=False, encoding="utf-8-sig")
    fig_metricas(met)

    print("Sessões que mudaram de rótulo:", (df.primary_label_original != df.primary_label_revisado).sum())
    print("Sessões que mudaram de experience:", (df.experience_original != df.experience_revisado).sum())
    print(df.primary_label_revisado.value_counts().to_string())
    cols = ["nome", "pureza_jornada_antes", "pureza_jornada_depois", "v_measure_antes", "v_measure_depois",
            "ari_clean_antes", "ari_clean_depois", "purity_real_experience_antes", "purity_real_experience_depois"]
    print(met[cols].to_string(index=False))


if __name__ == "__main__":
    main()
