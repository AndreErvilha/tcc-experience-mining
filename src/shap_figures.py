# shap_figures.py
# Gera as figuras de explicabilidade (SHAP) do TCC a partir dos artefatos do repositório.
#
#   Figura 4 do TCC  -> results/figures/shap_summary_bar_plot_global.png
#   Figura 5 do TCC  -> results/figures/shap_beeswarm_panel_2x2.png
#   Beeswarm de cada grupo, nos 6 modelos -> results/jornadas_por_grupo/<modelo>/grupos/grupo_NN/beeswarm.png
#       (grupos do MiniLM e do Gemini explicados pelos eventos: projeção na matriz TF-IDF; os pesos em
#       números saem do src/shap_explainer.py)
#   Sensibilidade colsample_bytree = 0,3 no Cluster 5 (slide de apoio A7)
#       -> results/dados/colsample_cluster5.json
#
# O modelo substituto é o mesmo de src/shap_explainer.py (XGBoost multiclasse sobre a matriz
# TF-IDF, sem as sessões de ruído, random_state=42). Os valores de assinatura reproduzem
# os pesos do HDBSCAN + TF-IDF em results/jornadas_por_grupo/hdbscan_tf_idf/grupos/grupo_NN/shap_pesos.csv.
#
# Observação (errata pós-entrega): os títulos da Figura 5 são os do TCC entregue. O Cluster 5
# é o grupo de resgate de CDB, não a jornada "Resgate CDB -> Fatura". Use --titulos-revisados
# para gerar a versão com os títulos corrigidos.
#
# Uso (da raiz do repositório):
#   python src/shap_figures.py
#   python src/shap_figures.py --titulos-revisados
import os
import json
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

PREFIXOS = {"cartoes": "Cartoes", "investimentos": "Invest", "pix": "PIX", "canais": "Home",
            "openfinance": "OpenFinance", "open_finance": "OpenFinance"}

TITULOS_TCC = {
    5: "Cluster 5: Padrão Composto (Resgate CDB -> Fatura)",
    3: "Cluster 3: Atrito em Histórico de Fatura (Abandono)",
    10: "Cluster 10: Pagamento de Fatura com Saldo",
    0: "Cluster 0: Transferência PIX por Chave Direta",
}
TITULOS_REVISADOS = {
    5: "Cluster 5: Resgate de CDB",
    3: "Cluster 3: Atrito em Histórico de Fatura (Abandono)",
    10: "Cluster 10: Pagamento de Fatura com Saldo",
    0: "Cluster 0: Transferência PIX por Chave Direta",
}
EIXO = "Impacto SHAP (empurra para dentro ou fora do cluster)"


def nome_legivel(token):
    """'investimentos:click_carteira_renda_fixa_detalhar_cdb' -> 'Invest_click_carteira_renda_fixa_detalhar_cdb'."""
    if ":" not in token:
        return token
    dominio, acao = token.split(":", 1)
    acao = acao.replace("view_", "tela_", 1) if acao.startswith("view_") else acao
    return f"{PREFIXOS.get(dominio, dominio.capitalize())}_{acao}"


def treinar_surrogate(X, labels, **xgb_kwargs):
    """Treina o XGBoost substituto sem as sessões de ruído e devolve (modelo, X, y_mapeado, classes)."""
    labels = np.asarray(labels)
    m = labels != -1
    Xv, yv = X[m], labels[m]
    classes = sorted(set(yv.tolist()))
    mapa = {c: i for i, c in enumerate(classes)}
    y = np.array([mapa[c] for c in yv])
    params = dict(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
    params.update(xgb_kwargs)
    clf = xgb.XGBClassifier(**params).fit(Xv, y)
    return clf, Xv, y, classes


def valores_shap(clf, Xv):
    sv = shap.TreeExplainer(clf).shap_values(Xv)
    if isinstance(sv, list):  # versões antigas do shap devolvem uma lista por classe
        sv = np.stack(sv, axis=2)
    return sv  # (amostras, features, classes)


def beeswarm(ax_sv, Xv, nomes, titulo, caminho=None, ax=None, max_display=7):
    plt.sca(ax) if ax is not None else plt.figure(figsize=(10, 6))
    shap.summary_plot(ax_sv, Xv, feature_names=nomes, max_display=max_display, show=False,
                      plot_size=None if ax is not None else (10, 6))
    plt.title(titulo, fontsize=12, fontweight="bold")
    plt.xlabel(EIXO, fontsize=9, fontweight="bold")
    if caminho:
        plt.tight_layout()
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.close()


MODELOS = ["kmeans_tf_idf", "kmeans_minilm", "kmeans_gemini", "hdbscan_tf_idf", "hdbscan_minilm", "hdbscan_gemini"]
NOMES_MODELOS = {"kmeans_tf_idf": "K-Means + TF-IDF", "kmeans_minilm": "K-Means + MiniLM",
                 "kmeans_gemini": "K-Means + Gemini", "hdbscan_tf_idf": "HDBSCAN + TF-IDF",
                 "hdbscan_minilm": "HDBSCAN + MiniLM", "hdbscan_gemini": "HDBSCAN + Gemini"}
JPG = "results/jornadas_por_grupo/"


def beeswarms_por_grupo(X, eventos):
    """Beeswarm de cada grupo, nos 6 modelos (grupos/grupo_NN/beeswarm.png). Os pesos em números saem do
    src/shap_explainer.py (grupos/grupo_NN/shap_pesos.csv). O ruído do HDBSCAN fica fora do substituto."""
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    for m in MODELOS:
        lab = pickle.load(open(f"clusters/{m}.pkl", "rb"))["cluster_labels"]
        clf, Xv, y, classes = treinar_surrogate(X, lab)
        sv = valores_shap(clf, Xv)
        for i, g in enumerate(classes):
            pasta = os.path.join(JPG, m, "grupos", f"grupo_{g:02d}")
            os.makedirs(pasta, exist_ok=True)
            beeswarm(sv[:, :, i], Xv, list(eventos), f"{NOMES_MODELOS[m]} · Cluster {g} (n={int((y == i).sum())})",
                     caminho=os.path.join(pasta, "beeswarm.png"))
        print(f"[+] {m}: {len(classes)} beeswarms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--titulos-revisados", action="store_true")
    ap.add_argument("--sem-individuais", action="store_true", help="Não gera os beeswarms por grupo.")
    args = ap.parse_args()

    os.makedirs("results/figures", exist_ok=True)
    tf = pickle.load(open("data/tfidf_vectors.pkl", "rb"))
    X = tf["X"]
    nomes = [nome_legivel(t) for t in tf["feature_names"]]

    # Modelo principal: HDBSCAN + TF-IDF
    cl = pickle.load(open("clusters/hdbscan_tf_idf.pkl", "rb"))["cluster_labels"]
    clf, Xv, y, classes = treinar_surrogate(X, cl)
    sv = valores_shap(clf, Xv)
    print(f"[+] Surrogate HDBSCAN + TF-IDF: acurácia de treino {clf.score(Xv, y) * 100:.2f}%")

    # Figura 4: importância global (média |SHAP| por classe, barras empilhadas)
    plt.figure(figsize=(12, 8))
    shap.summary_plot([sv[:, :, i] for i in range(sv.shape[2])], Xv, feature_names=nomes,
                      class_names=[f"Cluster {c}" for c in classes], plot_type="bar",
                      max_display=15, show=False, plot_size=None)
    plt.title("Importância global dos eventos na predição dos agrupamentos (SHAP)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/shap_summary_bar_plot_global.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figura 5: painel 2x2
    titulos = TITULOS_REVISADOS if args.titulos_revisados else TITULOS_TCC
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    for ax, c in zip(axes.ravel(), [5, 3, 10, 0]):
        i = classes.index(c)
        beeswarm(sv[:, :, i], Xv, nomes, titulos[c], ax=ax)
    plt.tight_layout()
    plt.savefig("results/figures/shap_beeswarm_panel_2x2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Figuras 4 e 5 salvas em results/figures/")

    if not args.sem_individuais:
        beeswarms_por_grupo(X, tf["feature_names"])

    # Sensibilidade: amostragem de atributos (colsample_bytree = 0,3) no Cluster 5
    c4, X4, y4, cls4 = treinar_surrogate(X, cl, colsample_bytree=0.3)
    s4 = valores_shap(c4, X4)
    i5 = cls4.index(5)
    media = s4[y4 == i5, :, i5].mean(axis=0)
    top = np.argsort(media)[::-1][:5]
    res = {"cluster": 5, "colsample_bytree": 0.3,
           "top_eventos": [{"evento": tf["feature_names"][j], "shap_medio": round(float(media[j]), 4)} for j in top]}
    os.makedirs("results/dados", exist_ok=True)
    json.dump(res, open("results/dados/colsample_cluster5.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("[+] Sensibilidade colsample_bytree = 0,3 (Cluster 5):")
    for t in res["top_eventos"]:
        print(f"    {t['evento']}: {t['shap_medio']:+.2f}")


if __name__ == "__main__":
    main()
