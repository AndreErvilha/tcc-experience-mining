"""Pesos SHAP de cada grupo, nos 6 modelos (passo 5 do run_all.py).

Modelo substituto: XGBoost multiclasse sobre a matriz TF-IDF (eventos), sem as sessões de ruído
(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42; o mesmo do TCC). Os grupos do
MiniLM e do Gemini também são explicados pelos eventos (projeção na matriz TF-IDF).

Saídas, em results/jornadas_por_grupo/:
  <modelo>/grupos/grupo_NN/shap_pesos.csv   pesos SHAP dos eventos do grupo (SHAP != 0), do mais forte ao mais fraco
  shap_pesos_todos_modelos.csv              todos os grupos dos 6 modelos num arquivo só
  shap_fidelidade_treino.csv                acurácia de treino do substituto de cada modelo

Até a v1.0, este passo explicava só o HDBSCAN + TF-IDF e gravava results/xai_shap/shap_cluster_signatures.json
(5 eventos que puxam e 3 que afastam por grupo). Os mesmos valores estão agora em
results/jornadas_por_grupo/hdbscan_tf_idf/grupos/grupo_NN/shap_pesos.csv.
Como ler: docs/leitura-shap.md. Figuras (beeswarms, Figuras 4 e 5): src/shap_figures.py.
Uso (da raiz do repositório): python src/shap_explainer.py
"""
import os
import pickle

import numpy as np
import pandas as pd

from shap_figures import MODELOS, treinar_surrogate, valores_shap

OUT = "results/jornadas_por_grupo/"
CSV = dict(sep=";", index=False, encoding="utf-8-sig")


def pesos_do_grupo(sv, Xv, y, i, eventos):
    """Uma linha por evento com SHAP != 0 para a classe i, ordenada pela magnitude no grupo."""
    dentro = y == i
    s_in, s_out = sv[dentro, :, i], sv[~dentro, :, i]
    linhas = []
    for j, ev in enumerate(eventos):
        if not np.any(sv[:, j, i] != 0):
            continue
        med = float(s_in[:, j].mean())
        linhas.append({"evento": ev, "shap_medio_no_grupo": round(med, 5),
                       "shap_abs_medio_no_grupo": round(float(np.abs(s_in[:, j]).mean()), 5),
                       "shap_medio_fora_do_grupo": round(float(s_out[:, j].mean()), 5) if len(s_out) else 0.0,
                       "pct_sessoes_do_grupo_com_evento": round(float((Xv[dentro, j] > 0).mean()), 3),
                       "pct_sessoes_fora_com_evento": round(float((Xv[~dentro, j] > 0).mean()), 3) if len(s_out) else 0.0,
                       "tfidf_medio_no_grupo": round(float(Xv[dentro, j].mean()), 5),
                       "direcao": "empurra PARA o grupo (+)" if med >= 0 else "afasta do grupo (-)"})
    return pd.DataFrame(linhas).sort_values("shap_abs_medio_no_grupo", ascending=False)


def main():
    tf = pickle.load(open("data/tfidf_vectors.pkl", "rb"))
    X = tf["X"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    eventos = list(tf["feature_names"])
    todos, fid = [], []
    for m in MODELOS:
        lab = pickle.load(open(f"clusters/{m}.pkl", "rb"))["cluster_labels"]
        clf, Xv, y, classes = treinar_surrogate(X, lab)
        sv = valores_shap(clf, Xv)
        fid.append({"modelo": m, "acuracia_treino_pct": round(clf.score(Xv, y) * 100, 2), "sessoes_sem_ruido": len(y)})
        print(f"\n[{m}] acurácia de treino do substituto: {fid[-1]['acuracia_treino_pct']}%")
        for i, g in enumerate(classes):
            p = pesos_do_grupo(sv, Xv, y, i, eventos)
            p.insert(0, "grupo", g)
            p.insert(0, "modelo", m)
            pasta = os.path.join(OUT, m, "grupos", f"grupo_{g:02d}")
            os.makedirs(pasta, exist_ok=True)
            p.to_csv(os.path.join(pasta, "shap_pesos.csv"), **CSV)
            todos.append(p)
            topo = p[p.shap_medio_no_grupo > 0].head(3)
            print(f"  grupo {g:2d} (n={int((y == i).sum())}): " + ", ".join(f"{e} ({v:+.2f})" for e, v in zip(topo.evento, topo.shap_medio_no_grupo)))
    pd.concat(todos).to_csv(OUT + "shap_pesos_todos_modelos.csv", **CSV)
    pd.DataFrame(fid).to_csv(OUT + "shap_fidelidade_treino.csv", **CSV)
    print(f"\n[+] Pesos SHAP por grupo em {OUT}<modelo>/grupos/grupo_NN/shap_pesos.csv e {OUT}shap_pesos_todos_modelos.csv")


if __name__ == "__main__":
    main()
