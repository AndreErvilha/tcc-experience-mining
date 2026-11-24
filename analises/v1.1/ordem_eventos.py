"""A ordem dos eventos importa nesta massa? (hipótese do "omelete", v1.1)

Ideia: num app, a tela de comprovante só aparece depois das anteriores; então a
contagem de eventos (o que o TF-IDF vê) já carregaria quase toda a ordem.

Testes, sem chamar nenhuma API:
1. Quantas sequências distintas existem e quantas continuam distintas quando
   se ignora a ordem (multiconjunto de eventos = o que o TF-IDF vê).
2. Nos pares com os mesmos eventos em ordens diferentes: têm o mesmo rótulo?
   O Gemini os separa (similaridade de cosseno e grupo)?
3. Linha de base: cosseno típico do vizinho mais próximo no Gemini.

Entradas: analises/v1.1/saidas/sessoes_por_modelo.csv, data/v1.0/gemini_embeddings.pkl
Saída: analises/v1.1/saidas/ordem_eventos_pares.csv e resumo impresso.
Limite: mostra se a ordem acrescenta informação NESTA massa; não testa o caso
geral. Um teste causal seria embaralhar a ordem e reembeddar (exige a API).
"""
from pathlib import Path
from collections import Counter
import pickle
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]


def main():
    s = pd.read_csv(AQUI / "saidas" / "sessoes_por_modelo.csv", sep=";").set_index("cd_sessao")
    g = pickle.load(open(RAIZ / "data" / "v1.0" / "gemini_embeddings.pkl", "rb"))
    X = np.asarray(g["X"], dtype=float)
    ids = list(g["session_ids"])
    s = s.loc[ids].copy()
    s["multiconjunto"] = s.text_representation.str.split().apply(lambda q: tuple(sorted(Counter(q).items())))

    n_seq = s.text_representation.nunique()
    n_bag = s.multiconjunto.nunique()
    print(f"Sessões: {len(s)} | sequências distintas: {n_seq} | distintas sem a ordem: {n_bag}")
    print(f"Sequências que só se distinguem pela ordem: {n_seq - n_bag} ({(n_seq - n_bag) / n_seq:.1%})")

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    pos = {k: i for i, k in enumerate(ids)}
    C = Xn @ Xn.T
    np.fill_diagonal(C, np.nan)
    print(f"Gemini: cosseno mediano do vizinho mais próximo = {np.nanmedian(np.nanmax(C, axis=1)):.4f}")

    pares = []
    for _, d in s.reset_index().groupby("multiconjunto"):
        r = d.drop_duplicates("text_representation")
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                a, b = r.iloc[i], r.iloc[j]
                pares.append({
                    "sessao_a": a.cd_sessao, "sessao_b": b.cd_sessao,
                    "rotulo_a": a.primary_label, "rotulo_b": b.primary_label,
                    "num_events": a.num_events,
                    "cos_gemini": round(float(C[pos[a.cd_sessao], pos[b.cd_sessao]]), 4),
                    "mesmo_grupo_hdbscan_gemini": a.hdbscan_gemini == b.hdbscan_gemini,
                    "mesmo_grupo_kmeans_gemini": a.kmeans_gemini == b.kmeans_gemini,
                })
    p = pd.DataFrame(pares)
    p.to_csv(AQUI / "saidas" / "ordem_eventos_pares.csv", sep=";", index=False)
    print(f"Pares mesmos eventos / ordem diferente: {len(p)}")
    print(f"  com o mesmo rótulo: {(p.rotulo_a == p.rotulo_b).mean():.0%}")
    print(f"  cosseno Gemini: {p.cos_gemini.min():.4f} a {p.cos_gemini.max():.4f}")
    print(f"  mesmo grupo HDBSCAN+Gemini: {p.mesmo_grupo_hdbscan_gemini.mean():.0%}"
          f" | K-Means+Gemini: {p.mesmo_grupo_kmeans_gemini.mean():.0%}")


if __name__ == "__main__":
    main()
