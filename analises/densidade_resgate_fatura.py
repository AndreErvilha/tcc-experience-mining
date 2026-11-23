"""As 18 sessões 'resgatou E pagou' viraram ruído/mistura por falta de densidade (poucas sessões)
ou porque a representação as coloca junto de 'só resgatou' / 'só pagou'?

1) Vizinhos: para cada sessão resgate+fatura, de que categoria são os k vizinhos mais próximos
   (mesma métrica euclidiana do HDBSCAN do projeto).
2) Distâncias: distância média dentro do grupo × até 'só resgatou' × até 'só pagou'.
3) HDBSCAN restrito: roda só sobre as sessões de resgate e/ou pagamento, com min_cluster_size
   pequeno, para ver se o grupo aparece quando a densidade "concorrente" some.

Uso (da raiz do repositório): python analises/densidade_resgate_fatura.py .
"""
import pickle, sys
from collections import Counter
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances

root = sys.argv[1] if len(sys.argv) > 1 else "."
tf = pickle.load(open(f"{root}/data/tfidf_vectors.pkl", "rb"))
df = tf["df"]
resg = df["is_cdb_resgatado"].values == 1
pag = df["is_fatura_paga_saldo"].values == 1
cat = np.where(resg & pag, "RF", np.where(resg, "R", np.where(pag, "P", "outra")))
print("categorias:", dict(Counter(cat)), " (RF = resgatou e pagou, R = só resgatou, P = só pagou)")
print("eventos por sessão (média):", {c: round(df.loc[cat == c, "num_events"].mean(), 1) for c in ["RF", "R", "P"]})

spaces = {"TF-IDF": tf["X"]}
for name, path in [("MiniLM", "minilm_embeddings.pkl"), ("Gemini", "gemini_embeddings.pkl")]:
    d = pickle.load(open(f"{root}/data/{path}", "rb"))
    spaces[name] = np.asarray(d["X"] if isinstance(d, dict) else d, dtype=np.float32)

rf = np.where(cat == "RF")[0]
K = 5
for name, X in spaces.items():
    D = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D[rf], axis=1)[:, :K]
    vc = Counter(cat[nn].ravel())
    tot = sum(vc.values())
    within = D[np.ix_(rf, rf)][np.isfinite(D[np.ix_(rf, rf)])].mean()
    to_r = D[np.ix_(rf, np.where(cat == "R")[0])].mean()
    to_p = D[np.ix_(rf, np.where(cat == "P")[0])].mean()
    print(f"\n=== {name} ===")
    print(f"  {K} vizinhos mais próximos das 18 RF: " +
          ", ".join(f"{k}={v / tot:.0%}" for k, v in vc.most_common()))
    print(f"  distância média: dentro de RF={within:.3f} | RF→R={to_r:.3f} | RF→P={to_p:.3f}")

    # densidade local por grupo: distância ao 5º vizinho (menor = mais denso), como no HDBSCAN
    core = np.sort(D, axis=1)[:, 4]
    prim = np.array(tf["primary_labels"], dtype=object)
    grupos = {"RF": cat == "RF", "R": cat == "R", "P": cat == "P"}
    for lab in ["pix_transferido", "cdb_aplicado", "fatura_parcelada", "openfinance_pago"]:
        grupos[lab] = prim == lab
    print("  densidade (mediana da dist. ao 5º vizinho): " +
          ", ".join(f"{g}={np.median(core[m]):.3f}" for g, m in grupos.items()))

    sub = np.where((cat == "RF") | (cat == "R") | (cat == "P"))[0]
    for mcs in (5, 10):
        lab = HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(X[sub])
        best = None
        for k in set(lab) - {-1}:
            ink = lab == k
            n_rf = np.sum(ink & (cat[sub] == "RF"))
            if n_rf and (best is None or n_rf / ink.sum() > best[0]):
                best = (n_rf / ink.sum(), n_rf, ink.sum(), k)
        rf_noise = np.sum((lab == -1) & (cat[sub] == "RF"))
        txt = (f"melhor cluster p/ RF: {best[1]} RF de {best[2]} ({best[0]:.0%})" if best
               else "nenhum cluster contém RF")
        print(f"  HDBSCAN só em R+P+RF, min_cluster_size={mcs}: {len(set(lab) - {-1})} clusters; "
              f"{txt}; RF no ruído={rf_noise}/18")
