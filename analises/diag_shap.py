"""Diagnóstico do XAI: por que o SHAP concentra tudo num único evento por cluster?"""
import pickle, sys
import numpy as np
import xgboost as xgb
import shap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold

root = sys.argv[1]
tf = pickle.load(open(f"{root}/data/tfidf_vectors.pkl", "rb"))
cl = pickle.load(open(f"{root}/clusters/hdbscan_tf_idf.pkl", "rb"))
X_tfidf, names = tf["X"], np.array(tf["feature_names"])
labels = np.array(cl["cluster_labels"])
texts = tf["df"]["text_representation"].fillna("").astype(str).tolist()

cv = CountVectorizer(lowercase=False, token_pattern=r"\S+", vocabulary=list(names))
X_count = cv.transform(texts).toarray().astype(np.float32)
X_bin = (X_count > 0).astype(np.float32)

m = labels != -1
y_raw = labels[m]
classes = sorted(set(y_raw)); y = np.array([classes.index(c) for c in y_raw])
print(f"sessões válidas={m.sum()}  ruído={(~m).sum()}  clusters={len(classes)}  features={len(names)}")

# 1) idf é constante por coluna -> árvores são invariantes a isso. O que varia é tf e a norma L2 da linha.
idf_only = X_count[m] * tf.get("idf", 1)
print("\n[1] Correlação (por feature, dentro das sessões que contêm o evento) entre TF-IDF e contagem bruta:")
cors = []
for j in range(len(names)):
    nz = X_count[m][:, j] > 0
    if nz.sum() > 5 and X_count[m][nz, j].std() > 0:
        cors.append(np.corrcoef(X_count[m][nz, j], X_tfidf[m][nz, j])[0, 1])
print(f"    mediana={np.median(cors):.2f}  (abaixo de 1 = a normalização L2 pelo tamanho da sessão distorce o valor)")

# 2) Redundância: quantos eventos têm EXATAMENTE o mesmo padrão de presença por sessão
pat = {}
for j in range(len(names)):
    pat.setdefault(X_bin[m][:, j].tobytes(), []).append(j)
groups = [g for g in pat.values() if len(g) > 1]
print(f"\n[2] Grupos de eventos com presença idêntica em todas as sessões: {len(groups)} grupos, "
      f"{sum(len(g) for g in groups)} eventos")
for g in sorted(groups, key=len, reverse=True)[:4]:
    print("    ", list(names[g])[:4], "..." if len(g) > 4 else "")

# 3) Surrogate: acurácia de treino vs validação cruzada, e concentração do SHAP
def run(X, tag):
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
                            eval_metric="mlogloss")
    cvacc = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    clf.fit(X, y)
    # TreeSHAP exato nativo do xgboost: (n, classes, features+bias) -> (n, features, classes)
    sv = clf.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :, :-1].transpose(0, 2, 1)
    used = (np.abs(sv).sum((0, 2)) > 0).sum()
    share = []
    for k in range(len(classes)):
        ms = sv[y == k, :, k].mean(0)
        pos = np.sort(ms[ms > 0])[::-1]
        share.append(pos[0] / pos.sum() if pos.size else 0)
    print(f"    {tag:8s} treino={clf.score(X, y):.3f}  CV={cvacc:.3f}  features c/ SHAP≠0={used}/{len(names)}  "
          f"peso do 1º evento na assinatura={np.mean(share):.0%}")
    return clf, sv

print("\n[3] Surrogate XGBoost:")
run(X_tfidf[m], "TF-IDF")
run(X_count[m], "contagem")
run(X_bin[m], "binário")

# 4) Alternativa descritiva (sem modelo): presença no cluster vs fora, ordenada por diferença
print("\n[4] Assinatura descritiva (presença dentro vs fora), 3 clusters de exemplo:")
for c in [3, 5, 13]:
    k = classes.index(c)
    inn = X_bin[m][y == k].mean(0); out = X_bin[m][y != k].mean(0)
    d = inn - out
    top = np.argsort(d)[::-1][:5]
    print(f"  Cluster {c} (N={(y == k).sum()}):")
    for j in top:
        print(f"     {names[j]:55s} dentro={inn[j]:.0%}  fora={out[j]:.0%}  "
              f"contagem média dentro={X_count[m][y == k, j].mean():.1f}")
