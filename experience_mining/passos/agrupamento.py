"""Passos de agrupamento (K-Means, HDBSCAN) com registro por nome."""
from pathlib import Path

import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from experience_mining.configuracao import ConfigAgrupamentos
from experience_mining.io import (
    carregar_representacao,
    salvar_agrupamento_pkl,
    salvar_agrupamento_portatil,
)


def find_best_k(X, k_min: int = 2, k_max: int = 15, random_state: int = 42):
    print(f"[*] Buscando K otimo via Silhouette Score no intervalo [{k_min}, {k_max}]...")
    best_k = k_min
    best_score = -1.0
    scores = {}

    for k in range(k_min, min(k_max + 1, len(X))):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"    k={k} -> Silhouette: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"[+] Melhor K selecionado: {best_k} (Silhouette = {best_score:.4f})")
    return best_k, scores


def _persistir_agrupamento(output_path: Path | str | None, results: dict, portatil: bool = False) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    if portatil:
        params = {
            k: v
            for k, v in results.items()
            if k
            in (
                "algorithm",
                "k",
                "n_clusters",
                "min_cluster_size",
                "min_samples",
                "metric",
                "noise_ratio",
                "silhouette_score",
                "inertia",
                "vectorizer_type",
            )
        }
        salvar_agrupamento_portatil(out, results["session_ids"], results["cluster_labels"], params=params)
        print(f"[+] Agrupamento portatil salvo em: {out}")
        return

    salvar_agrupamento_pkl(out, results)
    np_path = str(out).replace(".pkl", ".npy") if str(out).endswith(".pkl") else f"{out}.npy"
    np.save(
        np_path,
        {
            "algorithm": results["algorithm"],
            **({"k": results["k"]} if "k" in results else {"n_clusters": results["n_clusters"]}),
            "cluster_labels": results["cluster_labels"],
            **({"noise_ratio": results["noise_ratio"]} if "noise_ratio" in results else {}),
            "silhouette_score": results["silhouette_score"],
            "X": results["X"],
            "labels": results["primary_labels"],
            "experiences": results["experiences"],
        },
    )
    print(f"[+] Resultados do {results['algorithm']} salvos em: {out} e {np_path}")


def agrupar_kmeans(
    data: dict,
    output_path: Path | str | None = None,
    n_clusters: int = 7,
    auto_k: bool = False,
    random_state: int = 42,
    seed: int | None = None,
    portatil: bool = False,
    **_,
) -> dict:
    rs = seed if seed is not None else random_state
    X = data["X"]
    k_val = find_best_k(X, k_min=2, k_max=15, random_state=rs)[0] if auto_k else n_clusters

    print(f"[*] Executando KMeans (k={k_val}, random_state={rs}, init=k-means++)...")
    kmeans = KMeans(n_clusters=k_val, init="k-means++", random_state=rs, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    sil_score = None
    if len(set(cluster_labels)) > 1:
        sil_score = float(silhouette_score(X, cluster_labels))
        print(f"[+] Silhouette Score: {sil_score:.4f}")

    primary_labels = data.get("primary_labels")
    if primary_labels is None:
        primary_labels = data.get("labels")

    results = {
        "algorithm": "K-Means",
        "k": k_val,
        "cluster_labels": cluster_labels,
        "cluster_centers": kmeans.cluster_centers_,
        "inertia": float(kmeans.inertia_),
        "silhouette_score": sil_score,
        "X": X,
        "session_ids": data.get("session_ids"),
        "primary_labels": primary_labels,
        "experiences": data.get("experiences"),
        "vectorizer_type": data.get("vectorizer_type", "unknown"),
    }
    _persistir_agrupamento(output_path, results, portatil=portatil)
    return results


def grid_search_hdbscan(X, min_cluster_sizes=(5, 10, 15, 20), metric: str = "euclidean"):
    print(f"[*] Grid search HDBSCAN sobre min_cluster_size in {min_cluster_sizes} (metric={metric})...")
    best_params = None
    best_score = -1.0
    best_clusterer = None

    for mcs in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric=metric, core_dist_n_jobs=-1)
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.mean(labels == -1)

        valid_mask = labels != -1
        if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
            score = silhouette_score(X[valid_mask], labels[valid_mask])
        else:
            score = -1.0

        print(
            f"    min_cluster_size={mcs} -> Clusters: {n_clusters}, "
            f"Ruido: {noise_ratio*100:.1f}%, Silhouette (sem ruido): {score:.4f}"
        )
        effective_score = score * (1.0 - 0.5 * noise_ratio)
        if effective_score > best_score:
            best_score = effective_score
            best_params = {"min_cluster_size": mcs, "metric": metric}
            best_clusterer = clusterer

    print(f"[+] Melhores parametros: {best_params}")
    return best_clusterer, best_params


def agrupar_hdbscan(
    data: dict,
    output_path: Path | str | None = None,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "euclidean",
    auto_grid: bool = False,
    portatil: bool = False,
    **_,
) -> dict:
    X = data["X"]
    if auto_grid:
        clusterer, params = grid_search_hdbscan(X, min_cluster_sizes=[5, 10, 15, 20], metric=metric)
        min_cluster_size = params["min_cluster_size"]
    else:
        print(f"[*] Executando HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric={metric})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            core_dist_n_jobs=-1,
        )
        clusterer.fit(X)

    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))
    noise_ratio = float(np.mean(labels == -1))

    valid_mask = labels != -1
    sil_score = None
    if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
        sil_score = float(silhouette_score(X[valid_mask], labels[valid_mask]))

    print(f"[+] Clusters formados: {n_clusters} | Pontos de ruido: {noise_count} ({noise_ratio*100:.1f}%)")
    if sil_score is not None:
        print(f"[+] Silhouette Score (excluindo ruido): {sil_score:.4f}")

    primary_labels = data.get("primary_labels")
    if primary_labels is None:
        primary_labels = data.get("labels")

    results = {
        "algorithm": "HDBSCAN",
        "n_clusters": n_clusters,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
        "cluster_labels": labels,
        "probabilities": clusterer.probabilities_,
        "noise_ratio": noise_ratio,
        "silhouette_score": sil_score,
        "X": X,
        "session_ids": data.get("session_ids"),
        "primary_labels": primary_labels,
        "experiences": data.get("experiences"),
        "vectorizer_type": data.get("vectorizer_type", "unknown"),
    }
    _persistir_agrupamento(output_path, results, portatil=portatil)
    return results


AGRUPADORES = {
    "kmeans": agrupar_kmeans,
    "hdbscan": agrupar_hdbscan,
}


def executar_agrupamentos(secao: ConfigAgrupamentos, representacoes: dict[str, Path]) -> None:
    """Executa os agrupadores registrados sobre cada representação disponível."""
    if secao.itens:
        gerar_itens = [item for item in secao.itens.values() if item.modo == "gerar"]
        if not gerar_itens:
            return
        for nome_rep, caminho_rep in representacoes.items():
            if not Path(caminho_rep).exists():
                continue
            dados = carregar_representacao(caminho_rep)
            for item in gerar_itens:
                destino = item.caminhos_por_rep[nome_rep]
                AGRUPADORES[item.algoritmo](
                    dados,
                    output_path=destino,
                    portatil=True,
                    **item.params,
                )
        return

    if not secao.gerar:
        return
    for nome_rep, caminho_rep in representacoes.items():
        if not Path(caminho_rep).exists():
            continue
        dados = carregar_representacao(caminho_rep)
        AGRUPADORES["kmeans"](
            dados,
            output_path=secao.clusters_dir / f"kmeans_{nome_rep}.pkl",
            n_clusters=secao.n_clusters,
            auto_k=secao.auto_k,
            random_state=secao.kmeans_seed,
        )
        AGRUPADORES["hdbscan"](
            dados,
            output_path=secao.clusters_dir / f"hdbscan_{nome_rep}.pkl",
            min_cluster_size=secao.min_cluster_size,
            min_samples=secao.min_samples,
            metric=secao.hdbscan_metric,
            auto_grid=secao.auto_grid,
        )
