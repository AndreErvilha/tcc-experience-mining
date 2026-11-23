"""Cálculo puro das métricas de agrupamento, contagem e cobertura por jornada."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)

from experience_mining.nucleo.jornadas import JORNADAS, MODELOS


def calcular_pureza_experiencia(cluster_labels, experiences):
    if experiences is None or len(experiences) == 0:
        return None, {}

    unique_clusters = [c for c in set(cluster_labels) if c != -1]
    if not unique_clusters:
        return None, {}

    cluster_purities = {}
    total_valid_samples = 0
    weighted_purity_sum = 0.0

    for c in unique_clusters:
        mask = cluster_labels == c
        c_exps = [experiences[i] for i in range(len(experiences)) if mask[i]]
        valid_exps = [e for e in c_exps if e != "none"]
        if not valid_exps:
            continue

        n_success = sum(1 for e in valid_exps if e == "success")
        n_negative = sum(1 for e in valid_exps if e in ["attrition", "error"])
        total_c = n_success + n_negative
        if total_c == 0:
            continue

        purity_c = max(n_success, n_negative) / total_c
        cluster_purities[int(c)] = {
            "size": total_c,
            "success": n_success,
            "negative": n_negative,
            "purity": float(purity_c),
        }
        weighted_purity_sum += purity_c * total_c
        total_valid_samples += total_c

    global_purity = (weighted_purity_sum / total_valid_samples) if total_valid_samples > 0 else None
    return global_purity, cluster_purities


def avaliar_agrupamento(cluster_data: dict) -> dict:
    cluster_labels = np.array(cluster_data["cluster_labels"])
    X = cluster_data.get("X")
    primary_labels = cluster_data.get("primary_labels")
    if primary_labels is None:
        primary_labels = cluster_data.get("labels")
    experiences = cluster_data.get("experiences")
    algorithm = cluster_data.get("algorithm", "Unknown")
    vectorizer = cluster_data.get("vectorizer_type", "Unknown")

    n_total = len(cluster_labels)
    n_noise = int(np.sum(cluster_labels == -1))
    noise_ratio = float(n_noise / n_total)
    k_discovered = len(set(cluster_labels) - {-1})

    metrics = {
        "algorithm": algorithm,
        "vectorizer": vectorizer,
        "n_samples": n_total,
        "k_discovered": k_discovered,
        "noise_count": n_noise,
        "noise_ratio": noise_ratio,
    }

    if X is not None and k_discovered >= 2:
        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) > k_discovered:
            metrics["silhouette_score"] = float(silhouette_score(X[valid_mask], cluster_labels[valid_mask]))
        else:
            metrics["silhouette_score"] = None
    else:
        metrics["silhouette_score"] = None

    if primary_labels is not None:
        primary_labels = np.array(primary_labels)
        metrics["ari_all"] = float(adjusted_rand_score(primary_labels, cluster_labels))
        metrics["nmi_all"] = float(normalized_mutual_info_score(primary_labels, cluster_labels))

        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) > 1 and len(set(primary_labels[valid_mask])) > 1:
            metrics["ari_clean"] = float(adjusted_rand_score(primary_labels[valid_mask], cluster_labels[valid_mask]))
            metrics["nmi_clean"] = float(normalized_mutual_info_score(primary_labels[valid_mask], cluster_labels[valid_mask]))
            h, c, v = homogeneity_completeness_v_measure(primary_labels[valid_mask], cluster_labels[valid_mask])
            metrics["homogeneity"] = float(h)
            metrics["completeness"] = float(c)
            metrics["v_measure"] = float(v)
        else:
            metrics["ari_clean"] = None
            metrics["nmi_clean"] = None
    else:
        metrics["ari_all"] = None
        metrics["nmi_all"] = None

    if experiences is not None:
        global_purity, c_details = calcular_pureza_experiencia(cluster_labels, experiences)
        metrics["purity_real_experience"] = global_purity
        metrics["cluster_purity_details"] = c_details
    else:
        metrics["purity_real_experience"] = None

    return metrics


def calcular_contagem_jornadas(sessions_df: pd.DataFrame, jornadas=JORNADAS) -> pd.DataFrame:
    cen = ((sessions_df.cd_sessao.str[5:].astype(int) - 1) // 100 + 1).map(lambda c: f"{c:03d}")
    linhas = [
        {
            "jornada": j.id,
            "nome": j.nome,
            "cenario_origem": j.cenario_origem,
            "sessoes_do_cenario": int((cen == j.cenario_origem).sum()),
            "concluiram": int(sessions_df[f"is_{j.flag_sucesso}"].sum()),
            "nao_concluiram": int(sessions_df[f"is_{j.flag_abandono}"].sum()),
            "rotulo_principal": int((sessions_df.primary_label == j.flag_sucesso).sum()),
        }
        for j in jornadas
    ]
    return pd.DataFrame(linhas)


def calcular_cobertura_jornadas(
    sessions_df: pd.DataFrame,
    clusters_por_modelo: dict[str, dict],
    modelos: list[str] = MODELOS,
    limiar: float = 0.8,
) -> pd.DataFrame:
    y = sessions_df.set_index("cd_sessao").primary_label
    cats = [c for c, n in y.value_counts().items() if n >= 10]
    cob = []
    for c in cats:
        r = {"rotulo": c, "sessoes": int((y == c).sum())}
        for m in modelos:
            cl = clusters_por_modelo[m]
            lab = np.asarray(cl["cluster_labels"])
            inj = (y.loc[list(cl["session_ids"])] == c).values
            puros = [k for k in set(lab) - {-1} if np.mean(inj[lab == k]) >= limiar]
            r[m] = round(float(np.sum(inj & np.isin(lab, puros)) / inj.sum() * 100), 1)
        cob.append(r)
    cob_df = pd.DataFrame(cob)
    total = {
        "rotulo": "jornadas_com_cobertura_>=80%",
        "sessoes": len(cats),
        **{m: int((cob_df[m] >= 80).sum()) for m in modelos},
    }
    return pd.concat([cob_df, pd.DataFrame([total])])


def consolidar_sumario(itens: list[tuple[str, dict]]) -> pd.DataFrame:
    rows = []
    for exp_name, data in itens:
        silh = data.get("silhouette_score")
        ari_clean = data.get("ari_clean")
        v_meas = data.get("v_measure")
        purity = data.get("purity_real_experience")
        noise = data.get("noise_ratio", 0.0)
        rows.append(
            {
                "Experimento": exp_name,
                "Vetorizador": data.get("vectorizer", "-"),
                "Algoritmo": data.get("algorithm", "-"),
                "K": data.get("k_discovered", "-"),
                "Ruido (%)": f"{noise*100:.1f}%",
                "Silhouette": f"{silh:.4f}" if silh is not None else "-",
                "ARI (s/ ruido)": f"{ari_clean:.4f}" if ari_clean is not None else "-",
                "V-Measure": f"{v_meas:.4f}" if v_meas is not None else "-",
                "Pureza Real": f"{purity*100:.2f}%" if purity is not None else "-",
            }
        )
    return pd.DataFrame(rows)
