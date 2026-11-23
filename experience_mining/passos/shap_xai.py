"""Passos de explicabilidade SHAP sobre modelo substituto XGBoost."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from experience_mining.configuracao import ConfigAgrupamentos, ConfigFiguras, ConfigShap
from experience_mining.io import (
    carregar_agrupamento,
    carregar_representacao,
    salvar_csv,
    salvar_json,
)
from experience_mining.nucleo.jornadas import MODELOS, NOMES_MODELOS

PREFIXOS = {
    "cartoes": "Cartoes",
    "investimentos": "Invest",
    "pix": "PIX",
    "canais": "Home",
    "openfinance": "OpenFinance",
    "open_finance": "OpenFinance",
}

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


def _resolver_caminho_cluster(clusters: ConfigAgrupamentos | Path, modelo: str) -> Path:
    if isinstance(clusters, ConfigAgrupamentos):
        return clusters.caminho_modelo(modelo)
    return Path(clusters) / f"{modelo}.pkl"


def _modelos_presentes(clusters: ConfigAgrupamentos | Path) -> list[str]:
    return [m for m in MODELOS if _resolver_caminho_cluster(clusters, m).exists()]


def nome_legivel(token: str) -> str:
    if ":" not in token:
        return token
    dominio, acao = token.split(":", 1)
    acao = acao.replace("view_", "tela_", 1) if acao.startswith("view_") else acao
    return f"{PREFIXOS.get(dominio, dominio.capitalize())}_{acao}"


def treinar_surrogate(X, labels, random_state: int = 42, **xgb_kwargs):
    labels = np.asarray(labels)
    m = labels != -1
    Xv, yv = X[m], labels[m]
    classes = sorted(set(yv.tolist()))
    mapa = {c: i for i, c in enumerate(classes)}
    y = np.array([mapa[c] for c in yv])
    params = dict(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="mlogloss",
    )
    params.update(xgb_kwargs)
    clf = xgb.XGBClassifier(**params).fit(Xv, y)
    return clf, Xv, y, classes


def valores_shap(clf, Xv):
    sv = shap.TreeExplainer(clf).shap_values(Xv)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=2)
    return sv


def pesos_do_grupo(sv, Xv, y, i: int, eventos: list[str]) -> pd.DataFrame:
    dentro = y == i
    s_in, s_out = sv[dentro, :, i], sv[~dentro, :, i]
    linhas = []
    for j, ev in enumerate(eventos):
        if not np.any(sv[:, j, i] != 0):
            continue
        med = float(s_in[:, j].mean())
        linhas.append(
            {
                "evento": ev,
                "shap_medio_no_grupo": round(med, 5),
                "shap_abs_medio_no_grupo": round(float(np.abs(s_in[:, j]).mean()), 5),
                "shap_medio_fora_do_grupo": round(float(s_out[:, j].mean()), 5) if len(s_out) else 0.0,
                "pct_sessoes_do_grupo_com_evento": round(float((Xv[dentro, j] > 0).mean()), 3),
                "pct_sessoes_fora_com_evento": round(float((Xv[~dentro, j] > 0).mean()), 3) if len(s_out) else 0.0,
                "tfidf_medio_no_grupo": round(float(Xv[dentro, j].mean()), 5),
                "direcao": "empurra PARA o grupo (+)" if med >= 0 else "afasta do grupo (-)",
            }
        )
    return pd.DataFrame(linhas).sort_values("shap_abs_medio_no_grupo", ascending=False)


def beeswarm(ax_sv, Xv, nomes, titulo: str, caminho: Path | str | None = None, ax=None, max_display: int = 7) -> None:
    plt.sca(ax) if ax is not None else plt.figure(figsize=(10, 6))
    shap.summary_plot(
        ax_sv,
        Xv,
        feature_names=nomes,
        max_display=max_display,
        show=False,
        plot_size=None if ax is not None else (10, 6),
    )
    plt.title(titulo, fontsize=12, fontweight="bold")
    plt.xlabel(EIXO, fontsize=9, fontweight="bold")
    if caminho:
        dest = Path(caminho)
        dest.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close()


def executar_shap_pesos(
    secao: ConfigShap,
    tfidf_path: Path,
    clusters: ConfigAgrupamentos | Path,
) -> None:
    if not secao.enabled:
        return
    tf = carregar_representacao(tfidf_path)
    X = tf["X"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    eventos = list(tf["feature_names"])
    out_dir = Path(secao.output_jornadas_dir)
    todos, fid = [], []

    for m in _modelos_presentes(clusters):
        lab = carregar_agrupamento(_resolver_caminho_cluster(clusters, m))["cluster_labels"]
        clf, Xv, y, classes = treinar_surrogate(X, lab, random_state=secao.random_state)
        sv = valores_shap(clf, Xv)
        fid.append(
            {
                "modelo": m,
                "acuracia_treino_pct": round(clf.score(Xv, y) * 100, 2),
                "sessoes_sem_ruido": len(y),
            }
        )
        print(f"\n[{m}] acurácia de treino do substituto: {fid[-1]['acuracia_treino_pct']}%")
        for i, g in enumerate(classes):
            p = pesos_do_grupo(sv, Xv, y, i, eventos)
            p.insert(0, "grupo", g)
            p.insert(0, "modelo", m)
            salvar_csv(p, out_dir / m / "grupos" / f"grupo_{g:02d}" / "shap_pesos.csv", ptbr=True)
            todos.append(p)
            topo = p[p.shap_medio_no_grupo > 0].head(3)
            print(
                f"  grupo {g:2d} (n={int((y == i).sum())}): "
                + ", ".join(f"{e} ({v:+.2f})" for e, v in zip(topo.evento, topo.shap_medio_no_grupo))
            )

    if todos:
        salvar_csv(pd.concat(todos), out_dir / "shap_pesos_todos_modelos.csv", ptbr=True)
        salvar_csv(pd.DataFrame(fid), out_dir / "shap_fidelidade_treino.csv", ptbr=True)


def _salvar_shap_global_bar(sv, Xv, nomes: list[str], classes: list[int], titulo: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    sv_lista = [sv[:, :, i] for i in range(sv.shape[2])] if sv.ndim == 3 else [sv]
    shap.summary_plot(
        sv_lista,
        Xv,
        feature_names=nomes,
        class_names=[f"Cluster {c}" for c in classes],
        plot_type="bar",
        max_display=15,
        show=False,
        plot_size=None,
    )
    plt.title(titulo, fontweight="bold")
    plt.tight_layout()
    plt.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close()


def executar_shap_figuras(
    secao: ConfigShap,
    tfidf_path: Path,
    clusters: ConfigAgrupamentos | Path,
    figuras: ConfigFiguras | None = None,
) -> None:
    if not secao.enabled:
        return
    fig_ativa = figuras.enabled if figuras is not None else True
    fig_inc = figuras.include if figuras is not None else None
    quer_global = fig_ativa and (fig_inc is None or "shap_global" in fig_inc)
    quer_panel = fig_ativa and (fig_inc is None or "shap_panel" in fig_inc)
    quer_beeswarm = fig_ativa and secao.gerar_individuais and (fig_inc is None or "beeswarm" in fig_inc)
    titulos_revisados = figuras.revised_titles if figuras is not None else secao.revised_titles
    highlight_model = figuras.highlight_model if figuras is not None else "hdbscan_tf_idf"
    panel_clusters = figuras.shap_panel_clusters if figuras is not None else None

    fig_dir = Path(secao.output_figures_dir)
    tf = carregar_representacao(tfidf_path)
    X = tf["X"]
    X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    nomes = [nome_legivel(t) for t in tf["feature_names"]]

    if highlight_model and (quer_global or quer_panel):
        hl_path = _resolver_caminho_cluster(clusters, highlight_model)
        if hl_path.exists():
            cl = carregar_agrupamento(hl_path)["cluster_labels"]
            fig_dir.mkdir(parents=True, exist_ok=True)
            clf, Xv, y, classes = treinar_surrogate(X_arr, cl, random_state=secao.random_state)
            sv = valores_shap(clf, Xv)
            print(f"[+] Surrogate {NOMES_MODELOS[highlight_model]}: acurácia de treino {clf.score(Xv, y) * 100:.2f}%")

            if quer_global:
                titulo_global = (
                    "Importância global dos eventos na predição dos agrupamentos (SHAP)"
                    if highlight_model == "hdbscan_tf_idf"
                    else f"Importância global dos eventos na predição dos agrupamentos (SHAP) — {NOMES_MODELOS[highlight_model]}"
                )
                _salvar_shap_global_bar(
                    sv,
                    Xv,
                    nomes,
                    classes,
                    titulo_global,
                    fig_dir / "shap_summary_bar_plot_global.png",
                )

            if quer_panel:
                if panel_clusters is not None:
                    faltam = [c for c in panel_clusters if c not in classes]
                    if faltam:
                        print(f"[!] Aviso: clusters {faltam} ausentes em {highlight_model}; painel exibirá os existentes.")
                    escolhidos = [c for c in panel_clusters if c in classes]
                else:
                    escolhidos = sorted(classes, key=lambda c: (-int((y == classes.index(c)).sum()), c))[:4]

                if escolhidos:
                    usar_tcc = highlight_model == "hdbscan_tf_idf" and escolhidos == [5, 3, 10, 0]
                    titulos = (TITULOS_REVISADOS if titulos_revisados else TITULOS_TCC) if usar_tcc else {}
                    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
                    axes_flat = axes.ravel()
                    for ax, c in zip(axes_flat, escolhidos):
                        i = classes.index(c)
                        tit_c = titulos[c] if usar_tcc else f"{NOMES_MODELOS[highlight_model]} · Cluster {c} (n={int((y == i).sum())})"
                        beeswarm(sv[:, :, i], Xv, nomes, tit_c, ax=ax)
                    for ax in axes_flat[len(escolhidos) :]:
                        ax.set_visible(False)
                    plt.tight_layout()
                    plt.savefig(fig_dir / "shap_beeswarm_panel_2x2.png", dpi=150, bbox_inches="tight")
                    plt.close()
            print(f"[+] Figuras 4 e 5 salvas em {fig_dir}")

    cs = secao.colsample_sensitivity
    if cs:
        cs_model = cs["model"]
        cs_path = _resolver_caminho_cluster(clusters, cs_model)
        if cs_path.exists():
            cl_cs = carregar_agrupamento(cs_path)["cluster_labels"]
            cs_colsample = float(cs.get("colsample_bytree", 0.3))
            cs_cluster = int(cs["cluster"])
            cs_top_n = int(cs.get("top_n", 5))
            c4, X4, y4, cls4 = treinar_surrogate(
                X_arr, cl_cs, random_state=secao.random_state, colsample_bytree=cs_colsample
            )
            if cs_cluster in cls4:
                s4 = valores_shap(c4, X4)
                i_c = cls4.index(cs_cluster)
                media = s4[y4 == i_c, :, i_c].mean(axis=0)
                top = np.argsort(media)[::-1][:cs_top_n]
                top_eventos = [
                    {"evento": tf["feature_names"][j], "shap_medio": round(float(media[j]), 4)}
                    for j in top
                ]
                cs_filename = cs.get("output_filename")
                if cs_filename == "colsample_cluster5.json":
                    res = {"cluster": cs_cluster, "colsample_bytree": cs_colsample, "top_eventos": top_eventos}
                else:
                    cs_filename = cs_filename or f"colsample_{cs_model}_cluster{cs_cluster}.json"
                    res = {
                        "modelo": cs_model,
                        "cluster": cs_cluster,
                        "colsample_bytree": cs_colsample,
                        "top_eventos": top_eventos,
                    }
                salvar_json(res, Path(secao.output_dados_dir) / cs_filename, ensure_ascii=False)

    if quer_global or quer_beeswarm:
        out_dir = Path(secao.output_jornadas_dir)
        for m in _modelos_presentes(clusters):
            lab = carregar_agrupamento(_resolver_caminho_cluster(clusters, m))["cluster_labels"]
            c_m, Xv_m, y_m, cls_m = treinar_surrogate(X_arr, lab, random_state=secao.random_state)
            sv_m = valores_shap(c_m, Xv_m)
            if quer_global:
                _salvar_shap_global_bar(
                    sv_m,
                    Xv_m,
                    nomes,
                    cls_m,
                    f"Importância global dos eventos (SHAP) — {NOMES_MODELOS[m]}",
                    out_dir / m / "shap_global.png",
                )
            if quer_beeswarm:
                for i, g in enumerate(cls_m):
                    beeswarm(
                        sv_m[:, :, i],
                        Xv_m,
                        list(tf["feature_names"]),
                        f"{NOMES_MODELOS[m]} · Cluster {g} (n={int((y_m == i).sum())})",
                        caminho=out_dir / m / "grupos" / f"grupo_{g:02d}" / "beeswarm.png",
                    )
            print(f"[+] {m}: {'shap_global ' if quer_global else ''}{len(cls_m) if quer_beeswarm else 0} beeswarms")
