"""Passos envolvendo pm4py e dataset/gerar_massas (geração, rotulagem e DFG/Petri por grupo)."""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py

from experience_mining.configuracao import ConfigAgrupamentos, ConfigEventos, ConfigPm4py, ConfigRotulagem
from experience_mining.io import carregar_agrupamento, salvar_csv
from experience_mining.nucleo.jornadas import MODELOS


def _resolver_caminho_cluster(clusters: ConfigAgrupamentos | Path, modelo: str) -> Path:
    if isinstance(clusters, ConfigAgrupamentos):
        return clusters.caminho_modelo(modelo)
    return Path(clusters) / f"{modelo}.pkl"


def executar_geracao_eventos(secao: ConfigEventos, root: Path) -> dict:
    from dataset.gerar_massas import gerar_eventos

    return gerar_eventos(secao, root=root)


def executar_rotulagem(secao: ConfigRotulagem, events_csv_path: Path, root: Path) -> Path:
    from dataset.gerar_massas import rotular_massa

    return rotular_massa(secao, events_csv_path=events_csv_path, root=root)


def executar_petri_por_grupo(
    secao: ConfigPm4py,
    clusters: ConfigAgrupamentos | Path,
    sessions_csv: Path,
) -> None:
    if not secao.enabled or (not secao.dfg and not secao.pnml):
        return
    warnings.filterwarnings("ignore")
    out_dir = Path(secao.output_jornadas_dir)
    s = pd.read_csv(sessions_csv).set_index("cd_sessao")
    base = pd.Timestamp("2025-01-01")

    for m in MODELOS:
        caminho = _resolver_caminho_cluster(clusters, m)
        if not caminho.exists():
            continue
        cl = carregar_agrupamento(caminho)
        lab = np.asarray(cl["cluster_labels"])
        s_alinhado = s.loc[list(cl["session_ids"])].reset_index()
        (out_dir / m).mkdir(parents=True, exist_ok=True)
        par = []

        for g in sorted(set(lab)):
            t0 = time.time()
            sub = s_alinhado[lab == g]
            nome = "ruido" if g == -1 else f"grupo_{g:02d}"
            grupo_dir = out_dir / m / "grupos" / nome
            grupo_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            for _, r in sub.iterrows():
                ev = r.text_representation.split()
                rows += [(r.cd_sessao, e, base + pd.Timedelta(seconds=i)) for i, e in enumerate(ev)]

            log = pm4py.format_dataframe(
                pd.DataFrame(rows, columns=["c", "a", "t"]),
                case_id="c",
                activity_key="a",
                timestamp_key="t",
            )
            ruido = 0.2 if len(sub) > 100 else 0.0

            if secao.dfg:
                dfg, i, f = pm4py.discover_dfg(log)
                pm4py.save_vis_dfg(dfg, i, f, str(grupo_dir / "dfg.png"))
                pm4py.write_dfg(dfg, i, f, str(grupo_dir / "dfg.dfg"))

            if secao.pnml:
                n, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=ruido)
                pm4py.save_vis_petri_net(n, im, fm, str(grupo_dir / "rede_petri.png"))
                pm4py.write_pnml(n, im, fm, str(grupo_dir / "rede_petri.pnml"))
                lugares, transicoes = len(n.places), len(n.transitions)
            else:
                lugares, transicoes = 0, 0

            par.append(
                dict(
                    grupo=g,
                    sessoes=len(sub),
                    eventos=len(rows),
                    atividades_distintas=log["a"].nunique(),
                    filtro_ruido_inductive=ruido,
                    lugares=lugares,
                    transicoes=transicoes,
                )
            )
            print(m, nome, len(sub), ruido, round(time.time() - t0, 1), flush=True)

        salvar_csv(pd.DataFrame(par), out_dir / m / "petri_parametros.csv", ptbr=True)
