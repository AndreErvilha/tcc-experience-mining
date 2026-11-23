"""Orquestração do pipeline de Experience Mining em ordem determinística."""
from pathlib import Path
import time

import pandas as pd

from experience_mining.configuracao import ConfigPipeline, REP_JSON_PARA_MODELO
from experience_mining.nucleo.rotulo import (
    classificar_experiencia,
    extract_primary_label,
    format_event_token,
    processar_eventos_csv,
)
from experience_mining.passos.agrupamento import executar_agrupamentos
from experience_mining.passos.figuras import executar_figuras_tcc
from experience_mining.passos.jornadas_grupo import executar_jornadas_por_grupo
from experience_mining.passos.metricas import (
    executar_metricas,
    executar_sumario_metricas,
    executar_tabelas_jornadas,
)
from experience_mining.passos.processos_pm4py import (
    executar_geracao_eventos,
    executar_petri_por_grupo,
    executar_rotulagem,
)
from experience_mining.passos.representacao import executar_representacoes
from experience_mining.passos.shap_xai import executar_shap_figuras, executar_shap_pesos


def _etapa(titulo: str, fn, *args, **kwargs):
    print(f"\n{'='*80}\n[*] {titulo}\n{'='*80}")
    t0 = time.time()
    res = fn(*args, **kwargs)
    print(f"[+] Concluido em {time.time() - t0:.2f}s")
    return res


def _preparar_sessoes(config: ConfigPipeline) -> None:
    ev_cfg = config.eventos
    lbl_cfg = config.labeling

    if lbl_cfg is not None:
        lbl_csv = (
            (lbl_cfg.output_labeling_csv_path if lbl_cfg.modo == "gerar" else lbl_cfg.labeling_csv_path)
            or ev_cfg.massa_rotulada_csv_path
        )
        df_lbl = pd.read_csv(lbl_csv)
        # Caso legado: CSV por evento contendo flags is_* (como total-rotulado-v1.1.csv)
        if "cd_evento_tipo" in df_lbl.columns:
            processar_eventos_csv(
                lbl_csv,
                output_csv=ev_cfg.sessions_csv_path,
                include_product=ev_cfg.incluir_produto,
            )
            return

        # Caso de CSV por sessão carregado ou gerado junto com events_csv_path
        df_ev = pd.read_csv(ev_cfg.events_csv_path)
        if "timestamp" in df_ev.columns:
            df_ev["timestamp"] = pd.to_datetime(df_ev["timestamp"], errors="coerce")
            df_ev = df_ev.sort_values(by=["cd_sessao", "timestamp"], kind="stable")
        df_ev["event_token"] = df_ev.apply(lambda r: format_event_token(r, ev_cfg.incluir_produto), axis=1)
        sess_rows = []
        for sess_id, grp in df_ev.groupby("cd_sessao", sort=False):
            toks = grp["event_token"].tolist()
            sess_rows.append({"cd_sessao": sess_id, "text_representation": " ".join(toks), "num_events": len(toks)})
        df_text = pd.DataFrame(sess_rows)

        is_cols = [c for c in df_lbl.columns if c.startswith("is_")]
        df_lbl = df_lbl.copy()
        if "primary_label" not in df_lbl.columns:
            df_lbl["primary_label"] = df_lbl[is_cols].apply(extract_primary_label, axis=1)
        if "experience" not in df_lbl.columns:
            df_lbl["experience"] = df_lbl[is_cols].apply(
                lambda r: classificar_experiencia({c: int(r[c]) for c in is_cols}), axis=1
            )
        merged = df_text.merge(df_lbl, on="cd_sessao", how="inner")
        out = Path(ev_cfg.sessions_csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, index=False)
        return

    processar_eventos_csv(
        ev_cfg.massa_rotulada_csv_path,
        output_csv=ev_cfg.sessions_csv_path,
        include_product=ev_cfg.incluir_produto,
    )


class Pipeline:
    """Executa os passos configurados do pipeline entregando a cada um apenas sua seção."""

    def __init__(self, config: ConfigPipeline):
        self.config = config

    def executar(self) -> None:
        c = self.config
        print("=" * 80)
        print(f" REPRODUCAO DO PIPELINE EXPERIMENTAL DE EXPERIENCE MINING ({c.name})")
        print(" MBA Gestao de Negocios Digitais e IA - USP/ESALQ")
        print("=" * 80)

        if c.eventos.modo == "gerar":
            _etapa(
                "PASSO 0.1: Geracao da massa de eventos (dataset/gerar_massas.py)",
                executar_geracao_eventos,
                c.eventos,
                c.root,
            )
        if c.labeling is not None and c.labeling.modo == "gerar":
            _etapa(
                "PASSO 0.2: Rotulagem da massa inteira (dataset/gerar_massas.py)",
                executar_rotulagem,
                c.labeling,
                c.eventos.events_csv_path,
                c.root,
            )

        _etapa(
            "PASSO 1: Representacao Textual e rotulo das Sessoes (nucleo/rotulo)",
            _preparar_sessoes,
            c,
        )

        if c.representacoes.gerar:
            reps = _etapa(
                "PASSO 2: Vetorizacoes (TF-IDF, MiniLM, Gemini)",
                executar_representacoes,
                c.representacoes,
                c.eventos.sessions_csv_path,
            )
        else:
            print("\n[*] Reusando representacoes existentes.")
            if c.representacoes.itens:
                reps = {REP_JSON_PARA_MODELO[k]: item.path for k, item in c.representacoes.itens.items()}
            else:
                reps = {
                    "tf_idf": c.representacoes.tfidf_path,
                    "minilm": c.representacoes.minilm_path,
                    "gemini": c.representacoes.gemini_path,
                }

        if c.agrupamentos.gerar:
            _etapa(
                "PASSO 3: Agrupamentos (K-Means e HDBSCAN)",
                executar_agrupamentos,
                c.agrupamentos,
                reps,
            )

        _etapa(
            "PASSO 4: Metricas de avaliacao",
            executar_metricas,
            c.metricas,
            c.agrupamentos,
            c.eventos.sessions_csv_path,
            reps=c.representacoes,
        )

        if c.shap.enabled:
            _etapa(
                "PASSO 5: Pesos SHAP de cada grupo",
                executar_shap_pesos,
                c.shap,
                c.representacoes.tfidf_path,
                c.agrupamentos,
            )
            _etapa(
                "PASSO 5.1: Figuras SHAP e sensibilidade colsample",
                executar_shap_figuras,
                c.shap,
                c.representacoes.tfidf_path,
                c.agrupamentos,
                figuras=c.figuras,
            )

        _etapa(
            "PASSO 6: Jornadas por grupo (resumos, matrizes e t-SNE 2D/3D)",
            executar_jornadas_por_grupo,
            c.jornadas_grupo,
            c.agrupamentos,
            c.eventos.sessions_csv_path,
            reps=c.representacoes,
            figuras=c.figuras,
        )
        _etapa(
            "PASSO 6.1: Tabelas por jornada (contagem e cobertura)",
            executar_tabelas_jornadas,
            c.metricas,
            c.agrupamentos,
            c.eventos.sessions_csv_path,
        )

        if c.pm4py.enabled:
            _etapa(
                "PASSO 6.2: DFG e Redes de Petri por grupo (pm4py)",
                executar_petri_por_grupo,
                c.pm4py,
                c.agrupamentos,
                c.eventos.sessions_csv_path,
            )

        _etapa(
            "PASSO 7: Consolidacao do sumario de metricas",
            executar_sumario_metricas,
            c.metricas,
        )
        _etapa(
            "PASSO 8: Figuras de resultados (results/<versao>/figures/)",
            executar_figuras_tcc,
            c.figuras,
            c.metricas.output_metrics_dir,
            c.agrupamentos,
            c.eventos.sessions_csv_path,
            reps=c.representacoes,
        )

        print(f"\n{'='*80}\n[+] REPRODUCAO FINALIZADA COM SUCESSO DE PONTA A PONTA!\n{'='*80}")
