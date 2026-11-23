"""Testes do esquema JSON, precedência de seleção, contratos e pastas por versão."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experience_mining.configuracao import (
    ErroConfiguracao,
    carregar_config_json,
    resolver_caminho_config,
)
from experience_mining.orquestracao import Pipeline

ROOT = Path(__file__).resolve().parent.parent


def _config_minima_v1_1() -> dict:
    return {
        "name": "v1.1",
        "events": {"events_csv_path": "dataset/processed/total.csv"},
        "labeling": {"labeling_csv_path": "dataset/processed/total-rotulado-v1.1.csv"},
        "representations": {
            "tfidf": {"tfidf_pkl_path": "data/v1.0/tfidf_vectors.pkl"},
            "minilm": {"minilm_pkl_path": "data/v1.0/minilm_embeddings.pkl"},
            "gemini": {"gemini_pkl_path": "data/v1.0/gemini_embeddings.pkl"},
        },
        "clusters": {
            "kmeans": {
                "tfidf_pkl_path": "clusters/v1.0/kmeans_tf_idf.pkl",
                "minilm_pkl_path": "clusters/v1.0/kmeans_minilm.pkl",
                "gemini_pkl_path": "clusters/v1.0/kmeans_gemini.pkl",
            },
            "hdbscan": {
                "tfidf_pkl_path": "clusters/v1.0/hdbscan_tf_idf.pkl",
                "minilm_pkl_path": "clusters/v1.0/hdbscan_minilm.pkl",
                "gemini_pkl_path": "clusters/v1.0/hdbscan_gemini.pkl",
            },
        },
    }


def _escrever_cfg(tmp_path: Path, dados: dict, nome: str = "cfg.json") -> Path:
    arq = tmp_path / nome
    arq.write_text(json.dumps(dados), encoding="utf-8")
    return arq


def test_selecao_precedencia_e_erro(tmp_path, monkeypatch):
    monkeypatch.delenv("EM_CONFIG", raising=False)
    c1 = _escrever_cfg(tmp_path, _config_minima_v1_1(), "cli.json")
    c2 = _escrever_cfg(tmp_path, _config_minima_v1_1(), "env.json")
    c3 = _escrever_cfg(tmp_path, _config_minima_v1_1(), "dotenv.json")

    # Sem --config nem EM_CONFIG -> erro apontando para .env.example
    with pytest.raises(ErroConfiguracao, match=r"\.env\.example"):
        resolver_caminho_config(None, root=tmp_path)

    # .env funciona quando não há EM_CONFIG no ambiente nem --config
    (tmp_path / ".env").write_text(f"EM_CONFIG={c3}\n", encoding="utf-8")
    assert resolver_caminho_config(None, root=tmp_path) == c3.resolve()

    # EM_CONFIG do ambiente tem precedência sobre .env (override=False)
    monkeypatch.setenv("EM_CONFIG", str(c2))
    assert resolver_caminho_config(None, root=tmp_path) == c2.resolve()

    # --config tem precedência máxima
    assert resolver_caminho_config(str(c1), root=tmp_path) == c1.resolve()


def test_schema_rejeita_invalidos(tmp_path):
    # 1. Tipo errado ("shap": true)
    d = _config_minima_v1_1()
    d["shap"] = True
    with pytest.raises(ErroConfiguracao, match="shap deve ser objeto"):
        carregar_config_json(_escrever_cfg(tmp_path, d), root=ROOT, validar_contratos=False)

    # 2. Enum inválido em metrics.include
    d = _config_minima_v1_1()
    d["metrics"] = {"include": ["arii"]}
    with pytest.raises(ErroConfiguracao, match="arii"):
        carregar_config_json(_escrever_cfg(tmp_path, d), root=ROOT, validar_contratos=False)

    # 3. Chave desconhecida
    d = _config_minima_v1_1()
    d["outupt_dir_path"] = "results/x"
    with pytest.raises(ErroConfiguracao, match="outupt_dir_path"):
        carregar_config_json(_escrever_cfg(tmp_path, d), root=ROOT, validar_contratos=False)

    # 4. Dois modos em events
    d = _config_minima_v1_1()
    d["events"] = {
        "events_csv_path": "dataset/processed/total.csv",
        "events_params": [{"id": "001", "seed_csv_path": "a.csv"}],
    }
    with pytest.raises(ErroConfiguracao, match="nunca os dois"):
        carregar_config_json(_escrever_cfg(tmp_path, d), root=ROOT, validar_contratos=False)

    # 5. Nenhum modo em labeling
    d = _config_minima_v1_1()
    d["labeling"] = {}
    with pytest.raises(ErroConfiguracao, match="obrigatório"):
        carregar_config_json(_escrever_cfg(tmp_path, d), root=ROOT, validar_contratos=False)

    # 6. Acompanhante sem events_params
    d = _config_minima_v1_1()
    d["events"] = {
        "events_csv_path": "dataset/processed/total.csv",
        "defaults": {"num_traces": 100},
    }
    with pytest.raises(ErroConfiguracao, match="defaults"):
        carregar_config_json(_escrever_cfg(tmp_path, d), root=ROOT, validar_contratos=False)

    # 7. Dois modos em representations.tfidf e clusters.kmeans
    d = _config_minima_v1_1()
    d["representations"]["tfidf"] = {"tfidf_pkl_path": "x.pkl", "tfidf_params": {"min_df": 1}}
    with pytest.raises(ErroConfiguracao, match="nunca os dois"):
        carregar_config_json(_escrever_cfg(tmp_path, d), root=ROOT, validar_contratos=False)


def test_padroes_e_pastas_por_versao(tmp_path):
    d = _config_minima_v1_1()
    cfg = carregar_config_json(_escrever_cfg(tmp_path, d), root=ROOT, validar_contratos=True)
    assert cfg.name == "v1.1"
    assert cfg.output_dir == ROOT / "results" / "v1.1"
    assert cfg.eventos.sessions_csv_path == ROOT / "data" / "v1.1" / "sessions_represented.csv"
    assert cfg.metricas.enabled is True
    assert cfg.shap.enabled is True
    assert cfg.jornadas_grupo.enabled is True
    assert cfg.pm4py.enabled is True
    assert cfg.pm4py.dfg is True and cfg.pm4py.pnml is True
    assert cfg.figuras.enabled is True
    assert cfg.figuras.revised_titles is False

    # Passo desligado e output_dir_path customizado
    d["output_dir_path"] = "results/custom"
    d["shap"] = {"enabled": False}
    cfg2 = carregar_config_json(_escrever_cfg(tmp_path, d), root=ROOT, validar_contratos=False)
    assert cfg2.output_dir == ROOT / "results" / "custom"
    assert cfg2.shap.enabled is False
    assert cfg2.metricas.enabled is True

    # Modo gerar usa pastas <name> por padrão
    d_gen = {
        "name": "v2.0",
        "events": {"events_params": [{"id": "001", "seed_csv_path": "s.csv"}]},
        "labeling": {"labeling_params": [{"label_col": "is_x", "contain": ["ev"]}]},
        "representations": {"tfidf": {"tfidf_params": {"min_df": 1}}},
        "clusters": {"kmeans": {"kmeans_params": {"auto_k": True}}},
    }
    cfg3 = carregar_config_json(_escrever_cfg(tmp_path, d_gen), root=ROOT, validar_contratos=False)
    assert cfg3.eventos.output_events_csv_path == ROOT / "dataset" / "processed" / "v2.0" / "total.csv"
    assert cfg3.labeling.output_labeling_csv_path == ROOT / "dataset" / "processed" / "v2.0" / "total-rotulado.csv"
    assert cfg3.representacoes.tfidf_path == ROOT / "data" / "v2.0" / "tfidf"
    assert cfg3.agrupamentos.caminhos_modelo["kmeans_tf_idf"] == ROOT / "clusters" / "v2.0" / "kmeans_tf_idf"
    assert cfg3.output_dir == ROOT / "results" / "v2.0"


def test_configs_versionados_validos():
    cfg_v11 = carregar_config_json(ROOT / "configs" / "v1.1.json", root=ROOT, validar_contratos=True)
    assert cfg_v11.output_dir == ROOT / "results" / "v1.1"

    cfg_v20 = carregar_config_json(ROOT / "configs" / "v2.0.json", root=ROOT, validar_contratos=True)
    assert cfg_v20.output_dir == ROOT / "results" / "v2.0"
    assert len(cfg_v20.eventos.events_params) == 10
    assert len(cfg_v20.labeling.labeling_params) == 24
    assert cfg_v20.figuras.revised_titles is True


def test_execucao_portatil_com_opcoes_seletivas(tmp_path):
    cfg_dict = {
        "name": "teste_portatil",
        "output_dir_path": str(tmp_path / "results"),
        "events": {"events_csv_path": str(ROOT / "dataset" / "processed" / "total.csv")},
        "labeling": {"labeling_csv_path": str(ROOT / "dataset" / "processed" / "total-rotulado-v1.1.csv")},
        "representations": {
            "tfidf": {
                "output_tfidf_dir_path": str(tmp_path / "data" / "tfidf"),
                "tfidf_params": {"min_df": 1},
            }
        },
        "clusters": {
            "kmeans": {
                "output_clusters_dir_path": str(tmp_path / "clusters"),
                "kmeans_params": {"auto_k": False, "n_clusters": 5, "seed": 42},
            }
        },
        "metrics": {"enabled": True, "include": ["ari_clean", "v_measure"]},
        "shap": {"enabled": False},
        "journeys_by_cluster": {"enabled": True, "tsne_seed": 42},
        "pm4py": {"dfg": False, "pnml": False},
        "figures": {"enabled": True, "include": ["comparison_metrics"], "revised_titles": True},
    }
    cfg_path = _escrever_cfg(tmp_path, cfg_dict, "sel.json")
    cfg = carregar_config_json(cfg_path, root=ROOT, validar_contratos=True)
    cfg.eventos.sessions_csv_path = tmp_path / "sessions_represented.csv"
    Pipeline(cfg).executar()

    # Representação e agrupamento portáteis gravados sem rótulo
    assert (tmp_path / "data" / "tfidf" / "vectors.npy").exists()
    assert (tmp_path / "clusters" / "kmeans_tf_idf" / "groups.csv").exists()

    # Métricas filtradas e sem cobertura (não incluída)
    m = json.loads((tmp_path / "results" / "metrics" / "metrics_kmeans_tf_idf.json").read_text(encoding="utf-8"))
    assert "ari_clean" in m and "v_measure" in m
    assert "silhouette_score" not in m
    assert not (tmp_path / "results" / "metrics" / "cobertura_por_jornada.csv").exists()

    # SHAP desligado, pm4py desligado, figures seletivo
    assert not (tmp_path / "results" / "jornadas_por_grupo" / "shap_pesos_todos_modelos.csv").exists()
    assert (tmp_path / "results" / "figures" / "comparison_metrics.png").exists()
    assert not (tmp_path / "results" / "figures" / "tsne_clusters_visualizacao.png").exists()
    assert not (tmp_path / "results" / "jornadas_por_grupo" / "kmeans_tf_idf" / "espaco_semantico_tsne.png").exists()
    assert (tmp_path / "results" / "jornadas_por_grupo" / "kmeans_tf_idf" / "resumo_grupos.csv").exists()


def test_run_all_rejeita_flags_antigas_e_falta_de_config(tmp_path):
    for flag in (
        "--sem-shap",
        "--petri",
        "--reusar-agrupamentos",
        "--rotular-massa",
        "--titulos-revisados",
    ):
        res = subprocess.run(
            [sys.executable, str(ROOT / "run_all.py"), flag],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 2
        assert "configuracao" in res.stderr.lower()

    # Sem --config nem EM_CONFIG preenchido (sobrescreve eventual .env na raiz) -> falha apontando para .env.example
    env = dict(os.environ)
    env["EM_CONFIG"] = ""
    res_sem = subprocess.run(
        [sys.executable, str(ROOT / "run_all.py")],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert res_sem.returncode == 2
    assert ".env.example" in res_sem.stderr


def test_shap_global_por_modelo_e_destaques_configuraveis(tmp_path):
    # 1. highlight_model=null e sem colsample_sensitivity: gera shap_global do modelo sem figures/dados especificos
    sub1 = tmp_path / "sem_destaque"
    sub1.mkdir()
    cfg1_dict = {
        "name": "sem_destaque",
        "output_dir_path": str(sub1 / "results"),
        "events": {"events_csv_path": str(ROOT / "dataset" / "processed" / "total.csv")},
        "labeling": {"labeling_csv_path": str(ROOT / "dataset" / "processed" / "total-rotulado-v1.1.csv")},
        "representations": {
            "tfidf": {
                "output_tfidf_dir_path": str(sub1 / "data" / "tfidf"),
                "tfidf_params": {"min_df": 1},
            }
        },
        "clusters": {
            "kmeans": {
                "output_clusters_dir_path": str(sub1 / "clusters"),
                "kmeans_params": {"auto_k": False, "n_clusters": 4, "seed": 42},
            }
        },
        "metrics": {"enabled": True},
        "shap": {"enabled": True, "seed": 42},
        "journeys_by_cluster": {"enabled": False},
        "pm4py": {"dfg": False, "pnml": False},
        "figures": {"enabled": True, "highlight_model": None},
    }
    cfg1 = carregar_config_json(_escrever_cfg(sub1, cfg1_dict, "c1.json"), root=ROOT, validar_contratos=True)
    cfg1.eventos.sessions_csv_path = sub1 / "sessions_represented.csv"
    Pipeline(cfg1).executar()

    assert (sub1 / "results" / "figures" / "comparison_metrics.png").exists()
    assert not (sub1 / "results" / "figures" / "matriz_associacao_clusters_labels.png").exists()
    assert not (sub1 / "results" / "figures" / "tsne_clusters_visualizacao.png").exists()
    assert not (sub1 / "results" / "figures" / "shap_summary_bar_plot_global.png").exists()
    assert not (sub1 / "results" / "figures" / "shap_beeswarm_panel_2x2.png").exists()
    assert not (sub1 / "results" / "dados" / "colsample_cluster5.json").exists()
    assert (sub1 / "results" / "jornadas_por_grupo" / "kmeans_tf_idf" / "shap_global.png").exists()

    # 2. highlight_model='kmeans_tf_idf', shap_panel_clusters=null (auto top-4) e colsample_sensitivity customizado
    sub2 = tmp_path / "com_destaque"
    sub2.mkdir()
    cfg2_dict = {
        "name": "com_destaque",
        "output_dir_path": str(sub2 / "results"),
        "events": {"events_csv_path": str(ROOT / "dataset" / "processed" / "total.csv")},
        "labeling": {"labeling_csv_path": str(ROOT / "dataset" / "processed" / "total-rotulado-v1.1.csv")},
        "representations": {
            "tfidf": {
                "output_tfidf_dir_path": str(sub2 / "data" / "tfidf"),
                "tfidf_params": {"min_df": 1},
            }
        },
        "clusters": {
            "kmeans": {
                "output_clusters_dir_path": str(sub2 / "clusters"),
                "kmeans_params": {"auto_k": False, "n_clusters": 3, "seed": 42},
            }
        },
        "metrics": {"enabled": True},
        "shap": {
            "enabled": True,
            "seed": 42,
            "colsample_sensitivity": {"model": "kmeans_tf_idf", "cluster": 0, "colsample_bytree": 0.5},
        },
        "journeys_by_cluster": {"enabled": False},
        "pm4py": {"dfg": False, "pnml": False},
        "figures": {"enabled": True, "highlight_model": "kmeans_tf_idf", "shap_panel_clusters": None},
    }
    cfg2 = carregar_config_json(_escrever_cfg(sub2, cfg2_dict, "c2.json"), root=ROOT, validar_contratos=True)
    cfg2.eventos.sessions_csv_path = sub2 / "sessions_represented.csv"
    Pipeline(cfg2).executar()

    assert (sub2 / "results" / "figures" / "shap_summary_bar_plot_global.png").exists()
    assert (sub2 / "results" / "figures" / "shap_beeswarm_panel_2x2.png").exists()
    assert (sub2 / "results" / "figures" / "matriz_associacao_clusters_labels.png").exists()
    assert (sub2 / "results" / "figures" / "tsne_clusters_visualizacao.png").exists()
    cs_json = json.loads((sub2 / "results" / "dados" / "colsample_kmeans_tf_idf_cluster0.json").read_text(encoding="utf-8"))
    assert cs_json["modelo"] == "kmeans_tf_idf"
    assert cs_json["cluster"] == 0
