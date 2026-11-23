# O que tem em `results/` e quem gera

Cada versão gravada pelo `run_all.py` escreve na sua pasta (`results/<name>`, padrão em [configs/](../configs/)). Para regenerar a v1.1 completa em `results/v1.1/`, mantendo os agrupamentos do TCC e preservando `results/` da defesa:

```bash
python run_all.py --config configs/v1.1.json
```

`configs/v1.1.json` aponta para `dataset/processed/total.csv`, `total-rotulado-v1.1.csv`, `data/v1.0/*.pkl` e `clusters/v1.0/*.pkl` existentes, gravando em `data/v1.1/sessions_represented.csv` e `results/v1.1/`. O `pm4py` (DFG e Rede de Petri por grupo) roda por padrão na configuração (requer pm4py e Graphviz).

O SHAP depende das versões de `xgboost` e `shap` (ver `requirements-lock.txt`); com outras versões, o evento principal de cada grupo se mantém, mas os valores mudam.

| Arquivo (relativo a `results/<name>/` ou `results/`) | Módulo / Script | Passo do `run_all.py` |
| :--- | :--- | :--- |
| `../dataset/processed/<name>/total.csv` e `relatorio_geracao.csv` | `dataset/gerar_massas.py` (`gerar_eventos`) | 0.1 (`events_params`) |
| `../dataset/processed/<name>/total-rotulado.csv` | `dataset/gerar_massas.py` (`rotular_massa`) | 0.2 (`labeling_params`) |
| `metrics/metrics_<modelo>.json` | `experience_mining/passos/metricas.py` | 4 |
| `jornadas_por_grupo/<modelo>/grupos/grupo_NN/shap_pesos.csv`, `jornadas_por_grupo/shap_pesos_todos_modelos.csv`, `jornadas_por_grupo/shap_fidelidade_treino.csv` | `experience_mining/passos/shap_xai.py` | 5 |
| `jornadas_por_grupo/<modelo>/shap_global.png` (todos os modelos), `jornadas_por_grupo/<modelo>/grupos/grupo_NN/beeswarm.png` | `experience_mining/passos/shap_xai.py` | 5.1 |
| `figures/shap_summary_bar_plot_global.png` (Figura 4), `figures/shap_beeswarm_panel_2x2.png` (Figura 5), `dados/colsample_cluster5.json` *(destaque `figures.highlight_model` + `shap.colsample_sensitivity`; em `v1.1` = `hdbscan_tf_idf` / clusters `[5, 3, 10, 0]` / cluster `5`)* | `experience_mining/passos/shap_xai.py` | 5.1 |
| `jornadas_por_grupo/<modelo>/grupos/*/sessoes.csv`, `resumo_grupos.csv`, `matriz_grupos*.png`, `espaco_semantico_tsne.png`, `espaco_semantico_tsne_3d.png` | `experience_mining/passos/jornadas_grupo.py` | 6 |
| `dados/jornadas_contagem.csv`, `metrics/cobertura_por_jornada.csv` | `experience_mining/passos/metricas.py` | 6.1 |
| `jornadas_por_grupo/<modelo>/grupos/*/dfg.png`, `dfg.dfg`, `rede_petri.png`, `rede_petri.pnml`, `<modelo>/petri_parametros.csv` | `experience_mining/passos/processos_pm4py.py` | 6.2 (`pm4py`) |
| `metrics/comparacao_v1.0_v1.1.csv`, `figures/comparacao_v1.0_v1.1.png`, `dados/migracao_rotulos.*`, `dados/sessoes_rotulo_revisado.csv` | `analises/v1.1/comparacao_v1.0_v1.1.py` | Fora do `run_all.py` (análise v1.1) |
| `summary_metrics.csv` | `experience_mining/passos/metricas.py` | 7 |
| `figures/comparison_metrics.png` *(todos os modelos)*; `figures/matriz_associacao_clusters_labels.png` e `figures/tsne_clusters_visualizacao.png` *(destaque `figures.highlight_model`; em `v1.1` = `hdbscan_tf_idf`)* | `experience_mining/passos/figuras.py` | 8 |

> **Nota sobre destaques em `figures/` e `dados/`:** `comparison_metrics.png` agrega todos os modelos ativos; os demais gráficos em `figures/` e `dados/colsample_*.json` são recortes de um modelo específico configurável (`figures.highlight_model`, `figures.shap_panel_clusters`, `shap.colsample_sensitivity`). Na `v1.1`, eles mantêm os recortes de `hdbscan_tf_idf` referenciados no texto/errata do TCC; os equivalentes completos dos **6 modelos** (`matriz_grupos.png`, `espaco_semantico_tsne.png`, `shap_global.png` e `beeswarm.png`) ficam sempre em `jornadas_por_grupo/<modelo>/`.

A conferência independente do rótulo em Python puro fica em `tests/test_rotulagem.py`, e os testes do gerador/determinismo do playout ficam em `tests/test_gerador_massas.py` (`pytest`). O fluxo manual e os comandos de análise com `pm4py-cli` estão documentados em [pm4py-cli.md](pm4py-cli.md). As demais análises de `analises/` (`cluster5_composicao.py`, `densidade_resgate_fatura.py`, `desvios_e_truncamento.py`, `diag_shap.py`) imprimem no terminal e não escrevem em `results/`.
