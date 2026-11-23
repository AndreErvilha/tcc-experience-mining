# O que tem em `results/` e quem gera

Tudo em `results/` é gerado pelo `run_all.py`. Para regenerar do zero, mantendo os agrupamentos do TCC:

```bash
python run_all.py --reusar-agrupamentos --petri
```

`--reusar-agrupamentos` usa `data/*.pkl` e `clusters/*.pkl`, que ficam fora de `results/`. Sem ele, os vetores e os agrupamentos são refeitos e o Gemini só é gerado de novo se `data/gemini_embeddings.pkl` não existir. `--petri` é necessário para os DFGs e as Redes de Petri por grupo (pm4py e Graphviz).

O SHAP depende das versões de `xgboost` e `shap` (ver `requirements-lock.txt`); com outras versões, o evento principal de cada grupo se mantém, mas os valores mudam.

| Arquivo | Script | Passo do `run_all.py` |
| :--- | :--- | :--- |
| `metrics/metrics_<modelo>.json` | `src/eval_metrics.py` | 4 |
| `jornadas_por_grupo/<modelo>/grupos/grupo_NN/shap_pesos.csv`, `jornadas_por_grupo/shap_pesos_todos_modelos.csv`, `jornadas_por_grupo/shap_fidelidade_treino.csv` | `src/shap_explainer.py` | 5 |
| `figures/shap_summary_bar_plot_global.png` (Figura 4), `figures/shap_beeswarm_panel_2x2.png` (Figura 5), `dados/colsample_cluster5.json` | `src/shap_figures.py` | 5.1 |
| `jornadas_por_grupo/<modelo>/grupos/grupo_NN/beeswarm.png` | `src/shap_figures.py` | 5.1 |
| `jornadas_por_grupo/<modelo>/grupos/*/sessoes.csv`, `resumo_grupos.csv`, `matriz_grupos*.png`, `espaco_semantico_tsne.png`, `espaco_semantico_tsne_3d.png` | `src/jornadas_por_grupo.py` | 6 |
| `dados/jornadas_contagem.csv`, `metrics/cobertura_por_jornada.csv` | `src/jornadas_tabelas.py` | 6.1 |
| `jornadas_por_grupo/<modelo>/grupos/*/dfg.png`, `rede_petri.png`, `rede_petri.pnml`, `<modelo>/petri_parametros.csv` | `src/petri_por_grupo.py` | 6.2 (`--petri`) |
| `metrics/comparacao_v1.0_v1.1.csv`, `figures/comparacao_v1.0_v1.1.png`, `dados/migracao_rotulos.*`, `dados/sessoes_rotulo_revisado.csv` | `analises/rotulo_revisado.py` | 7.1 |
| `dados/conferencia_rotulos.csv` | `analises/conferir_rotulo_tabela1.py` | 7.2 |
| `summary_metrics.csv` | `run_all.py` (consolidação) | 8 |
| `figures/comparison_metrics.png`, `figures/matriz_associacao_clusters_labels.png`, `figures/tsne_clusters_visualizacao.png` | `src/figuras_tcc.py` | 9 |

As demais análises de `analises/` (`cluster5_composicao.py`, `densidade_resgate_fatura.py`, `desvios_e_truncamento.py`, `diag_shap.py`) imprimem no terminal e não escrevem em `results/`.
