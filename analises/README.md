# Análises citadas na errata

Scripts que sustentam correções e limitações apontadas na errata da versão entregue. Não fazem parte do
pipeline de resultados (`run_all.py` roda só `rotulo_revisado.py` e `conferir_rotulo_tabela1.py`).
Uso, da raiz do repositório: `python analises/<script>.py .`

| Script | O que mostra | Item da errata |
|---|---|---|
| `rotulo_revisado.py` | Métricas com o rótulo da v1.0 × v1.1 e de onde veio cada rótulo novo (`results/metrics/comparacao_v1.0_v1.1.csv`, `results/dados/migracao_rotulos.csv`) | Regras de rotulagem |
| `conferir_rotulo_tabela1.py` | Refaz as regras da Tabela 1 revisada em Python puro e confere com o rótulo do pipeline (`results/dados/conferencia_rotulos.csv`) | Regras de rotulagem |
| `cluster5_composicao.py` | Composição do Cluster 5 do HDBSCAN + TF-IDF: 49 só resgataram, 12 resgataram e pagaram | Cluster 5 = resgate de CDB |
| `densidade_resgate_fatura.py` | Vizinhos e densidade das sessões de resgate + fatura em cada representação | H2: a jornada composta vira ruído |
| `desvios_e_truncamento.py` | Truncamento do MiniLM em 256 subtokens (71% das sessões) | Limitações |
| `diag_shap.py` | Por que o SHAP concentra o crédito em um evento por grupo (blocos de eventos redundantes) | SHAP: marcadores, não causas |

`cluster5_composicao.py`, `densidade_resgate_fatura.py` e `diag_shap.py` reproduzem os números da errata com os artefatos
da versão entregue (tag `v1.0-entrega`). `desvios_e_truncamento.py` baixa só o tokenizer público do `all-MiniLM-L6-v2`.
