# Como ler o SHAP por grupo

Pesos e figuras (beeswarms e Figuras 4 e 5): `experience_mining/passos/shap_xai.py` (passos 5 e 5.1 do `run_all.py`). Os 6 modelos foram analisados; o TCC entregue só citou o HDBSCAN + TF-IDF.

**Modelo substituto:** XGBoost multiclasse sobre a matriz TF-IDF (eventos), sem as sessões de ruído (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `random_state=42`, os mesmos do TCC). Os grupos do MiniLM e do Gemini também são explicados pelos eventos (projeção na matriz TF-IDF). A acurácia de treino do substituto de cada modelo está em `results/<name>/jornadas_por_grupo/shap_fidelidade_treino.csv`.

## Beeswarm (`<modelo>/grupos/grupo_NN/beeswarm.png`)

- Cada ponto é uma sessão (todas as sessões fora do ruído); cada linha é um evento (os 7 mais fortes), com o nome completo do log, ex.: `investimentos:view_home_investimentos`.
- Eixo x = valor SHAP para o grupo: à direita, o evento empurra a sessão **para** o grupo; à esquerda, para fora.
- Cor = valor do evento na sessão: vermelho = presente/forte, azul = ausente/fraco.
- Leitura típica: pontos vermelhos à direita = o evento caracteriza o grupo; pontos vermelhos à esquerda = o evento indica outro grupo.

## Pesos (`<modelo>/grupos/grupo_NN/shap_pesos.csv`)

Uma linha por (grupo, evento) com SHAP diferente de zero, ordenada pela magnitude:

| Coluna | Significado |
| :--- | :--- |
| `shap_medio_no_grupo` | SHAP médio do evento, para a classe do grupo, nas sessões do grupo. Positivo = empurra para o grupo |
| `shap_abs_medio_no_grupo` | Magnitude média (ordena a lista) |
| `shap_medio_fora_do_grupo` | O mesmo, nas sessões dos outros grupos |
| `pct_sessoes_do_grupo_com_evento` / `pct_sessoes_fora_com_evento` | Presença do evento (TF-IDF > 0). Evento forte e presente também fora do grupo discrimina menos |
| `tfidf_medio_no_grupo` | Peso médio do evento no grupo |

`shap_pesos_todos_modelos.csv` (raiz de `results/<name>/jornadas_por_grupo/`): os pesos de todos os grupos dos 6 modelos num arquivo só, com as colunas `modelo` e `grupo`. Substitui o `results/xai_shap/shap_cluster_signatures.json` da v1.0, que cobria só o HDBSCAN + TF-IDF.

SHAP indica associação com o grupo, não causa (`analises/diag_shap.py`).
