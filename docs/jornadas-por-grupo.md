# Jornadas por grupo (`results/jornadas_por_grupo/`)

Gerado por `experience_mining/passos/jornadas_grupo.py` (passo 6 do `run_all.py`). Rótulo: `data/sessions_represented.csv` (Tabela 1 revisada; `dataset/gerar_massas.py` → `experience_mining/nucleo/rotulo.py`).

Uma pasta por modelo (`kmeans_|hdbscan_` × `tf_idf|minilm|gemini`). Em cada uma:

- `resumo_grupos.csv`: uma linha por grupo, com tamanho, jornada dominante, % de sucesso e composição por jornada.
- `matriz_grupos.png` (ou `matriz_grupos_KdeN.png`, quando há muitos grupos): grupos × rótulo.
  - Colunas pelo id da jornada (001, 002...), com o sucesso ao lado do atrito da mesma jornada. As 12 jornadas aparecem sempre, mesmo sem sessões (`N=0`, em cinza claro). `001+008` = concluiu as duas jornadas sem formar uma composta (ex.: pagou a fatura antes de resgatar o CDB; a 009 exige resgatar e depois pagar). `neutro` por último. Azul = sucesso, vermelho = atrito.
  - Célula = % das sessões do grupo com o rótulo e, entre parênteses, % das sessões do rótulo que estão no grupo. `n` = sessões do grupo; `N` = sessões com o rótulo.
  - Ruído (HDBSCAN) na última linha. A legenda dos ids está no rodapé.
- `espaco_semantico_tsne.png`: t-SNE 2D da representação do modelo (TF-IDF, MiniLM ou Gemini), colorido por grupo, com o número do grupo no centro dele; ruído em x. Grupos próximos no mapa têm sessões parecidas na representação.
- `espaco_semantico_tsne_3d.png`: o mesmo mapa em 3 dimensões (matplotlib), visto de dois ângulos.
- `shap_global.png`: importância global dos eventos na predição dos agrupamentos desse modelo (SHAP sobre o substituto XGBoost).
- `petri_parametros.csv`: parâmetros do Inductive Miner por grupo.
- `grupos/grupo_NN/` (e `grupos/ruido/` no HDBSCAN): `sessoes.csv`, `shap_pesos.csv` (`experience_mining/passos/shap_xai.py`; [leitura-shap.md](leitura-shap.md)), `dfg.png`, `dfg.dfg`, `rede_petri.png`/`.pnml` (`experience_mining/passos/processos_pm4py.py`, seção `pm4py` da configuração) e `beeswarm.png` (`experience_mining/passos/shap_xai.py`).

Na raiz: `resumo_grupos.csv` (os 6 modelos), `shap_pesos_todos_modelos.csv` e `shap_fidelidade_treino.csv`.

Colunas de `sessoes.csv`: `primary_label` (rótulo; sessões que concluíram duas jornadas recebem os nomes unidos por "+"), `experience` (success, attrition ou none), `ordem_conclusao` (ids das jornadas na ordem em que foram concluídas), `num_events`, `flags_ativas` (flags `is_*` = 1 pela regra da Tabela 1) e `text_representation` (sequência de eventos).

Separador `;`, UTF-8 com BOM.
