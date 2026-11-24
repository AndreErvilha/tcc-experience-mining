# Análises de apoio às hipóteses (v1.1)

Análises descritivas sobre os resultados da v1.1. Não alteram agrupamentos, rótulos nem métricas oficiais.

| Script | Pergunta | Saída (saidas/) |
|---|---|---|
| `montar_tabela_sessoes.py` | Tabela única sessão × grupo dos 6 modelos (base das outras) | `sessoes_por_modelo.csv` |
| `achado_011_012.py` | As jornadas 011/012, não previstas na Tabela 1 original, formam grupo sem supervisão? A IA (cega) as nomeia? Incluí-las infla ARI/V-Measure? | `achado_011_012.csv`, `sensibilidade_011_012.csv` |
| `ordem_eventos.py` | A ordem dos eventos acrescenta informação nesta massa (hipótese do "omelete")? | `ordem_eventos_pares.csv` |
| `atrito_invisivel.py` | Grupos da mesma jornada com a mesma conversão e esforço muito diferente; qual evento o SHAP aponta | `atrito_invisivel.csv` |

Ordem: `montar_tabela_sessoes.py` primeiro. `achado_011_012.py` lê `.ia-rotulos/v1.1/` (fora do git): sem a pasta, a coluna do nome da IA fica vazia.
