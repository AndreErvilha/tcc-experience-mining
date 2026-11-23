# Mineração de Jornadas de Experiência do Usuário (*Experience Mining*)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Institution: USP/ESALQ](https://img.shields.io/badge/USP%2FESALQ-MBA%20Gest%C3%A3o%20de%20Neg%C3%B3cios%20Digitais%20e%20IA-navy)](https://mbauspesalq.com/)
[![Status: Reproducible](https://img.shields.io/badge/Status-100%25%20Reproducible-success.svg)](#reprodução-em-1-clique)

Repositório científico oficial do Trabalho de Conclusão de Curso (TCC) do **MBA em Gestão de Negócios Digitais e Inteligência Artificial da Universidade de São Paulo (USP / ESALQ)**.

* **Autor:** André Eduardo Ribeiro Ervilha ([andre.eduardo.ervilha@gmail.com](mailto:andre.eduardo.ervilha@gmail.com))
* **Orientadora:** Profa. Dra. Aline Araujo Perini
* **Ano de Apresentação:** 2026

---

## 📌 Resumo da Pesquisa (*Abstract*)

A análise contemporânea de experiência do cliente (*Customer Experience* - CX) em ecossistemas bancários digitais enfrenta um dilema estrutural: *dashboards* baseados em métricas agregadas (*bounce rate*, NPS) são estáticos e reativos, enquanto a mineração de processos clássica (*Process Mining*) sobre telemetria bruta gera grafos excessivamente densos e indecifráveis (*spaghetti models*).

Este trabalho propõe e valida experimentalmente o framework de **Experience Mining**, combinando:
1. **Modelagem Formal em Redes de Petri** e geração estocástica via *playout* de 10 jornadas bancárias (516 sessões, 28.786 eventos atômicos);
2. **Estudo Fatorial de Representações Vetoriais:** TF-IDF ponderado por funil, Sentence-BERT (`all-MiniLM-L6-v2`, 384D) e Google Gemini (`models/gemini-embedding-001`, 3072D);
3. **Clusterização Não-Supervisionada:** K-Means particional e HDBSCAN densitário;
4. **Explicabilidade Causal (XAI):** Surrogate XGBoost com valores SHAP (*TreeExplainer*) para descoberta das causas-raízes de conversão e atrito;
5. **Operacionalização em Streaming:** Detecção de anomalias em tempo de execução ($<1$ ms) via similaridade de cosseno contra centroides calibrados.

---

## 📊 Principais Resultados da Tríade Experimental

| Experimento | Representação | Algoritmo | $K$ | Ruído (%) | Silhouette | ARI (s/ ruído) | V-Measure | Pureza por Macro-Exp. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HDBSCAN + Gemini** | Gemini (3072D) | HDBSCAN | 21 | 19.4% | 0.3793 | 0.6538 | 0.8346 | **97.63%** 🏆 |
| **HDBSCAN + TF-IDF** | TF-IDF | HDBSCAN | 14 | 26.9% | **0.6477** 🏆 | **0.7268** 🏆 | **0.8851** 🏆 | 95.41% |
| **HDBSCAN + MiniLM** | MiniLM (384D) | HDBSCAN | 26 | 24.0% | 0.5426 | 0.4258 | 0.7183 | 92.76% |
| **K-Means + Gemini** | Gemini (3072D) | K-Means | 6 | 0.0% | 0.2942 | 0.4911 | 0.6590 | 91.37% |
| **K-Means + TF-IDF** | TF-IDF | K-Means | 15 | 0.0% | 0.4923 | 0.6076 | 0.7971 | 91.13% |
| **K-Means + MiniLM** | MiniLM (384D) | K-Means | 4 | 0.0% | 0.4037 | 0.2377 | 0.5078 | 90.17% |

**Versão 1.1 (revisão pós-entrega).** Os agrupamentos, o ruído e o Silhouette são os do TCC entregue (tag `v1.0-entrega`). ARI, V-Measure e Pureza usam o rótulo da **Tabela 1 revisada**: 12 jornadas definidas a partir do modelo de cada cenário, cada uma com gatilho e conversão; 005, 009 e 010 são compostas (`dataset/gerar_massas.py`). Comparação com a versão entregue: `results/metrics/comparacao_v1.0_v1.1.csv`. Cobertura por jornada: `results/metrics/cobertura_por_jornada.csv`. Sessões por jornada: `results/dados/jornadas_contagem.csv`.

> **Achado de produto (revisado):** os agrupamentos isolam sem regras as jornadas de um produto (pagamento de fatura, PIX, aplicação e resgate de CDB, parcelamento). As jornadas compostas, que cruzam produtos (Open Finance + fatura, resgate de CDB + fatura, fatura + PIX copia e cola), têm de 6 a 15 sessões e não formam grupo próprio em nenhum modelo. O Cluster 5 do HDBSCAN + TF-IDF é o grupo de resgate de CDB (ver a errata).

---

## 📂 Arquitetura do Repositório

```
tcc-experience-mining/
├── run_all.py                     # Orquestrador (reprodução de ponta a ponta)
├── requirements.txt / requirements-lock.txt
├── dataset/
│   ├── raw/                       # Logs-semente, Redes de Petri (.pnml) e playouts dos 10 cenários
│   ├── processed/
│   │   ├── total.csv              # Massa entregue: 516 sessões, 28.786 eventos
│   │   ├── total-rotulado.csv     # Rótulos da versão entregue (v1.0)
│   │   └── total-rotulado-v1.1.csv  # Rótulos pela Tabela 1 revisada (entrada do pipeline)
│   └── gerar_massas.py            # Gera e rotula a massa (pm4py-cli); na v1.1 rodamos só a rotulagem (--saida total-rotulado-v1.1.csv)
├── data/                          # Sessões em texto e vetores (TF-IDF, MiniLM, Gemini)
├── clusters/                      # Agrupamentos (K-Means e HDBSCAN × 3 representações)
├── src/                           # Pipeline
│   ├── events2vector.py           # Eventos → texto por sessão + flags + rótulo principal
│   ├── tfidf_vectorizer.py, embeddings_vectorizer.py
│   ├── kmeans_cluster.py, hdbscan_cluster.py
│   ├── eval_metrics.py            # Silhouette, ARI, V-Measure, pureza
│   ├── shap_explainer.py, shap_figures.py   # XAI (modelo substituto XGBoost + SHAP)
│   ├── jornadas_por_grupo.py      # Sessões e resumo de cada grupo
│   ├── jornadas_tabelas.py        # Sessões por jornada e cobertura por jornada
│   ├── petri_por_grupo.py         # DFG e Rede de Petri de cada grupo (run_all.py --petri)
│   ├── figuras_tcc.py             # Figuras de resultados
│   └── stream_inference.py        # Inferência em tempo de execução
├── analises/                      # Análises citadas na errata (ver analises/README.md)
├── docs/                          # Documentação dos resultados (ver docs/README.md)
├── tools/pm4py_cli/               # CLI própria de Process Mining, playout e rotulagem
└── results/
    ├── summary_metrics.csv
    ├── metrics/                   # Métricas por modelo, cobertura por jornada, comparação v1.0 × v1.1
    ├── figures/                   # Figuras de resultados e SHAP
    ├── dados/                     # Sessões por jornada, conferência e migração dos rótulos
    └── jornadas_por_grupo/        # Por modelo: resumo, matriz, mapa t-SNE e grupos/grupo_NN/ (sessões, pesos SHAP, beeswarm, DFG, Petri)
```

---

## 🚀 Como Reproduzir os Resultados

### 1. Pré-requisitos
* Python 3.10, 3.11 ou 3.12 instalado.
* Git instalado.

### 2. Instalação do Ambiente

```bash
# Clone o repositório
git clone https://github.com/AndreErvilha/tcc-experience-mining.git
cd tcc-experience-mining

# Crie e ative o ambiente virtual
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Instale as dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Reprodução em 1 Clique

Para rodar todo o pipeline de ponta a ponta (representações, rótulos, agrupamentos, métricas, SHAP, jornadas por grupo e figuras):

```bash
python run_all.py
```

Opções:

| Opção | O que faz |
| :--- | :--- |
| `--reusar-agrupamentos` | Não refaz vetorização nem agrupamento: usa `data/*.pkl` e `clusters/*.pkl` existentes. Refaz rótulos, métricas, SHAP, jornadas por grupo e figuras. Use para reproduzir os resultados do TCC sem chamar a API do Gemini nem baixar o MiniLM. |
| `--rotular-massa` | Antes de tudo, rotula de novo `dataset/processed/total.csv` com o `dataset/gerar_massas.py` (Tabela 1 revisada). Requer o pm4py-cli. |
| `--sem-shap` | Pula o SHAP (não depende dos rótulos; os pesos, os beeswarms e as figuras SHAP ficam como estão). |
| `--sem-revisao` | Pula a comparação v1.0 × v1.1 e a conferência do rótulo (`analises/`). |
| `--petri` | Gera também as Redes de Petri e os DFGs por grupo (requer pm4py e Graphviz; demorado). |

> **Massa nova:** ao gerar outra massa com o `dataset/gerar_massas.py` (playout), o `data/gemini_embeddings.pkl` existente fica desatualizado. O `run_all.py` só gera o Gemini quando o arquivo não existe: apague-o ou gere-o de novo com o comando do Gemini antes de rodar.

Etapas e saídas:

0. (opcional, `--rotular-massa`) `dataset/gerar_massas.py` → `dataset/processed/total-rotulado-v1.1.csv`: rotula todas as sessões pela Tabela 1 revisada, sem usar o cenário que as gerou (jornada simples: gatilho e, depois, a conversão; composta: etapa 1 e, depois, etapa 2). Requer o pm4py-cli (`tools/pm4py_cli`; sem instalar, `PYTHONPATH=tools/pm4py_cli/src`)
1. `src/events2vector.py` → `data/sessions_represented.csv` (sequência de eventos, flags e rótulo principal de cada sessão)
2. Vetorização (`tfidf_vectorizer.py`, `embeddings_vectorizer.py`) e agrupamento (`kmeans_cluster.py`, `hdbscan_cluster.py`) → `data/*.pkl`, `clusters/*.pkl` (os rótulos não entram aqui)
3. `src/eval_metrics.py --labels data/sessions_represented.csv` → `results/metrics/metrics_<modelo>.json`
4. `src/shap_explainer.py` → pesos SHAP de cada grupo nos 6 modelos (`results/jornadas_por_grupo/<modelo>/grupos/grupo_NN/shap_pesos.csv` e `shap_pesos_todos_modelos.csv`); `src/shap_figures.py` → Figuras 4 e 5 (`results/figures/shap_*`) e beeswarm de cada grupo
5. `src/jornadas_por_grupo.py` → `results/jornadas_por_grupo/<modelo>/` (sessões de cada grupo, `resumo_grupos.csv`, `matriz_grupos.png`, `espaco_semantico_tsne.png` e `_3d.png`)
5.1 `src/jornadas_tabelas.py` → `results/dados/jornadas_contagem.csv` e `results/metrics/cobertura_por_jornada.csv`; com `--petri`, `src/petri_por_grupo.py` → `results/jornadas_por_grupo/<modelo>/grupos/grupo_NN/{dfg.png,rede_petri.*}`
6. `analises/rotulo_revisado.py` e `analises/conferir_rotulo_tabela1.py` → comparação v1.0 × v1.1 e conferência do rótulo em Python puro
7. Consolidação → `results/summary_metrics.csv`; `src/figuras_tcc.py` → `results/figures/`

As figuras `comparison_metrics.png`, `matriz_associacao_clusters_labels.png` e `tsne_clusters_visualizacao.png` são geradas por `src/figuras_tcc.py` (na v1.0 não tinham script).

Todo arquivo de `results/` tem script: a lista completa, arquivo por arquivo, está em [docs/resultados.md](docs/resultados.md). Para regenerar a pasta do zero com os agrupamentos do TCC: `python run_all.py --reusar-agrupamentos --petri`.

---

## 🔬 Execução Passo a Passo (Manual)

Se desejar inspecionar ou executar etapas individuais:

1. **Extração de Sequências Textuais:**
   ```bash
   python src/events2vector.py --input dataset/processed/total-rotulado-v1.1.csv --output data/sessions_represented.csv
   ```
2. **Vetorização TF-IDF:**
   ```bash
   python src/tfidf_vectorizer.py --input data/sessions_represented.csv --output data/tfidf_vectors.pkl
   ```
3. **Agrupamento HDBSCAN + TF-IDF:**
   ```bash
   python src/hdbscan_cluster.py --input data/tfidf_vectors.pkl --output clusters/hdbscan_tf_idf.pkl --auto-grid
   ```
4. **Rótulo pela Tabela 1 e avaliação de métricas:**
   ```bash
   python src/eval_metrics.py --input clusters/hdbscan_tf_idf.pkl --output results/metrics/metrics_hdbscan_tf_idf.json --labels data/sessions_represented.csv
   ```
5. **Auditoria Causal SHAP:**
   ```bash
   python src/shap_explainer.py
   ```
6. **Inferência Contínua em Streaming:**
   ```bash
   python src/stream_inference.py
   ```

---

## ⚠️ Limitações conhecidas

- **Massa sintética** gerada por playout de Redes de Petri descobertas de **uma sessão-semente por cenário**, escrita pelo autor. As jornadas compostas ficaram com poucas sessões (005: 12, 009: 15, 010: 6), insuficientes para formar grupo por densidade.
- **33 das 516 sessões começam na home do PIX**: o CSV-semente do cenário 010 tem uma segunda sessão incompleta, e o playout reproduziu esse início. As sessões foram mantidas.
- **MiniLM (`all-MiniLM-L6-v2`) lê no máximo 256 subtokens**: 71% das sessões são cortadas, e o modelo lê cerca de 19 dos 37 eventos de uma sessão típica (`analises/desvios_e_truncamento.py`).
- **TF-IDF e o vetor de presença não consideram a ordem dos eventos.**
- **SHAP explica um modelo substituto** (XGBoost sobre TF-IDF), não o agrupamento diretamente; indica eventos associados aos grupos, não causas (`analises/diag_shap.py`).
- **Pureza por Macro-Experiência** mede só a homogeneidade sucesso × abandono e exclui o ruído; por isso a avaliação inclui a cobertura por jornada.
- **Geração da massa na v1.0 não era reproduzível** (ordem de sorteio do playout); corrigido em `tools/pm4py_cli`. Os resultados da v1.1 são reproduzíveis porque a massa entregue está versionada.

---

## 📖 Citação Acadêmica

Se este código ou dataset for útil em sua pesquisa, por favor cite:

```bibtex
@mastersthesis{ervilha2026experience,
  author       = {Andr{'e} Eduardo Ribeiro Ervilha},
  title        = {Minera{\c{c}}{\~a}o de Jornadas de Experi{\^e}ncia do Usu{'a}rio (Experience Mining): Um Estudo Comparativo de Representa{\c{c}}{\~o}es Vetoriais, Agrupamento N{\~a}o-Supervisionado e Explicabilidade com XAI em Ecossistemas Digitais Multi-Produto},
  school       = {Escola Superior de Agricultura Luiz de Queiroz, Universidade de S{\~a}o Paulo (USP/ESALQ)},
  year         = {2026},
  address      = {Piracicaba, SP, Brasil},
  type         = {Trabalho de Conclus{\~a}o de Curso (Especializa{\c{c}}{\~a}o em Gest{\~a}o de Neg{'o}cios Digitais e Intelig{\^e}ncia Artificial)}
}
```

---

## 📄 Licença

Este projeto é disponibilizado sob a licença [MIT](LICENSE).
