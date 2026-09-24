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
| **HDBSCAN + Gemini** | Gemini (3072D) | HDBSCAN | 21 | 19.4% | 0.3793 | 0.6434 | 0.8355 | **97.68%** 🏆 |
| **HDBSCAN + TF-IDF** | TF-IDF | HDBSCAN | 14 | 26.9% | **0.6477** 🏆 | **0.7469** 🏆 | **0.8886** 🏆 | 95.61% |
| **HDBSCAN + MiniLM** | MiniLM (384D) | HDBSCAN | 26 | 24.0% | 0.5426 | 0.4148 | 0.7027 | 92.68% |
| **K-Means + Gemini** | Gemini (3072D) | K-Means | 6 | 0.0% | 0.2942 | 0.4213 | 0.6173 | 91.65% |
| **K-Means + TF-IDF** | TF-IDF | K-Means | 15 | 0.0% | 0.4923 | 0.5917 | 0.8014 | 91.42% |
| **K-Means + MiniLM** | MiniLM (384D) | K-Means | 4 | 0.0% | 0.4037 | 0.2239 | 0.4882 | 87.24% |

> **Achado de Produto Crítico:** O algoritmo isolou espontaneamente o **Cluster 5 (CDB $\rightarrow$ Fatura)** com 63 sessões e 96,8% de pureza real, evidenciando uma jornada de atrito com 18 passos redundantes onde o usuário resgata renda fixa para quitar o cartão. A pesquisa fundamenta o redesenho (*BPR*) para a funcionalidade de *Resgate Automático com Quitação em 2 Cliques*.

---

## 📂 Arquitetura do Repositório

```
tcc-experience-mining/
├── README.md                      # Documentação oficial
├── requirements.txt               # Dependências pinadas para reproducibilidade
├── run_all.py                     # Orquestrador mestre (1-click reproduction)
├── LICENSE                        # Licença MIT
├── CITATION.cff                   # Citação acadêmica formal
│
├── dataset/                       # Corpus experimental
│   ├── raw/                       # Redes de Petri (.pnml), DFGs e logs sementes das 10 jornadas
│   ├── processed/                 # total-rotulado.csv (516 sessões com ground truth)
│   └── gerar_massas.py            # Gerador estocástico e rotulador via pm4py-cli
│
├── data/                          # Dados vetorizados pré-calculados
│   ├── sessions_represented.csv   # Sequências textuais de eventos
│   ├── tfidf_vectors.pkl          # Vetores TF-IDF
│   ├── minilm_embeddings.pkl      # Embeddings Sentence-BERT (384D)
│   └── gemini_embeddings.pkl      # Embeddings Gemini (3072D)
│
├── src/                           # Pipeline modular em Python
│   ├── events2vector.py           # Converte eventos em sequências textuais
│   ├── tfidf_vectorizer.py        # Vetorização TF-IDF
│   ├── embeddings_vectorizer.py   # Vetorização neural MiniLM e Gemini
│   ├── kmeans_cluster.py          # Agrupamento K-Means com busca de K
│   ├── hdbscan_cluster.py         # Agrupamento HDBSCAN com grid search
│   ├── eval_metrics.py            # Avaliação de métricas (Silhouette, ARI, Pureza)
│   ├── shap_explainer.py          # XAI: XGBoost surrogate e valores SHAP
│   └── stream_inference.py        # Inferência em tempo de execução (<1 ms)
│
├── tools/
│   └── pm4py_cli/                 # CLI própria de Process Mining e Playout
│
├── results/                       # Métricas e saídas computadas
│   ├── summary_metrics.csv        # Tabela comparativa consolidada
│   ├── metrics/                   # JSONs individuais por experimento
│   └── figures/                   # As 5 figuras oficiais em 300 DPI
│
└── paper/                         # Dissertação Oficial em LaTeX
    ├── tcc.tex                    # Código-fonte LaTeX formatado
    ├── tcc_referencias.bib        # Bibliografia em ABNT
    └── tcc.pdf                    # Artigo final compilado
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

Para rodar todo o pipeline de ponta a ponta, regenerar todas as representações, clusterizações, métricas e o modelo SHAP:

```bash
python run_all.py
```

O script executará todos os passos, salvará as métricas em `results/metrics/` e exibirá a tabela final consolidada no terminal.

---

## 🔬 Execução Passo a Passo (Manual)

Se desejar inspecionar ou executar etapas individuais:

1. **Extração de Sequências Textuais:**
   ```bash
   python src/events2vector.py --input dataset/processed/total-rotulado.csv --output data/sessions_represented.csv
   ```
2. **Vetorização TF-IDF:**
   ```bash
   python src/tfidf_vectorizer.py --input data/sessions_represented.csv --output data/tfidf_vectors.pkl
   ```
3. **Agrupamento HDBSCAN + TF-IDF:**
   ```bash
   python src/hdbscan_cluster.py --input data/tfidf_vectors.pkl --output clusters/hdbscan_tf_idf.pkl --auto-grid
   ```
4. **Avaliação de Métricas:**
   ```bash
   python src/eval_metrics.py --input clusters/hdbscan_tf_idf.pkl --output results/metrics/metrics_hdbscan_tf_idf.json
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
