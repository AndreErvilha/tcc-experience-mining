# Guia de Introdução (Getting Started) - `pm4py-cli`

Bem-vindo ao **`pm4py-cli`**! Esta é uma ferramenta profissional de linha de comando desenvolvida em Python para modelagem, simulação e análises estatísticas de **jornadas de experiência do usuário (UX)** utilizando as bibliotecas nativas de **Process Mining (PM4Py)**.

A ferramenta foi projetada de forma modular e estruturada em três grandes áreas lógicas:

1. **`discovery`**: Mapeia logs brutos e gera Redes de Petri e DFGs.
2. **`playout`**: Simula novas jornadas de usuários (playout estocástico).
3. **`analysis`** (ou o atalho em português **`analise`**): Uma sub-suíte que agrupa todas as ferramentas de diagnóstico, tempos, filtros e conformidade de jornadas.

---

## 🛠️ Instalação Rápida

1. Abra o terminal na pasta do projeto:
   ```bash
   cd "experimentos/jornada UX/jornada_ux_cli"
   ```
2. Ative o ambiente virtual criado para o projeto:
   ```powershell
   # No Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```
3. Instale a ferramenta no ambiente em modo editável (`-e`):
   ```bash
   pip install -e .
   ```
4. Verifique a instalação:
   ```bash
   pm4py-cli --help
   # ou usando diretamente:
   python run.py --help
   ```

---

## ⚙️ Formatação e Estrutura de Saída

O CLI permite selecionar a pasta de destino (`--output_dir` ou `-o`) e o nome do arquivo (`--output_base`) separadamente.
Ao definir uma pasta de saída como `--output_dir example` e utilizar como entrada o arquivo `tagueamento.csv`, o CLI criará automaticamente a estrutura organizada por módulo:

- **Discovery**: `example/discovery/tagueamento-petri.png`, `tagueamento-petri.pnml`, `tagueamento-dfg.png`, `tagueamento-dfg.dfg`
- **Playout**: `example/playout/tagueamento-simulado.csv`
- **Conformidade**: `example/analysis/conformance/tagueamento-conformance.csv`
- **Performance**: `example/analysis/performance/tagueamento-performance.png`
- **Variantes**: `example/analysis/variants/tagueamento-variants.csv`
- **Retrabalho (Rework)**: `example/analysis/rework/tagueamento-rework.csv`
- **Filtro**: `example/analysis/filter/tagueamento-filtrado.csv`

Você também pode fornecer o caminho exato de saída através de `--output_csv` ou `--output_png`, sobrescrevendo o caminho padrão.

---

## 🚀 Guia Prático com Exemplos

Todas as ferramentas podem ser testadas utilizando o arquivo de exemplo contido em `example/tagueamento.csv`. Abaixo está o passo a passo completo de como extrair valor de cada módulo.

### 1. Descoberta de Processos (`discovery`)
Gera a estrutura visual e matemática da jornada do cliente em formato vertical (Rede de Petri e Grafo de Transição Direta - DFG).

* **Comando:**
  ```bash
  pm4py-cli discovery --csv example/tagueamento.csv --output_dir example
  ```
* **O que gera na pasta `example/discovery/`:**
  * `tagueamento-petri.png` (Grafo visual vertical da Rede de Petri)
  * `tagueamento-petri.pnml` (Estrutura matemática do modelo para importação no ProM/simuladores)
  * `tagueamento-dfg.png` (Grafo de transições diretas simplificado)
  * `tagueamento-dfg.dfg` (Arquivo de serialização bruto do DFG)

---

### 2. Simulação e Geração de Dados (`playout`)
Gera jornadas simuladas realistas baseadas na Rede de Petri e nas frequências observadas no CSV semente.

* **Comando (Gera 100 jornadas simuladas):**
  ```bash
  pm4py-cli playout --pnml example/discovery/tagueamento-petri.pnml --csv example/tagueamento.csv --output_dir example --num_traces 100 --seed 42
  ```
* **O que gera em `example/playout/`:**
  * `tagueamento-simulado.csv` (Dataset tabular contendo os eventos simulados formatados nas colunas originais do log).

---

### 3. Análise de Conformidade (`analysis conformance`)
Calcula a aderência (**Fitness Score**) de cada sessão individual de um log de eventos em relação ao modelo de Rede de Petri de referência (Caminho Feliz).

* **Comando:**
  ```bash
  pm4py-cli analysis conformance --csv example/tagueamento.csv --pnml example/discovery/tagueamento-petri.pnml --output_dir example
  ```
  *(Ou alternativamente: `pm4py-cli analise conformance ...`)*
* **O que gera em `example/analysis/conformance/`:**
  * `tagueamento-conformance.csv` (Tabela com Fitness Score por sessão de 0.0 a 1.0).

---

### 4. Análise de Desempenho e Tempos (`analysis performance`)
Calcula a diferença de tempo real entre os cliques do usuário e gera o mapa de calor de performance (Performance DFG), destacando visualmente onde estão os maiores atritos temporais na jornada.

* **Comando (Exibindo a média das durações de transição):**
  ```bash
  pm4py-cli analysis performance --csv example/tagueamento.csv --aggregation mean --output_dir example
  ```
* **O que gera em `example/analysis/performance/`:**
  * `tagueamento-performance.png` (DFG colorido com o mapa de tempos entre telas/botões).

---

### 5. Estatísticas de Loops e Atrito (`analysis rework`)
Identifica estatisticamente quais telas possuem maior taxa de retrabalho (*rework*), ou seja, onde os usuários retornam ou entram em loops.

* **Comando:**
  ```bash
  pm4py-cli analysis rework --csv example/tagueamento.csv --output_dir example
  ```
* **O que gera em `example/analysis/rework/`:**
  * `tagueamento-rework.csv` (Ranking de telas com maior taxa de retrabalho).

---

### 6. Ranking de Rotas mais Comuns (`analysis variants`)
Mapeia e agrupa os usuários pelos caminhos exatos de navegação tomados, exibindo o ranking no terminal e salvando o resultado.

* **Comando:**
  ```bash
  pm4py-cli analysis variants --csv example/tagueamento.csv --top_n 5 --output_dir example
  ```
* **O que gera em `example/analysis/variants/`:**
  * Ranking no terminal e `tagueamento-variants.csv` com volume de casos e percentual de ocorrência.

---

### 7. Pré-processamento e Filtragem (`analysis filter`)
Filtra e limpa dinamicamente logs mantendo apenas jornadas que atendam a critérios de início, fim ou duração.

* **Comando (Mantém jornadas que iniciam em 'home_banco'):**
  ```bash
  pm4py-cli analysis filter --csv example/tagueamento.csv --start_activities home_banco --output_dir example
  ```
* **O que gera em `example/analysis/filter/`:**
  * `tagueamento-filtrado.csv` (Dataset limpo mantendo o formato original).

---

## ⚠️ Windows / PowerShell: caracteres especiais

Os nomes de atividade dos filtros contêm `[ ] ( ) :` (ex.: `[canais:home:home].click_button(menu_produtos)`) e o wildcard `...`. No PowerShell/cmd, **coloque cada um entre aspas**:

```powershell
python run.py analysis filter --csv ".\output\playout\009-simulado.csv" --trace_include "..." "[canais:home:home].click_button(menu_produtos)" "..." "[investimentos:resgate:input_valor_resgate].page_view()" "..."
```

Se você usa **pyenv-win**, `python`/`pm4py-cli` são shims `.bat` que corrompem esses caracteres (o filtro volta vazio ou dá erro `... foi inesperado neste momento`). Rode pelo executável real ou pelo helper:

```powershell
# executável real (sem o .bat)
& "C:\Users\<seu-usuario>\.pyenv\pyenv-win\versions\<versao>\Scripts\pm4py-cli.exe" analysis filter ...

# ou o helper do projeto (localiza o python real sozinho)
.\run.ps1 analysis filter ...
```

---

## 🔒 Segurança Contra Sobrescrita

Todos os subcomandos que salvam arquivos contam com o sistema de **Overwriting Protection**. Se um arquivo de saída já existir na pasta de destino, o CLI solicitará sua confirmação (`y/n`) no terminal antes de efetuar alterações.
