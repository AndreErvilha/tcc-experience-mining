# Jornada UX - CLI

Ferramenta profissional de Linha de Comando (CLI) para modelagem, simulação (playout) e análise de jornadas de usuários utilizando técnicas avançadas de **Process Mining** (com PM4Py).

Desenvolvido como projeto de TCC para estruturação, análise e validação científica de perfis de jornadas de clientes (sucesso, fricção, abandono).

---

## 🚀 Estrutura do CLI

O aplicativo está estruturado de forma modular com três grandes comandos principais no terminal:

1. **`discovery`**: Gera Redes de Petri e DFGs a partir de logs de navegação (Inductive, Alpha, Heuristics Miner).
2. **`backbone`**: Extrai a Rede de Petri simplificada apenas com as transições de maior peso (espinha dorsal do processo).
3. **`funnel`** (ou `funil`): Calcula a retenção, taxas de conversão entre etapas, desistências (drop-offs) e tempos de navegação por passo do funil.
4. **`labeling`** (ou `rotular`): Classifica sessões em bases não rotuladas atribuindo rótulos comportamentais heurísticos (`Converted`, `Abandoned`, `Bounced`, `Loop_Friction`).
5. **`features`**: Extrai uma matriz tabular de características por sessão (Bag of Words, Bigramas DFG e métricas de complexidade) para alimentar algoritmos não supervisionados (K-Means, DBSCAN, etc.).
6. **`playout`**: Simula novas jornadas de usuário a partir de um modelo Petri Net (Stochastic Playout).
7. **`analysis`** (com alias em português **`analise`**): Sub-suíte especializada que agrupa ferramentas focadas de diagnóstico e métricas:
   * `analysis conformance`: Análise de conformidade (Token-based Replay) com cálculo de Fitness Score por sessão.
   * `analysis performance`: Análise de tempos e geração de mapas de calor de gargalos (Performance DFG).
   * `analysis variants`: Mapeamento e ranking dos caminhos de navegação mais comuns.
   * `analysis filter`: Limpeza e filtragem dinâmica de logs de eventos.
   * `analysis rework`: Estatísticas de retrabalho (*rework*) e atrito por tela (loops) via PM4Py nativo.
   * `analysis ngrams`: Extrai e ranqueia N-gramas (sequências contíguas de N passos) mais frequentes, com DFG visual.
   * `analysis loops`: Detecta atividades com loops (repetições na mesma sessão) e gera o DFG das transições de fricção.
   * `analysis funnel`: Funil de conversão da jornada.
   * `analysis labeling`: Classificação e rotulagem comportamental.
   * `analysis features`: Extração de vetores para Machine Learning.

---

## 🛠️ Instalação (Modo Desenvolvimento)

Para instalar a ferramenta localmente de forma que o comando `pm4py-cli` fique disponível globalmente no seu terminal:

1. Ative seu ambiente virtual (venv):
   ```bash
   # No Windows (PowerShell)
   .venv/Scripts/Activate.ps1
   ```
2. Instale o pacote em modo de edição (`-e`):
   ```bash
   pip install -e .
   ```

Pronto! Agora você pode digitar `pm4py-cli` (ou utilizar `python run.py`) diretamente no terminal.

---

## 📖 Como Usar (Formatos de Saída e Exemplos)

O CLI permite definir o diretório base de saída (`--output_dir` ou `-o`) e o nome base do arquivo (`--output_base`). Os arquivos gerados são automaticamente organizados em subpastas referentes ao módulo executado no padrão: `<output_dir>/<modulo>/<output_base>-<tipo>.<extensão>`.

### 1. Descobrir Processos (`discovery`)
```bash
python run.py discovery --csv example/tagueamento.csv --output_dir example
```
* **Gera em `example/discovery/`:**
  - `tagueamento-petri.png` e `tagueamento-petri.pnml`
  - `tagueamento-dfg.png` e `tagueamento-dfg.dfg`

### 2. Espinha Dorsal do Processo (`backbone`)
```bash
# Manter arestas até acumular 50% do peso total do DFG
python run.py backbone --csv example/tagueamento.csv --top_pct 50 --output_dir example

# Manter apenas as 5 arestas de maior peso
python run.py backbone --csv example/tagueamento.csv --top_n 5 --output_dir example

# Manter arestas com ao menos 3 ocorrências
python run.py backbone --csv example/tagueamento.csv --min_weight 3 --output_dir example
```
* **Gera em `example/backbone/`:**
  - `tagueamento-backbone.pnml` (Rede de Petri simplificada)
  - `tagueamento-backbone.png` (Visualização da espinha dorsal)

### 3. Funil de Conversão (`funnel`)
```bash
# Descoberta heurística automática do funil principal
python run.py funnel --csv example/tagueamento.csv --output_dir example

# Ou especificando passos customizados da jornada
python run.py funnel --csv example/tagueamento.csv --steps "home" "hub_cartoes" "form_dados_pessoais" "sucesso" --output_dir example
```
* **Gera em `example/funnel/`:**
  - `tagueamento-funil.csv` (Tabela do funil)
  - `tagueamento-funil.png` (Gráfico visual do funil)

### 4. Rotulagem Heurística de Casos (`labeling`)
```bash
python run.py labeling --csv example/tagueamento.csv --output_dir example
```
* **Gera em `example/labeling/`:** `tagueamento-rotulado.csv` (Casos classificados em Converted, Abandoned, Bounced, Loop_Friction, etc.)

### 5. Extração de Features para Machine Learning (`features`)
```bash
python run.py features --csv example/tagueamento.csv --output_dir example
```
* **Gera em `example/features/`:** `tagueamento-features_ml.csv` (Matriz tabular de atributos por sessão)

### 6. Simular Processos (`playout`)
```bash
python run.py playout --pnml example/discovery/tagueamento-petri.pnml --csv example/tagueamento.csv --output_dir example --num_traces 200 --seed 42
```
* **Gera em `example/playout/`:**
  - `tagueamento-simulado.csv`

### 7. Analisar Jornadas (`analysis` ou `analise`)
- **Conformidade (Fitness Score por sessão):**
  ```bash
  python run.py analysis conformance --csv example/tagueamento.csv --pnml example/discovery/tagueamento-petri.pnml --output_dir example
  ```
  * **Gera em `example/analysis/conformance/`:** `tagueamento-conformance.csv`

- **Mapa de calor de performance:**
  ```bash
  python run.py analise performance --csv example/tagueamento.csv --output_dir example
  ```
  * **Gera em `example/analysis/performance/`:** `tagueamento-performance.png`

- **Ranking de variantes:**
  ```bash
  python run.py analysis variants --csv example/tagueamento.csv --output_dir example
  ```
  * **Gera em `example/analysis/variants/`:** `tagueamento-variants.csv`

- **Análise de retrabalho / loops (PM4Py nativo):**
  ```bash
  python run.py analysis rework --csv example/tagueamento.csv --output_dir example
  ```
  * **Gera em `example/analysis/rework/`:** `tagueamento-rework.csv`

- **N-gramas de navegação (com DFG visual):**
  ```bash
  # Bigramas mais frequentes
  python run.py analysis ngrams --csv example/tagueamento.csv --n 2 --output_dir example

  # Trigramas com corte mínimo de 2 ocorrências
  python run.py analysis ngrams --csv example/tagueamento.csv --n 3 --min_count 2 --output_dir example
  ```
  * **Gera em `example/analysis/ngrams/`:**
    - `tagueamento-ngrams.csv` (Ranking de n-gramas)
    - `tagueamento-ngrams-dfg.png` (DFG visual dos n-gramas)
    - `tagueamento-ngrams-dfg.dfg` (DFG serializado)

- **Loops por atividade (com DFG de fricção):**
  ```bash
  python run.py analysis loops --csv example/tagueamento.csv --output_dir example
  ```
  * **Gera em `example/analysis/loops/`:**
    - `tagueamento-loops.csv` (Ranking de atividades com loop)
    - `tagueamento-loops-detalhe.csv` (Detalhamento por sessão)
    - `tagueamento-loops-dfg.png` (DFG visual das transições de loop)
    - `tagueamento-loops-dfg.dfg` (DFG serializado)

- **Filtragem de logs:**
  ```bash
  python run.py analysis filter --csv example/tagueamento.csv --start_activities home_banco --output_dir example
  ```
  * **Gera em `example/analysis/filter/`:** `tagueamento-filtrado.csv`

---

## ⚠️ Windows / PowerShell: caracteres especiais

Os nomes de atividade usados nos filtros têm este formato e **contêm caracteres especiais**:

```
[cd_produto:cd_sub_produto:page_name].cd_evento_tipo(button_text)
```

Ex.: `[canais:home:home].click_button(menu_produtos)`.

No PowerShell/cmd, `[ ] ( ) :` e o wildcard `...` precisam ficar **entre aspas**, senão o shell tenta interpretá-los e o comando quebra (erro como `... foi inesperado neste momento`). Exemplo correto:

```powershell
python run.py analysis filter --csv ".\output\playout\009-simulado.csv" --trace_include "..." "[canais:home:home].click_button(menu_produtos)" "..." "[investimentos:resgate:input_valor_resgate].page_view()" "..."
```

### Se você usa pyenv-win

No pyenv-win, `python` e `pm4py-cli` são *shims* `.bat` que repassam os argumentos pelo `cmd.exe`, o que **corrompe** os `[ ] ( )` e faz o filtro retornar vazio (ou dar erro). Nesse caso, rode pelo executável real ou pelo helper do projeto:

1. Executável real (sem o `.bat`):

   ```powershell
   & "C:\Users\<seu-usuario>\.pyenv\pyenv-win\versions\<versao>\Scripts\pm4py-cli.exe" analysis filter ...
   ```

   (encontre o caminho exato com `pyenv which python` e use a pasta `Scripts` correspondente).

2. Helper `run.ps1` (já incluído no projeto), que localiza o python real sozinho:

   ```powershell
   .\run.ps1 analysis filter ...
   ```

---

## 🔒 Segurança de Dados (Proteção de Sobrescrita)

O CLI possui um sistema inteligente de segurança que verifica o diretório de destino antes de salvar. Caso algum arquivo resultante já exista, o CLI solicitará a confirmação do usuário (`y/n`) no terminal, evitando perdas acidentais de dados.
