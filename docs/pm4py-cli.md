# `pm4py-cli` — Criação de Massa e Análises de Process Mining

A CLI própria em [`tools/pm4py_cli/`](../tools/pm4py_cli/) implementa descoberta de modelos (Rede de Petri e DFG), simulação estocástica (*playout*), rotulagem por padrões de comportamento e análises de Process Mining sobre o formato tabular do TCC.

O código-fonte fica em [`tools/pm4py_cli/src/jornada_ux/`](../tools/pm4py_cli/src/jornada_ux/). Ao contrário do orquestrador `run_all.py` (guiado por JSON) e de [`dataset/gerar_massas.py`](../dataset/gerar_massas.py) (funções chamadas pela configuração da versão), a `pm4py-cli` **mantém flags CLI próprias** para uso interativo e reprodução isolada.

---

## Como invocar a CLI

Direto do repositório (garante uso exato do código versionado aqui):

```bash
python tools/pm4py_cli/run.py --help
```

Opcionalmente, instale em modo editável na *virtualenv* ativa para habilitar o comando `pm4py-cli`:

```bash
pip install -e tools/pm4py_cli
```

> **Windows / PowerShell (`pyenv-win`):** atividades seguem o padrão `[cd_produto:cd_sub_produto:page_name].cd_evento_tipo(button_text)`. Coloque tokens com `[ ] ( ) :` e o wildcard `...` entre aspas. Se usar `pyenv-win`, evite os shims `.bat` (que corrompem `[ ] ( )`) chamando `python tools/pm4py_cli/run.py`, o executável real `pm4py-cli.exe` ou [`tools/pm4py_cli/run.ps1`](../tools/pm4py_cli/run.ps1).

---

## Fluxo de criação de massa (`semente → modelo → playout → rótulo`)

Quando a configuração da versão usa `events.events_params` e `labeling.labeling_params` (ex.: [`configs/v2.0.json`](../configs/v2.0.json)), o módulo [`dataset/gerar_massas.py`](../dataset/gerar_massas.py) automatiza esse fluxo invocando `tools/pm4py_cli/run.py`:

1. **Inspecionar atividades exatas (`list-activities`)**:
   ```bash
   python tools/pm4py_cli/run.py list-activities \
     --csv "dataset/raw/[001][cartao] pagar fatura com saldo/cartoes - pagar fatura com saldo completo.csv"
   ```
   Saída observada: lista as 23 atividades distintas do cenário (ex.: `[canais:home:home].page_view()`, `[canais:home:home].click_button(menu_produtos)`, `[cartoes:pagamento:comprovante_pagamento].page_view()`).

2. **Descoberta da Rede de Petri e do DFG (`discovery`)**:
   ```bash
   python tools/pm4py_cli/run.py discovery \
     --mode both --algo heuristics \
     --csv "dataset/raw/[001][cartao] pagar fatura com saldo/cartoes - pagar fatura com saldo completo.csv" \
     --output_dir "/tmp/exemplo_cli" --yes
   ```
   Gera em `/tmp/exemplo_cli/discovery/`:
   - `cartoes - pagar fatura com saldo completo-petri.pnml` e `-petri.png`
   - `cartoes - pagar fatura com saldo completo-dfg.dfg` e `-dfg.png`

3. **Simulação estocástica reproduzível (`playout`)**:
   ```bash
   python tools/pm4py_cli/run.py playout \
     --pnml "/tmp/exemplo_cli/discovery/cartoes - pagar fatura com saldo completo-petri.pnml" \
     --csv "dataset/raw/[001][cartao] pagar fatura com saldo/cartoes - pagar fatura com saldo completo.csv" \
     --num_traces 20 --seed 42 --start_session_at 1 \
     --output_dir "/tmp/exemplo_cli" --yes
   ```
   Saída observada: `670` eventos em `20` sessões gravados em `/tmp/exemplo_cli/playout/cartoes - pagar fatura com saldo completo-simulado.csv`.
   - **Determinismo (v2 em diante):** [`_ativar_ordem_deterministica()`](../tools/pm4py_cli/src/jornada_ux/modules/playout.py) ordena as transições habilitadas por nome e rótulo antes do sorteio, garantindo que duas execuções com o mesmo PNML, CSV e `--seed` produzam CSVs idênticos byte a byte.
   - **Massa histórica v1:** a massa entregue em [`dataset/processed/total.csv`](../dataset/processed/total.csv) não pode ser regerada porque essa ordenação não existia na v1.0; por isso `v1.0` e `configs/v1.1.json` carregam `events_csv_path` diretamente.

4. **Rotulagem por padrões (`labeling`)**:
   ```bash
   python tools/pm4py_cli/run.py labeling \
     --csv "/tmp/exemplo_cli/playout/cartoes - pagar fatura com saldo completo-simulado.csv" \
     --pattern "[cartoes:fatura:fatura_atual].click_button(opcoes_pagamento)" "[cartoes:pagamento:comprovante_pagamento].page_view()" \
     --label_col is_fatura_paga_saldo \
     --output_dir "/tmp/exemplo_cli" --yes
   ```
   Saída observada: `Sessões com Match (1): 10 (50.0%)`, `Sessões sem Match (0): 10 (50.0%)`, salvo em `/tmp/exemplo_cli/labeling/cartoes - pagar fatura com saldo completo-simulado-rotulado.csv`. Suporta `--pattern` (*eventually-follows*, vários pares = E), `--not_pattern`, `--contain` (OU) e `--exclude`. No pipeline (`rotular_massa`), todas as regras são aplicadas à massa inteira de uma só vez e derivam `primary_label` e `experience` usando `absorbs_label_cols`.

---

## Comandos de análise sobre a massa do TCC

Todos os exemplos abaixo usam diretamente a massa versionada do TCC ([`dataset/processed/total.csv`](../dataset/processed/total.csv), `516` sessões / `28.786` eventos) gravando em `/tmp/analises_tcc`:

### 1. Conformidade (`analysis conformance`)
Calcula o *Fitness Score* (Token-Based Replay) de cada sessão contra um modelo PNML de referência (ex.: cenário `001`):

```bash
python tools/pm4py_cli/run.py analysis conformance \
  --csv "dataset/processed/total.csv" \
  --pnml "dataset/raw/[001][cartao] pagar fatura com saldo/output/discovery/cartoes - pagar fatura com saldo completo-petri.pnml" \
  --output_dir "/tmp/analises_tcc" --yes
```

* **Saída observada:** `Média do Fitness Score: 0.8976` | `Jornadas perfeitamente conformes: 133 de 516`.
* **Arquivo gerado:** `analysis/conformance/total-conformance.csv`.

### 2. Variantes (`analysis variants`)
Ranqueia as rotas exatas de navegação das sessões:

```bash
python tools/pm4py_cli/run.py analysis variants \
  --csv "dataset/processed/total.csv" --top_n 5 \
  --output_dir "/tmp/analises_tcc" --yes
```

* **Saída observada:** imprime top 5 variantes (ex.: Variante #5 tem `7` casos, `1.36%`) e salva tabela completa em CSV.
* **Arquivo gerado:** `analysis/variants/total-variants.csv`.

### 3. Filtragem (`analysis filter`)
Filtra sessões por sequência *eventually-follows* (`--pattern`, `--not_pattern`), presença (`--contain`, `--exclude`), segmentos (`--trace_include`, `--trace_exclude`), início/fim ou duração/tamanho:

```bash
python tools/pm4py_cli/run.py analysis filter \
  --csv "dataset/processed/total.csv" \
  --pattern "[cartoes:fatura:fatura_atual].click_button(opcoes_pagamento)" "[cartoes:pagamento:comprovante_pagamento].page_view()" \
  --output_dir "/tmp/analises_tcc" --yes
```

* **Saída observada:** `Total de casos (sessões) inicial: 516` → `pós-filtragem: 104` (`Redução de 79.8%`).
* **Arquivo gerado:** `analysis/filter/total-filtrado.csv`.

### 4. Funil de conversão (`funnel`)
Descobre o caminho mais provável no DFG (ou recebe etapas explícitas via `--steps`) e calcula retenção, *drop-off* e tempos entre etapas:

```bash
python tools/pm4py_cli/run.py funnel \
  --csv "dataset/processed/total.csv" \
  --output_dir "/tmp/analises_tcc" --yes
```

* **Arquivos gerados:** `funnel/total-funil.csv` e `funnel/total-funil.png`.

### 5. Loops (`analysis loops`)
Identifica atividades repetidas dentro da mesma sessão e monta o DFG das transições de loop:

```bash
python tools/pm4py_cli/run.py analysis loops \
  --csv "dataset/processed/total.csv" --min_repetitions 2 --top_k 10 \
  --output_dir "/tmp/analises_tcc" --yes
```

* **Saída observada:** top atividade `[cartoes:home_cartoes:home_cartoes].page_view()` (`283 sessões | 1532 revisitas | máx 28x`); DFG de loops com `125 arestas, peso total = 27856`.
* **Arquivos gerados:** `analysis/loops/total-loops.csv`, `total-loops-detalhe.csv`, `total-loops-dfg.png` e `total-loops-dfg.dfg`.

### 6. Retrabalho (`analysis rework`)
Calcula a taxa de retrabalho por atividade via PM4Py nativo:

```bash
python tools/pm4py_cli/run.py analysis rework \
  --csv "dataset/processed/total.csv" \
  --output_dir "/tmp/analises_tcc" --yes
```

* **Saída observada:** destaca `[investimentos:resgate:detalhe_posicao_cdb].page_view()` (`92.96%`, em `66 de 71 casos`, `532` ocorrências) entre outras atividades.
* **Arquivo gerado:** `analysis/rework/total-rework.csv`.

### 7. N-gramas (`analysis ngrams`)
Extrai os N-gramas contíguos mais frequentes e gera o DFG combinado dos top K:

```bash
python tools/pm4py_cli/run.py analysis ngrams \
  --csv "dataset/processed/total.csv" --n 2 --top_k 10 \
  --output_dir "/tmp/analises_tcc" --yes
```

* **Saída observada:** bigramas como `[canais:home:menu_produtos].click_button(cartoes) -> [cartoes:home_cartoes:home_cartoes].page_view()` (`809x | 63.0% sessões`); DFG combinado com `10 arestas, peso total = 9170`.
* **Arquivos gerados:** `analysis/ngrams/total-ngrams.csv`, `total-ngrams-dfg.png` (quando todas as arestas formam ciclo sem início isolado, o exportador `.dfg` registra o aviso correspondente sem impedir o CSV/PNG).

---

## Outros subcomandos disponíveis

| Subcomando | Função |
| :--- | :--- |
| `backbone` | Simplifica a Rede de Petri mantendo as arestas mais frequentes (`--top_pct`, `--top_n`, `--min_weight`) |
| `analysis performance` | Mapa de calor de tempos médios/medianos entre transições (`Performance DFG`) |
| `analysis edge-weights` | Distribuição de pesos das arestas com detecção de cotovelo |
| `features` | Matriz tabular de *features* por sessão (*Bag of Words* / bigramas) |
