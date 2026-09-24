# ðŸ“‹ Backlog de RefatoraÃ§Ã£o e PublicaÃ§Ã£o: `pm4py-tools` (antigo `jornada_ux_cli`)

> **Objetivo:** Transformar o utilitÃ¡rio CLI atual em uma ferramenta profissional, modular e declarativa (baseada em YAML e comandos atÃ´micos), orientada a estender o ecossistema **PM4Py** para descoberta, simulaÃ§Ã£o, consolidaÃ§Ã£o e rotulagem de logs de processos/jornadas sem reinventar a roda.

---

## ðŸŽ¯ VisÃ£o do Produto
* **Nome do Pacote:** `pm4py-tools` (ou `pm4py-flow`)
* **Comando no Terminal:** `pm4py-tools` (com alias `pm4py-cli`)
* **Proposta de Valor:** CLI utilitÃ¡rio e orquestrador declarativo em YAML para quem utiliza PM4Py no dia a dia.

---

##  epic 1: Orquestrador Declarativo de Pipeline (YAML)
Permitir a execuÃ§Ã£o de pipelines completos de ponta a ponta a partir de um Ãºnico arquivo `.yaml`.

- [ ] **US1.1: EspecificaÃ§Ã£o do Schema YAML**
  - Definir estrutura formal em YAML com suporte a etapas sequenciais (`pipeline: [ {step: ...}, ... ]`).
  - Suporte a variÃ¡veis de ambiente, tags e modo de execuÃ§Ã£o seletiva por step.
- [ ] **US1.2: ImplementaÃ§Ã£o do `PipelineRunner` (`pm4py-tools run pipeline.yaml`)**
  - MÃ³dulo responsÃ¡vel por carregar o YAML, validar as configuraÃ§Ãµes de cada etapa e disparar a execuÃ§Ã£o atÃ´mica em sequÃªncia.
  - Implementar flag `--dry-run` para simular os passos e exibir o plano de execuÃ§Ã£o sem modificar arquivos.
  - Implementar flag `--steps` para executar apenas etapas especÃ­ficas (ex: `--steps merge labeling`).
- [ ] **US1.3: MigraÃ§Ã£o dos 10 CenÃ¡rios do TCC para `pipeline.yaml`**
  - Converter a matriz `CENARIOS_CONFIG` do script [gerar_massas.py](file:///C:/Users/andre/IdeaProjects/pessoal/TCC_vs3/experimentos/dataset/v1/gerar_massas.py) para o novo formato declarativo.

---

## ðŸ§± Epic 2: Comandos AtÃ´micos e Desacoplamento dos Passos
Garantir que cada comando do CLI execute uma Ãºnica responsabilidade de forma limpa e independente.

- [ ] **US2.1: Novo Subcomando `merge` (ConsolidaÃ§Ã£o de Event Logs)**
  - Comando CLI: `pm4py-tools merge --inputs "dataset/v1/**/playout/*-simulado.csv" --output dataset/v1/total/total.csv`
  - Funcionalidades:
    - ConcatenaÃ§Ã£o de mÃºltiplos logs de eventos CSV.
    - ReindexaÃ§Ã£o contÃ­nua de IDs de sessÃ£o (`cd_sessao` / `case_id`) para evitar colisÃ£o entre cenÃ¡rios.
    - Filtro de qualidade opcional (`--min_events N` por sessÃ£o).
- [ ] **US2.2: Isolamento do `playout` e `playout_batch`**
  - Playout focado exclusivamente em simulaÃ§Ã£o estocÃ¡stica via Rede de Petri (`.pnml`).
  - CriaÃ§Ã£o do modo em lote (`playout_batch`) para rodar mÃºltiplos cenÃ¡rios em paralelo ou sequÃªncia a partir do YAML.
- [ ] **US2.3: Isolamento do `labeling` sobre Bases Consolidadas**
  - ExecuÃ§Ã£o de rotulagem como etapa autÃ´noma (apenas lÃª CSV consolidado de entrada e grava novo CSV rotulado de saÃ­da).
  - Suporte a mÃºltiplas regras em lote no mesmo comando ou etapa do YAML.
- [ ] **US2.4: PadronizaÃ§Ã£o dos Comandos Principais**
  - Garantir ergonomia nos 5 comandos de nÃºcleo:
    1. `discovery`: Inductive, Heuristics, Alpha Miner $\rightarrow$ DFG & Petri Net.
    2. `playout`: Stochastic Simulation $\rightarrow$ Event Log CSV.
    3. `merge`: UnificaÃ§Ã£o de logs e controle de sessÃµes.
    4. `filter`: Filtragem por conformidade, tempo, padrÃ£o ou tamanho de sessÃ£o.
    5. `labeling`: CriaÃ§Ã£o de flags binÃ¡rias 0/1 com regras de negÃ³cio.
    6. `list-activities`: DiagnÃ³stico rÃ¡pido do vocabulÃ¡rio do log.

---

## ðŸ§¹ Epic 3: HigienizaÃ§Ã£o da Base de CÃ³digo e FlexibilizaÃ§Ã£o
Preparar o cÃ³digo para consumo pÃºblico e repositÃ³rios abertos.

- [ ] **US3.1: FlexibilizaÃ§Ã£o dos Nomes de Colunas (`utils.py`)**
  - Permitir mapear os nomes das colunas via flags CLI ou YAML (`--case_id`, `--activity_col`, `--timestamp_col`).
  - Usar os nomes do PM4Py como fallback inteligente (`case:concept:name`, `concept:name`, `time:timestamp`).
  - Manter compatibilidade com os nomes do TCC (`cd_sessao`, `page_name`, etc.) como default customizado.
- [ ] **US3.2: Limpeza de Arquivos Locais e `.gitignore`**
  - Remover pastas de execuÃ§Ãµes legadas (`output/009`, arquivos temporÃ¡rios de teste).
  - Garantir `.gitignore` robusto ignorando `.venv/`, `.codewhale/`, `output/`, `*.egg-info`, `.pytest_cache/`.
- [ ] **US3.3: RemoÃ§Ã£o de CÃ³digo Morto em `gerar_massas.py` e MÃ³dulos**
  - Limpar blocos comentados e variÃ¡veis nÃ£o utilizadas.

---

## ðŸ“¦ Epic 4: Empacotamento, Testes e PublicaÃ§Ã£o (PyPI & GitHub)

- [ ] **US4.1: ConfiguraÃ§Ã£o do `pyproject.toml`**
  - Atualizar metadados do projeto:
    - `name = "pm4py-tools"`
    - `description = "Declarative CLI utilities and YAML pipeline runner for PM4Py"`
    - `authors`, `license = "MIT"`, `readme = "README.md"`, `keywords = ["process-mining", "pm4py", "cli", "user-journeys"]`
    - `scripts`: `pm4py-tools = "jornada_ux.main:main"` e alias `pm4py-cli = "jornada_ux.main:main"`.
- [ ] **US4.2: SuÃ­te de Testes Automatizados (`pytest`)**
  - Criar pasta `tests/` com testes rÃ¡pidos ponta a ponta cobrindo:
    - `test_discovery.py`
    - `test_playout.py`
    - `test_merge.py`
    - `test_labeling.py`
    - `test_pipeline_yaml.py`
- [ ] **US4.3: AtualizaÃ§Ã£o do `README.md`**
  - DocumentaÃ§Ã£o em inglÃªs e portuguÃªs com:
    - InstalaÃ§Ã£o via `pip install pm4py-tools`.
    - Exemplos de comandos CLI rÃ¡pidos.
    - Exemplo de arquivo `pipeline.yaml` de ponta a ponta.
    - CrÃ©ditos e integraÃ§Ã£o clara com a biblioteca `pm4py`.

