# gerar_massas.py
# Gera a massa sintética (descoberta -> playout -> consolidação) e rotula cada sessão.
# Só rotular a massa existente (dataset/processed/total.csv), sem gerar sessões novas:
#   python dataset/gerar_massas.py --skip_discovery_original --skip_playout --skip_discovery_simulado \
#       --usar_total_existente --saida <arquivo>.csv
#   -> dataset/processed/<arquivo>.csv
# Na revisão pós-entrega (v1.1), rodamos esse comando com --saida total-rotulado-v1.1.csv
# (o run_all.py --rotular-massa faz o mesmo).
# Requer o pm4py-cli (tools/pm4py_cli): instalado, ou PYTHONPATH=tools/pm4py_cli/src.
#
# Rótulos (Tabela 1 revisada, a partir do modelo de cada cenário):
#   - toda conversão que os modelos permitem é uma jornada; as compostas são sequências delas;
#   - jornada simples: sucesso = gatilho e, depois dele, a conversão; abandono = gatilho sem a conversão depois;
#   - jornada composta (005, 009, 010): sucesso = etapa 1 e, depois dela, a etapa 2; abandono = concluiu a
#     etapa 1, começou a etapa 2 (005 e 009: abriu as opções de pagamento depois) e não a concluiu;
#   - 011 (PIX copia e cola avulso) e 012 (Open Finance avulso) são os caminhos simples que os modelos
#     da 010 e da 005 permitem; ficam no cenário de origem.
#   "pattern": pares [A, B] passados ao pm4py-cli como --pattern A B (eventually-follows); vários pares = "E".
#   "not_pattern": pares que NÃO podem ocorrer. "contain": lista "OU".
# Versão entregue (v1.0): o sucesso usava só o evento de conversão (a 009 ficava igual à 001) e o
# abandono não verificava o gatilho. Ver a errata.
# (O playout da v1.0 não era determinístico com --seed; corrigido em tools/pm4py_cli/.../playout.py.)
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd

# Diretório base dos datasets (onde este script reside)
BASE_DIR = Path(__file__).resolve().parent

# Configuração detalhada de cada cenário / dataset com nomes exatos extraídos via list-activities
# - folder: nome da pasta do cenário dentro de dataset/raw
# - csv: nome do arquivo CSV de entrada (None para detecção automática)
# - label_col: nome da coluna binária criada pelo labeling
# - contain: lista de termos/atividades exatas para match positivo
# - exclude: lista de termos/atividades para exclusão (omitido se vazio/None)
CENARIOS_CONFIG = [
    {
        "id": "001",
        "folder": "[001][cartao] pagar fatura com saldo",
        "csv": "cartoes - pagar fatura com saldo completo.csv",
        "num_traces": 100,
        "min_events": 8,
        "labels": [
            {
                # 001 · Pagar fatura com saldo: gatilho e, depois dele, a conversão
                "label_col": "is_fatura_paga_saldo",
                "pattern": [
                    ["[cartoes:fatura:fatura_atual].click_button(opcoes_pagamento)",
                     "[cartoes:pagamento:comprovante_pagamento].page_view()"]
                ],
            },
            {
                # 001 · abandono: gatilho sem a conversão depois dele
                "label_col": "is_iniciou_pagar_saldo_sem_converter",
                "contain": ["[cartoes:fatura:fatura_atual].click_button(opcoes_pagamento)"],
                "not_pattern": [
                    ["[cartoes:fatura:fatura_atual].click_button(opcoes_pagamento)",
                     "[cartoes:pagamento:comprovante_pagamento].page_view()"]
                ],
            },
        ],
    },
    {
        "id": "002",
        "folder": "[002][cartao] pagar fatura copiar pix",
        "csv": "cartoes - pagar fatura copiar pix.csv",
        "num_traces": 100,
        "min_events": 7,
        "labels": [
            {
                # 002 · Copiar o PIX da fatura: gatilho e, depois dele, a conversão
                "label_col": "is_pix_copiado",
                "pattern": [
                    ["[cartoes:pagamento:opcoes_pagamento].click_button(copiar_pix)",
                     "[cartoes:pagamento:sucesso_copia].page_view()"]
                ],
            },
            {
                # 002 · abandono: gatilho sem a conversão depois dele
                "label_col": "is_iniciou_copiar_pix_sem_converter",
                "contain": ["[cartoes:pagamento:opcoes_pagamento].click_button(copiar_pix)"],
                "not_pattern": [
                    ["[cartoes:pagamento:opcoes_pagamento].click_button(copiar_pix)",
                     "[cartoes:pagamento:sucesso_copia].page_view()"]
                ],
            },
        ],
    },
    {
        "id": "003",
        "folder": "[003][cartao] pagar fatura parcelar",
        "csv": "cartoes - pagar fatura parcelar.csv",
        "num_traces": 100,
        "min_events": 10,
        "labels": [
            {
                # 003 · Parcelar a fatura: gatilho e, depois dele, a conversão
                "label_col": "is_fatura_parcelada",
                "pattern": [
                    ["[cartoes:fatura:fatura_atual].click_button(simular_parcelamento)",
                     "[cartoes:parcelamento:comprovante_parcelamento].page_view()"]
                ],
            },
            {
                # 003 · abandono: gatilho sem a conversão depois dele
                "label_col": "is_iniciou_parcelamento_sem_converter",
                "contain": ["[cartoes:fatura:fatura_atual].click_button(simular_parcelamento)"],
                "not_pattern": [
                    ["[cartoes:fatura:fatura_atual].click_button(simular_parcelamento)",
                     "[cartoes:parcelamento:comprovante_parcelamento].page_view()"]
                ],
            },
        ],
    },
    {
        "id": "004",
        "folder": "[004][cartao] historico de fatura cartao",
        "csv": "cartoes - historico de fatura completo.csv",
        "num_traces": 100,
        "min_events": 7,
        "labels": [
            {
                # 004 · Histórico de faturas em PDF: gatilho e, depois dele, a conversão
                "label_col": "is_historico_visualizado",
                "pattern": [
                    ["[cartoes:historico:historico_faturas].click_button(mes_anterior)",
                     "[cartoes:historico:visualizacao_pdf].page_view()"]
                ],
            },
            {
                # 004 · abandono: gatilho sem a conversão depois dele
                "label_col": "is_iniciou_historico_sem_converter",
                "contain": ["[cartoes:historico:historico_faturas].click_button(mes_anterior)"],
                "not_pattern": [
                    ["[cartoes:historico:historico_faturas].click_button(mes_anterior)",
                     "[cartoes:historico:visualizacao_pdf].page_view()"]
                ],
            },
        ],
    },
    {
        "id": "005",
        "folder": "[005][cartao-openfinance] trazer dinheiro e pagar fatura cartao",
        "csv": "005.csv",
        "num_traces": 100,
        "min_events": 15,
        "labels": [
            {
                # 005 · Trazer dinheiro (Open Finance) e pagar a fatura (composta): etapa 1 e, depois dela, a etapa 2
                "label_col": "is_openfinance_fatura_paga",
                "pattern": [
                    ["[open_finance:home:home_open_finance].click_button(trazer_dinheiro)",
                     "[open_finance:trazer_dinheiro:comprovante_pix_open_finance].page_view()"],
                    ["[open_finance:trazer_dinheiro:comprovante_pix_open_finance].page_view()",
                     "[cartoes:pagamento:comprovante_pagamento].page_view()"]
                ],
            },
            {
                # 005 · abandono: trouxe o dinheiro, abriu as opções de pagamento depois e não pagou
                "label_col": "is_iniciou_openfinance_fatura_sem_converter",
                "pattern": [
                    ["[open_finance:home:home_open_finance].click_button(trazer_dinheiro)",
                     "[open_finance:trazer_dinheiro:comprovante_pix_open_finance].page_view()"],
                    ["[open_finance:trazer_dinheiro:comprovante_pix_open_finance].page_view()",
                     "[cartoes:fatura:fatura_atual].click_button(opcoes_pagamento)"]
                ],
                "not_pattern": [
                    ["[open_finance:trazer_dinheiro:comprovante_pix_open_finance].page_view()",
                     "[cartoes:pagamento:comprovante_pagamento].page_view()"]
                ],
            },
            {
                # 012 · Trazer dinheiro (Open Finance) avulso: gatilho e, depois dele, a conversão
                "label_col": "is_openfinance_pago",
                "pattern": [
                    ["[open_finance:home:home_open_finance].click_button(trazer_dinheiro)",
                     "[open_finance:trazer_dinheiro:comprovante_pix_open_finance].page_view()"]
                ],
            },
            {
                # 012 · abandono: gatilho sem a conversão depois dele
                "label_col": "is_iniciou_openfinance_sem_converter",
                "contain": ["[open_finance:home:home_open_finance].click_button(trazer_dinheiro)"],
                "not_pattern": [
                    ["[open_finance:home:home_open_finance].click_button(trazer_dinheiro)",
                     "[open_finance:trazer_dinheiro:comprovante_pix_open_finance].page_view()"]
                ],
            },
        ],
    },
    {
        "id": "006",
        "folder": "[006][PIX] Transferencia por meio de chave pix",
        "csv": "006.csv",
        "num_traces": 100,
        "min_events": 10,
        "labels": [
            {
                # 006 · PIX por chave: gatilho e, depois dele, a conversão
                "label_col": "is_pix_transferido",
                "pattern": [
                    ["[pix:transferencia:home_pix].click_button(transferir_chave)",
                     "[pix:transferencia:comprovante_pix].page_view()"]
                ],
            },
            {
                # 006 · abandono: gatilho sem a conversão depois dele
                "label_col": "is_iniciou_pix_sem_converter",
                "contain": ["[pix:transferencia:home_pix].click_button(transferir_chave)"],
                "not_pattern": [
                    ["[pix:transferencia:home_pix].click_button(transferir_chave)",
                     "[pix:transferencia:comprovante_pix].page_view()"]
                ],
            },
        ],
    },
    {
        "id": "007",
        "folder": "[007][investimento] aplicação no CDB",
        "csv": "007.csv",
        "num_traces": 100,
        "min_events": 10,
        "labels": [
            {
                # 007 · Aplicar em CDB: gatilho e, depois dele, a conversão
                "label_col": "is_cdb_aplicado",
                "pattern": [
                    ["[investimentos:renda_fixa:home_investimentos].click_button(renda_fixa)",
                     "[investimentos:renda_fixa:comprovante_investimento].page_view()"]
                ],
            },
            {
                # 007 · abandono: gatilho sem a conversão depois dele
                "label_col": "is_iniciou_cdb_sem_converter",
                "contain": ["[investimentos:renda_fixa:home_investimentos].click_button(renda_fixa)"],
                "not_pattern": [
                    ["[investimentos:renda_fixa:home_investimentos].click_button(renda_fixa)",
                     "[investimentos:renda_fixa:comprovante_investimento].page_view()"]
                ],
            },
        ],
    },
    {
        "id": "008",
        "folder": "[008][investimento] resgate aplicação CDB",
        "csv": "008.csv",
        "num_traces": 100,
        "min_events": 10,
        "labels": [
            {
                # 008 · Resgatar CDB: gatilho e, depois dele, a conversão
                "label_col": "is_cdb_resgatado",
                "pattern": [
                    ["[investimentos:meus_investimentos:carteira_renda_fixa].click_button(detalhar_cdb)",
                     "[investimentos:resgate:comprovante_resgate].page_view()"]
                ],
            },
            {
                # 008 · abandono: gatilho sem a conversão depois dele
                "label_col": "is_iniciou_resgate_sem_converter",
                "contain": ["[investimentos:meus_investimentos:carteira_renda_fixa].click_button(detalhar_cdb)"],
                "not_pattern": [
                    ["[investimentos:meus_investimentos:carteira_renda_fixa].click_button(detalhar_cdb)",
                     "[investimentos:resgate:comprovante_resgate].page_view()"]
                ],
            },
        ],
    },
    {
        "id": "009",
        "folder": "[009][cartao-investimento] resgate CDB pagar fatura cartao",
        "csv": "009.csv",
        "num_traces": 100,
        "min_events": 18,
        "labels": [
            {
                # 009 · Resgatar CDB e pagar a fatura (composta): etapa 1 e, depois dela, a etapa 2
                "label_col": "is_resgate_pago_fatura",
                "pattern": [
                    ["[investimentos:meus_investimentos:carteira_renda_fixa].click_button(detalhar_cdb)",
                     "[investimentos:resgate:comprovante_resgate].page_view()"],
                    ["[investimentos:resgate:comprovante_resgate].page_view()",
                     "[cartoes:pagamento:comprovante_pagamento].page_view()"]
                ],
            },
            {
                # 009 · abandono: resgatou, abriu as opções de pagamento depois e não pagou
                "label_col": "is_iniciou_resgate_pagar_fatura_sem_converter",
                "pattern": [
                    ["[investimentos:meus_investimentos:carteira_renda_fixa].click_button(detalhar_cdb)",
                     "[investimentos:resgate:comprovante_resgate].page_view()"],
                    ["[investimentos:resgate:comprovante_resgate].page_view()",
                     "[cartoes:fatura:fatura_atual].click_button(opcoes_pagamento)"]
                ],
                "not_pattern": [
                    ["[investimentos:resgate:comprovante_resgate].page_view()",
                     "[cartoes:pagamento:comprovante_pagamento].page_view()"]
                ],
            },
        ],
    },
    {
        "id": "010",
        "folder": "[010][cartao-pix] copia pix fatura depois paga",
        "csv": "010.csv",
        "num_traces": 100,
        "min_events": 10,
        "labels": [
            {
                # 010 · Ver a fatura e pagar pelo PIX copia e cola (composta): etapa 1 e, depois dela, a etapa 2
                "label_col": "is_fatura_paga_pix_copia_cola",
                "pattern": [
                    ["[cartoes:fatura:fatura_atual].click_button(opcoes_pagamento)",
                     "[pix:home:home_pix].click_button(pix_copia_e_cola)"],
                    ["[pix:home:home_pix].click_button(pix_copia_e_cola)",
                     "[pix:pix_copia_e_cola:comprovante_pix].page_view()"]
                ],
            },
            {
                # 010 · abandono: viu as opções da fatura, entrou no PIX copia e cola e não pagou
                "label_col": "is_iniciou_fatura_pix_copia_cola_sem_converter",
                "pattern": [
                    ["[cartoes:fatura:fatura_atual].click_button(opcoes_pagamento)",
                     "[pix:home:home_pix].click_button(pix_copia_e_cola)"]
                ],
                "not_pattern": [
                    ["[pix:home:home_pix].click_button(pix_copia_e_cola)",
                     "[pix:pix_copia_e_cola:comprovante_pix].page_view()"]
                ],
            },
            {
                # 011 · PIX copia e cola avulso: gatilho e, depois dele, a conversão
                "label_col": "is_copia_pix_paga",
                "pattern": [
                    ["[pix:home:home_pix].click_button(pix_copia_e_cola)",
                     "[pix:pix_copia_e_cola:comprovante_pix].page_view()"]
                ],
            },
            {
                # 011 · abandono: gatilho sem a conversão depois dele
                "label_col": "is_iniciou_copia_pix_sem_converter",
                "contain": ["[pix:home:home_pix].click_button(pix_copia_e_cola)"],
                "not_pattern": [
                    ["[pix:home:home_pix].click_button(pix_copia_e_cola)",
                     "[pix:pix_copia_e_cola:comprovante_pix].page_view()"]
                ],
            },
        ],
    },
]

def resolve_pm4py_cmd():
    """Localiza o executável pm4py-cli mais adequado para o ambiente."""
    cli_exe = shutil.which("pm4py-cli") or shutil.which("pm4py-cli.exe")
    if cli_exe:
        return [cli_exe]
    
    cli_bat = shutil.which("pm4py-cli.bat")
    if cli_bat:
        return [cli_bat]
    
    # Fallback chamando o módulo Python diretamente se existir em tools/
    tools_cli = BASE_DIR.parent / "tools" / "pm4py_cli" / "src" / "jornada_ux" / "main.py"
    if tools_cli.exists():
        return [sys.executable, str(tools_cli)]
    
    return [sys.executable, "-m", "jornada_ux.main"]


def run_command(cmd_args, dry_run=False, cwd=None):
    """Executa um comando pm4py com exibição clara no terminal."""
    cmd_str = " ".join(f'"{a}"' if " " in str(a) or "[" in str(a) else str(a) for a in cmd_args)
    print(f"\n[EXEC] {cmd_str}")
    if dry_run:
        print("  -> [DRY-RUN] Comando simulado com sucesso.")
        return 0

    res = subprocess.run(cmd_args, cwd=cwd)
    if res.returncode != 0:
        print(f"[ERRO] O comando falhou com código de saída: {res.returncode}")
        sys.exit(res.returncode)  # um rótulo que falha não pode passar em silêncio
    return res.returncode


def find_scenario_csv(scenario_dir, configured_csv=None):
    """Localiza o CSV de entrada do cenário."""
    if configured_csv:
        csv_path = scenario_dir / configured_csv
        if csv_path.exists():
            return csv_path
    
    csv_candidates = [f for f in scenario_dir.glob("*.csv") if not f.name.endswith("-simulado.csv") and not f.name.endswith("-rotulado.csv")]
    if csv_candidates:
        return csv_candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Gera massas de dados para o TCC via pm4py-cli com discovery, playout dinâmico e rotulagem."
    )
    parser.add_argument(
        "--num_traces", "-n", type=int, default=100,
        help="Quantidade de traces/jornadas a serem simuladas por cenário no playout (padrão: 100)."
    )
    parser.add_argument(
        "--start_session_at", "-s", type=int, default=1,
        help="Número inicial dinâmico da sessão (padrão: 1). Incrementado a cada cenário conforme num_traces."
    )
    parser.add_argument(
        "--algo", type=str, default="heuristics", choices=["heuristics", "inductive", "alpha"],
        help="Algoritmo de descoberta para a Rede de Petri no discovery (padrão: 'heuristics')."
    )
    parser.add_argument(
        "--mode", type=str, default="both", choices=["both", "petri", "dfg"],
        help="Modo de geração do discovery ('both', 'petri', 'dfg'; padrão: 'both')."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semente aleatória para reprodutibilidade do playout (padrão: 42)."
    )
    parser.add_argument(
        "--label_col", type=str, default=None,
        help="Se fornecido, sobrescreve o nome da coluna de rotulagem para todos os cenários."
    )
    parser.add_argument(
        "--contain", nargs="+", default=None,
        help="Se fornecido, sobrescreve os termos do filtro --contain do labeling para todos os cenários."
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="Se fornecido, sobrescreve os termos do filtro --exclude do labeling para todos os cenários."
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Filtra apenas os cenários especificados (ex: --scenarios 001 002 ou 1 2)."
    )
    parser.add_argument(
        "--skip_discovery_original", action="store_true", default=False,
        help="Pula a etapa de discovery dos CSVs originais."
    )
    parser.add_argument(
        "--skip_playout", action="store_true", default=False,
        help="Pula a etapa de playout (simulação)."
    )
    parser.add_argument(
        "--skip_discovery_simulado", action="store_true", default=False,
        help="Pula a etapa de discovery da massa simulada pelo playout."
    )
    parser.add_argument(
        "--skip_labeling", action="store_true", default=False,
        help="Pula a etapa de rotulagem (labeling)."
    )
    parser.add_argument(
        "--skip_total", action="store_true", default=False,
        help="Pula a consolidação de todos os cenários em total.csv e respectivo discovery."
    )
    parser.add_argument(
        "--usar_total_existente", action="store_true", default=False,
        help="Não reconstrói o total a partir dos cenários; rotula o dataset/processed/total.csv existente "
             "(as mesmas sessões do TCC)."
    )
    parser.add_argument(
        "--saida", default="total-rotulado.csv",
        help="Nome do CSV rotulado gravado em dataset/processed/ (ex.: total-rotulado-v1.1.csv)."
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Apenas exibe os comandos que seriam executados, sem executá-los de fato."
    )

    args = parser.parse_args()

    pm4py_base_cmd = resolve_pm4py_cmd()
    print("=" * 80)
    print("GERADOR E PROCESSADOR DE MASSAS - TCC")
    print(f"Executável pm4py: {' '.join(pm4py_base_cmd)}")
    print(f"Traces por cenário (--num_traces): {args.num_traces}")
    print(f"Sessão inicial (--start_session_at): {args.start_session_at}")
    print(f"Algoritmo Discovery: {args.algo} | Modo: {args.mode}")
    print("=" * 80)

    # Filtra cenários se solicitado
    cenarios = CENARIOS_CONFIG
    if args.scenarios:
        allowed = {s.zfill(3) if s.isdigit() else s for s in args.scenarios}
        cenarios = [c for c in CENARIOS_CONFIG if c["id"] in allowed or str(int(c["id"])) in allowed]

    current_session = args.start_session_at
    simulated_csv_paths = []
    labeled_csv_paths = []

    for cenario in cenarios:
        cid = cenario["id"]
        cfolder_name = cenario["folder"]
        
        # Suporta localização em dataset/raw/ ou direto em dataset/
        s_dir = BASE_DIR / "raw" / cfolder_name if (BASE_DIR / "raw" / cfolder_name).exists() else BASE_DIR / cfolder_name

        if not s_dir.exists():
            print(f"\n[AVISO] Diretório do cenário não encontrado: {s_dir}. Pulando...")
            continue

        input_csv = find_scenario_csv(s_dir, cenario.get("csv"))
        if not input_csv or not input_csv.exists():
            print(f"\n[AVISO] CSV de entrada não encontrado em {s_dir}. Pulando...")
            continue

        base_name = input_csv.stem
        out_dir = s_dir / "output"
        discovery_dir = out_dir / "discovery"
        playout_dir = out_dir / "playout"
        labeling_path = playout_dir / "labeling"
        pnml_path = discovery_dir / f"{base_name}-petri.pnml"
        simulated_csv = playout_dir / f"{base_name}-simulado.csv"
        labeled_csv = labeling_path / f"{base_name}-simulado-rotulado.csv"

        print("\n" + "#" * 80)
        print(f"PROCESSANDO CENÁRIO [{cid}]: {cfolder_name}")
        print(f"CSV de Origem: {input_csv.name}")
        print(f"Sessões deste cenário: SESS_{current_session:03d} a SESS_{current_session + args.num_traces - 1:03d}")
        print("#" * 80)

        # 1. DISCOVERY ORIGINAL
        if not args.skip_discovery_original:
            print(f"\n>>> [1/4] Discovery do CSV original: {input_csv.name}")
            cmd_disc = pm4py_base_cmd + [
                "discovery",
                "--mode", args.mode,
                "--algo", args.algo,
                "--csv", str(input_csv),
                "--output_dir", str(out_dir),
                "--yes",
            ]
            run_command(cmd_disc, dry_run=args.dry_run)

        # 2. PLAYOUT
        if not args.skip_playout:
            print(f"\n>>> [2/4] Playout: Gerando {args.num_traces} traces a partir da sessão {current_session}")
            cmd_playout = pm4py_base_cmd + [
                "playout",
                "--pnml", str(pnml_path),
                "--csv", str(input_csv),
                "--start_session_at", str(current_session),
                "--num_traces", str(args.num_traces),
                "--seed", str(args.seed),
                "--output_dir", str(out_dir),
                "--yes",
            ]
            run_command(cmd_playout, dry_run=args.dry_run)

        current_session += args.num_traces
        simulated_csv_paths.append((simulated_csv, cenario.get("min_events")))

        # 3. DISCOVERY DA MASSA SIMULADA
        if not args.skip_discovery_simulado:
            print(f"\n>>> [3/4] Discovery da massa simulada: {simulated_csv.name}")
            playout_output_dir = playout_dir / "output"
            cmd_disc_sim = pm4py_base_cmd + [
                "discovery",
                "--mode", args.mode,
                "--algo", args.algo,
                "--csv", str(simulated_csv),
                "--output_dir", str(playout_output_dir),
                "--yes",
            ]
            run_command(cmd_disc_sim, dry_run=args.dry_run)

        # 4. LABELING
        if not args.skip_labeling and not args.usar_total_existente:  # com o total existente, só o total é rotulado
            for lbl in cenario.get("labels", []):
                label_col = args.label_col or lbl.get("label_col") or "label"
                contain_list = args.contain if args.contain is not None else lbl.get("contain", [])
                exclude_list = args.exclude if args.exclude is not None else lbl.get("exclude", [])

                if labeled_csv.exists():
                    print(f"   [!] Arquivo existente encontrado: {labeled_csv.name}. Usando como origem.")
                    csv_origem = labeled_csv
                else:
                    print(f"   [!] Arquivo existente não encontrado {labeled_csv.name}")
                    csv_origem = simulated_csv

                print(f"\n>>> [4/4] Labeling '{label_col}' na massa simulada")
                cmd_label = pm4py_base_cmd + [
                    "labeling",
                    "--csv", str(csv_origem),
                    "--label_col", str(label_col),
                    "--output_dir", str(playout_dir),
                    "--output_base", base_name + "-simulado",
                    "--yes",
                ]

                if contain_list:
                    cmd_label.append("--contain")
                    cmd_label.extend([str(c) for c in contain_list])

                if exclude_list:
                    cmd_label.append("--exclude")
                    cmd_label.extend([str(e) for e in exclude_list])

                for a, b in lbl.get("pattern", []) or []:
                    cmd_label.extend(["--pattern", str(a), str(b)])
                for a, b in lbl.get("not_pattern", []) or []:
                    cmd_label.extend(["--not_pattern", str(a), str(b)])

                run_command(cmd_label, dry_run=args.dry_run)

            labeled_csv_paths.append(labeled_csv)

    # 5. CONSOLIDAÇÃO GERAL (total.csv e total-rotulado.csv)
    if not args.skip_total and not args.dry_run:
        print("\n" + "=" * 80)
        print("CONSOLIDAÇÃO GERAL: UNINDO MASSAS SIMULADAS EM total.csv")
        print("=" * 80)

        total_dir = BASE_DIR / "processed" if (BASE_DIR / "processed").exists() else BASE_DIR / "total"
        total_dir.mkdir(parents=True, exist_ok=True)
        total_csv_path = total_dir / "total-nova.csv"

        dfs = []
        total_filtradas = 0
        for sim_path, min_ev in simulated_csv_paths:
            if sim_path.exists():
                try:
                    df = pd.read_csv(sim_path)
                    sessoes_antes = df["cd_sessao"].nunique()

                    if min_ev is not None:
                        counts = df.groupby("cd_sessao")["cd_sessao"].transform("count")
                        df = df[counts >= min_ev].copy()
                        sessoes_depois = df["cd_sessao"].nunique()
                        filtradas = sessoes_antes - sessoes_depois
                        total_filtradas += filtradas
                        if filtradas > 0:
                            print(f"  [{sim_path.parent.parent.parent.name}] Filtro min_events={min_ev}: "
                                  f"{sessoes_antes} → {sessoes_depois} sessões ({filtradas} removidas)")

                    dfs.append(df)
                except Exception as e:
                    print(f"[AVISO] Erro ao ler {sim_path}: {e}")

        if total_filtradas > 0:
            print(f"\nTotal de sessões removidas pelo filtro min_events: {total_filtradas}")

        if args.usar_total_existente:
            dfs = []
            print("Usando o total.csv existente (mesmas sessões do TCC); consolidação e discovery do total puladas.")
        if dfs:
            df_total = pd.concat(dfs, ignore_index=True)
            df_total.to_csv(total_csv_path, index=False)
            print(f"\nTotal consolidado salvo com sucesso em: {total_csv_path}")
            print(f"Total de registros: {len(df_total)} | Total de sessões: {df_total['cd_sessao'].nunique()}")

            # Discovery geral
            total_playout_output = total_dir / "output" / "playout" / "output"
            print(f"\n>>> Executando Discovery no arquivo consolidado: {total_csv_path.name}")
            cmd_total_disc = pm4py_base_cmd + [
                "discovery",
                "--mode", args.mode,
                "--algo", args.algo,
                "--csv", str(total_csv_path),
                "--output_dir", str(total_playout_output),
                "--yes",
            ]
            run_command(cmd_total_disc, dry_run=args.dry_run)
        elif not args.usar_total_existente:
            print("[AVISO] Nenhum CSV simulado foi encontrado para consolidar total.csv.")

        # Rotulagem no consolidado
        base_name = "total"
        out_dir = total_dir / "output"
        labeling_path = out_dir / "labeling"
        simulated_csv = total_dir / ("total.csv" if args.usar_total_existente else "total-nova.csv")
        labeled_csv = labeling_path / f"{base_name}-rotulado.csv"

        if not args.skip_labeling and simulated_csv.exists():
            primeiro = True
            for cenario in cenarios:
                for lbl in cenario.get("labels", []):
                    label_col = args.label_col or lbl.get("label_col") or "label"
                    contain_list = args.contain if args.contain is not None else lbl.get("contain", [])
                    exclude_list = args.exclude if args.exclude is not None else lbl.get("exclude", [])

                    csv_origem = simulated_csv if primeiro else labeled_csv
                    primeiro = False

                    print(f"\n>>> [Total] Labeling '{label_col}' no arquivo consolidado")
                    cmd_label = pm4py_base_cmd + [
                        "labeling",
                        "--csv", str(csv_origem),
                        "--label_col", str(label_col),
                        "--output_dir", str(out_dir),
                        "--output_base", base_name,
                        "--yes",
                    ]

                    if contain_list:
                        cmd_label.append("--contain")
                        cmd_label.extend([str(c) for c in contain_list])

                    if exclude_list:
                        cmd_label.append("--exclude")
                        cmd_label.extend([str(e) for e in exclude_list])

                    for a, b in lbl.get("pattern", []) or []:
                        cmd_label.extend(["--pattern", str(a), str(b)])
                    for a, b in lbl.get("not_pattern", []) or []:
                        cmd_label.extend(["--not_pattern", str(a), str(b)])

                    run_command(cmd_label, dry_run=args.dry_run)

            if not args.dry_run and not labeled_csv.exists():
                sys.exit(f"[ERRO] Rotulagem não gerou {labeled_csv}")
            if not args.dry_run and labeled_csv.exists():
                destino = total_dir / args.saida
                shutil.copyfile(labeled_csv, destino)
                print(f"\nMassa rotulada copiada para {destino} (entrada do run_all.py)")

    print("\n" + "=" * 80)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Próxima sessão livre para novas execuções: SESS_{current_session:03d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
