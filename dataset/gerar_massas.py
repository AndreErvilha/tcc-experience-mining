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
                "label_col": "is_fatura_paga_saldo",
                "contain": ["[cartoes:pagamento:comprovante_pagamento].page_view()"],
                "exclude": [],
            },
            {
                # Entrou nas opções de pagamento, mas não chegou ao comprovante
                "label_col": "is_iniciou_pagar_saldo_sem_converter",
                "contain": ["[cartoes:pagamento:opcoes_pagamento].page_view()"],
                "exclude": ["[cartoes:pagamento:comprovante_pagamento].page_view()"],
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
                "label_col": "is_pix_copiado",
                "contain": ["[cartoes:pagamento:sucesso_copia].page_view()"],
                "exclude": [],
            },
            {
                # Entrou nas opções de pagamento, mas não chegou à tela de sucesso da cópia
                "label_col": "is_iniciou_copiar_pix_sem_converter",
                "contain": ["[cartoes:pagamento:opcoes_pagamento].page_view()"],
                "exclude": ["[cartoes:pagamento:sucesso_copia].page_view()"],
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
                "label_col": "is_fatura_parcelada",
                "contain": ["[cartoes:parcelamento:comprovante_parcelamento].page_view()"],
                "exclude": [],
            },
            {
                # Abriu o slider de parcelas, mas não chegou ao comprovante de parcelamento
                "label_col": "is_iniciou_parcelamento_sem_converter",
                "contain": ["[cartoes:parcelamento:slider_parcelas].page_view()"],
                "exclude": ["[cartoes:parcelamento:comprovante_parcelamento].page_view()"],
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
                "label_col": "is_historico_visualizado",
                "contain": ["[cartoes:historico:visualizacao_pdf].page_view()"],
                "exclude": [],
            },
            {
                # Acessou o histórico de faturas, mas não chegou à visualização do PDF
                "label_col": "is_iniciou_historico_sem_converter",
                "contain": ["[cartoes:historico:historico_faturas].page_view()"],
                "exclude": ["[cartoes:historico:visualizacao_pdf].page_view()"],
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
                "label_col": "is_openfinance_pago",
                "contain": ["[open_finance:trazer_dinheiro:comprovante_pix_open_finance].page_view()"],
                "exclude": [],
            },
            {
                # Selecionou o banco de origem, mas não concluiu a transferência via Open Finance
                "label_col": "is_iniciou_openfinance_sem_converter",
                "contain": ["[open_finance:trazer_dinheiro:selecao_banco_consentido].page_view()"],
                "exclude": ["[open_finance:trazer_dinheiro:comprovante_pix_open_finance].page_view()"],
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
                "label_col": "is_pix_transferido",
                "contain": ["[pix:transferencia:comprovante_pix].page_view()"],
                "exclude": [],
            },
            {
                # Digitou a chave PIX, mas não chegou ao comprovante de transferência
                "label_col": "is_iniciou_pix_sem_converter",
                "contain": ["[pix:transferencia:input_chave_pix].page_view()"],
                "exclude": ["[pix:transferencia:comprovante_pix].page_view()"],
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
                "label_col": "is_cdb_aplicado",
                "contain": ["[investimentos:renda_fixa:comprovante_investimento].page_view()"],
                "exclude": [],
            },
            {
                # Acessou o detalhe do CDB, mas não chegou ao comprovante de aplicação
                "label_col": "is_iniciou_cdb_sem_converter",
                "contain": ["[investimentos:renda_fixa:detalhe_cdb_pos].page_view()"],
                "exclude": ["[investimentos:renda_fixa:comprovante_investimento].page_view()"],
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
                "label_col": "is_cdb_resgatado",
                "contain": ["[investimentos:resgate:comprovante_resgate].page_view()"],
                "exclude": [],
            },
            {
                # Acessou o detalhe da posição CDB, mas não chegou ao comprovante de resgate
                "label_col": "is_iniciou_resgate_sem_converter",
                "contain": ["[investimentos:resgate:detalhe_posicao_cdb].page_view()"],
                "exclude": ["[investimentos:resgate:comprovante_resgate].page_view()"],
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
                "label_col": "is_resgate_pago_fatura",
                "contain": ["[cartoes:pagamento:comprovante_pagamento].page_view()"],
                "exclude": [],
            },
            {
                # Iniciou o resgate do CDB, mas não chegou ao comprovante de pagamento da fatura
                "label_col": "is_iniciou_resgate_pagar_fatura_sem_converter",
                "contain": ["[investimentos:resgate:detalhe_posicao_cdb].page_view()"],
                "exclude": ["[cartoes:pagamento:comprovante_pagamento].page_view()"],
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
                "label_col": "is_copia_pix_paga",
                "contain": ["[pix:pix_copia_e_cola:comprovante_pix].page_view()"],
                "exclude": [],
            },
            {
                # Entrou na tela de PIX copia e cola, mas não chegou ao comprovante
                "label_col": "is_iniciou_copia_pix_sem_converter",
                "contain": ["[pix:pix_copia_e_cola:pix_copia_e_cola].page_view()"],
                "exclude": ["[pix:pix_copia_e_cola:comprovante_pix].page_view()"],
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
        if not args.skip_labeling:
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

                run_command(cmd_label, dry_run=args.dry_run)

            labeled_csv_paths.append(labeled_csv)

    # 5. CONSOLIDAÇÃO GERAL (total.csv e total-rotulado.csv)
    if not args.skip_total and not args.dry_run:
        print("\n" + "=" * 80)
        print("CONSOLIDAÇÃO GERAL: UNINDO MASSAS SIMULADAS EM total.csv")
        print("=" * 80)

        total_dir = BASE_DIR / "processed" if (BASE_DIR / "processed").exists() else BASE_DIR / "total"
        total_dir.mkdir(parents=True, exist_ok=True)
        total_csv_path = total_dir / "total.csv"

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
        else:
            print("[AVISO] Nenhum CSV simulado foi encontrado para consolidar total.csv.")

        # Rotulagem no consolidado
        base_name = "total"
        out_dir = total_dir / "output"
        labeling_path = out_dir / "labeling"
        simulated_csv = total_dir / "total.csv"
        labeled_csv = labeling_path / f"{base_name}-rotulado.csv"

        if not args.skip_labeling and simulated_csv.exists():
            for cenario in cenarios:
                for lbl in cenario.get("labels", []):
                    label_col = args.label_col or lbl.get("label_col") or "label"
                    contain_list = args.contain if args.contain is not None else lbl.get("contain", [])
                    exclude_list = args.exclude if args.exclude is not None else lbl.get("exclude", [])

                    csv_origem = labeled_csv if labeled_csv.exists() else simulated_csv

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

                    run_command(cmd_label, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Próxima sessão livre para novas execuções: SESS_{current_session:03d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
