import os
import re
import random
import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.simulation.playout.petri_net import algorithm as playout_handler
from pm4py.algo.simulation.playout.petri_net.variants import stochastic_playout
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite

def run_playout(args):
    # 1. Configurar sementes para reprodutibilidade
    print(f"Configurando semente de aleatoriedade: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.pnml):
        print(f"Erro: Arquivo PNML '{args.pnml}' não encontrado.")
        return

    # 2. Carregar a rede de Petri do PNML
    print(f"Carregando Rede de Petri de: {args.pnml}")
    try:
        net, initial_marking, final_marking = pm4py.read_pnml(args.pnml)
        print("Rede de Petri carregada com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar PNML: {e}")
        return

    # 3. Carregar e formatar o CSV semente para o PM4Py
    print(f"Lendo e formatando CSV semente para cálculo de frequências: {args.csv}")
    _, _, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    # 4. Determinar o caminho de saída e checar sobrescrita
    input_file = args.csv if args.csv else args.pnml
    output_csv = resolve_output_file_path(args, input_file, "playout", "simulado", "csv", explicit_override=args.output_csv)

    if not check_overwrite(output_csv, args):
        return

    # 5. Executar o Stochastic Playout nativo do PM4Py usando o log de frequências
    print(f"\nIniciando Simulação Estocástica para gerar {args.num_traces} jornadas...")
    try:
        parameters = {
            stochastic_playout.Parameters.LOG: df_event_log,
            stochastic_playout.Parameters.NO_TRACES: args.num_traces
        }
        
        simulated_log = playout_handler.apply(
            net,
            initial_marking,
            final_marking=final_marking,
            variant=playout_handler.Variants.STOCHASTIC_PLAYOUT,
            parameters=parameters
        )
        print("Simulação concluída com sucesso!")
    except Exception as e:
        print(f"Erro ao simular playout: {e}")
        return

    # 6. Converter o log simulado de volta para DataFrame do Pandas
    try:
        simulated_df = pm4py.convert_to_dataframe(simulated_log)
    except Exception as e:
        print(f"Erro ao converter log simulado para DataFrame: {e}")
        return

    # 7. Traduzir as strings programáticas de volta para colunas normais
    print("Realizando conversão reversa do log simulado para formato tabular...")
    activity_re_generic = re.compile(r"^\[([^:]+):([^:]+):([^\]]+)\]\.([a-zA-Z0-9_]+)\((.*)\)$")

    parsed_rows = []
    for _, row in simulated_df.iterrows():
        case_id = str(int(row['case:concept:name']) + args.start_session_at)
        try:
            cd_sessao = f"SESS_{int(case_id) + 0:03d}"
        except ValueError:
            cd_sessao = f"SESS_{case_id}"

        activity_str = str(row['concept:name'])
        timestamp = row['time:timestamp']

        match = activity_re_generic.match(activity_str)
        if match:
            cd_produto = match.group(1)
            cd_sub_produto = match.group(2)
            page_name = match.group(3)
            cd_evento_tipo = match.group(4)
            button_text = match.group(5)
            if button_text == "":
                button_text = None
        else:
            cd_produto = "Unknown"
            cd_sub_produto = "Unknown"
            page_name = "Unknown"
            cd_evento_tipo = activity_str
            button_text = None

        parsed_rows.append({
            "cd_sessao": cd_sessao,
            "cd_produto": cd_produto,
            "cd_sub_produto": cd_sub_produto,
            "cd_evento_tipo": cd_evento_tipo,
            "button_text": button_text,
            "page_name": page_name,
            "timestamp": timestamp
        })

    df_output = pd.DataFrame(parsed_rows)
    df_output = df_output.sort_values(by=["cd_sessao", "timestamp"]).reset_index(drop=True)

    try:
        df_output.to_csv(output_csv, index=False)
        print(f"\nSucesso! Arquivo simulado gerado e salvo em: {os.path.abspath(output_csv)}")
        print(f"Total de eventos gerados: {len(df_output)}")
        print(f"Total de sessões simuladas: {df_output['cd_sessao'].nunique()}")
    except Exception as e:
        print(f"Erro ao salvar arquivo CSV de saída: {e}")
