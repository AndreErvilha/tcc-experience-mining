import os
import pm4py
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite, validate_csv_schema

def run_discovery(args):
    # Validar schema do CSV antes de processar
    if not validate_csv_schema(args.csv, stop_on_error=True):
        return

    df_raw, df_filtered, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    # Definir caminhos de saída
    png_petri = resolve_output_file_path(args, args.csv, "discovery", "petri", "png")
    pnml_petri = resolve_output_file_path(args, args.csv, "discovery", "petri", "pnml")
    png_dfg = resolve_output_file_path(args, args.csv, "discovery", "dfg", "png")
    dfg_file = resolve_output_file_path(args, args.csv, "discovery", "dfg", "dfg")

    # Verificar arquivos a serem checados
    files_to_check = []
    if args.mode in ["petri", "both"]:
        files_to_check.extend([png_petri, pnml_petri])
    if args.mode in ["dfg", "both"]:
        files_to_check.extend([png_dfg, dfg_file])

    if not check_overwrite(files_to_check, args):
        return

    print(f"\nTotal de registros lidos: {len(df_raw)}")
    print(f"Registros após filtragem de eventos ({args.events}): {len(df_filtered)}")

    print("\nAmostra dos eventos formatados:")
    for name in df_filtered['event_name'].head(5):
        print(f" - {name}")

    # Geração de Rede de Petri se solicitado
    if args.mode in ["petri", "both"]:
        print(f"\n=== Descobrindo e Exportando Rede de Petri ({args.algo}) ===")
        try:
            if args.algo == "inductive":
                net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(df_event_log)
            elif args.algo == "alpha":
                net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(df_event_log)
            elif args.algo == "heuristics":
                net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(df_event_log)
            else:
                print(f"Erro: Algoritmo '{args.algo}' desconhecido.")
                return

            print("Rede de Petri descoberta com sucesso!")
            
            # Exportar PNML
            print(f"Salvando PNML em: {os.path.abspath(pnml_petri)}")
            pm4py.write_pnml(net, initial_marking, final_marking, pnml_petri)
            print("PNML exportado com sucesso!")
            
            # Exportar PNG vertical
            print(f"Salvando PNG da Rede de Petri (vertical) em: {os.path.abspath(png_petri)}")
            pm4py.save_vis_petri_net(net, initial_marking, final_marking, png_petri, rankdir='TB')
            print("PNG da Rede de Petri exportado com sucesso!")
        except Exception as e:
            print(f"Erro na geração da Rede de Petri: {e}")

    # Geração de DFG se solicitado
    if args.mode in ["dfg", "both"]:
        print("\n=== Descobrindo e Exportando DFG (Directly-Follows Graph) ===")
        try:
            dfg, start_activities, end_activities = pm4py.discover_dfg(df_event_log)
            print("DFG descoberto com sucesso!")
            
            # Exportar PNG vertical
            print(f"Salvando PNG do DFG (vertical) em: {os.path.abspath(png_dfg)}")
            pm4py.save_vis_dfg(dfg, start_activities, end_activities, png_dfg, rankdir='TB')
            print("PNG do DFG exportado com sucesso!")

            # Exportar arquivo em formato .dfg
            print(f"Salvando arquivo de serialização .dfg em: {os.path.abspath(dfg_file)}")
            pm4py.write_dfg(dfg, start_activities, end_activities, dfg_file)
            print("Arquivo .dfg exportado com sucesso!")
        except Exception as e:
            print(f"Erro na geração do DFG: {e}")
