"""
Módulo list-activities: Lista todos os nomes exatos de atividade de um CSV de jornada,
para que o usuário saiba quais valores usar nos comandos de filtro.
"""

import sys
from jornada_ux.modules.utils import prepare_event_log, validate_csv_schema


def _shell_quote(value):
    """Envolve o valor em aspas duplas para uso seguro em linha de comando
    (PowerShell/cmd), preservando espaços, colchetes, parênteses e dois-pontos."""
    return '"' + str(value) + '"'


def run_list_activities(args):
    """Lista as atividades distintas de um CSV de logs."""

    # Validar schema antes de processar
    if not validate_csv_schema(args.csv, stop_on_error=False):
        print("AVISO: O CSV não atende ao schema esperado. Tentando prosseguir mesmo assim...\n")

    # Carregar e formatar
    df_raw, df_filtered, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    # Extrair atividades distintas da coluna concept:name
    activities = sorted(df_event_log["concept:name"].dropna().unique())

    if not activities:
        print("Nenhuma atividade encontrada com os filtros atuais.")
        return

    print(f"\nAtividades disponíveis ({len(activities)}):")
    for act in activities:
        print(f"  {act}")

    # Dicas de uso
    print(f"\nDicas de uso — copie e cole os nomes exatos:")
    csv = _shell_quote(args.csv)
    wild = _shell_quote("...")
    if len(activities) >= 1:
        act = _shell_quote(activities[0])
        print(f"\n  # Filtrar casos que contêm uma atividade:")
        print(f"  python run.py analysis filter --csv {csv} --contain {act}")
    if len(activities) >= 2:
        act_a = _shell_quote(activities[0])
        act_b = _shell_quote(activities[-1])
        print(f"\n  # Filtrar casos onde A aparece antes de B:")
        print(f"  python run.py analysis filter --csv {csv} --trace_include {wild} {act_a} {wild} {act_b} {wild}")

    print(f"\n  Windows/PowerShell: prefira o executável real (pm4py-cli.exe) ou run.ps1 ao")
    print(f"  shim .bat do pyenv, que corrompe argumentos com [ ] ( ) :. Veja a seção Windows no README.")
