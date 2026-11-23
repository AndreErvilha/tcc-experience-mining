import os
import pm4py
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite


def _apply_contain_filter(values, df_event_log, mode):
    """Filtra casos que contêm (mode='contain') ou exclui casos que contêm (mode='exclude')
       os nomes de atividade fornecidos. Aceita match exato, substring e limpa aspas residuais."""
    all_activities = list(df_event_log['concept:name'].unique())
    matched = set()
    missing = []

    for raw_v in values:
        v = raw_v.strip("'\"")
        # 1. Match exato
        if v in all_activities:
            matched.add(v)
            continue

        # 2. Match por substring (case-insensitive)
        sub_matches = [act for act in all_activities if v.lower() in act.lower()]
        if sub_matches:
            matched.update(sub_matches)
        else:
            missing.append(raw_v)

    matched_list = sorted(list(matched))
    if missing:
        print(f"Aviso: Atividades/termos não encontrados no log (ignorados): {missing}")
    if not matched_list:
        if mode == "contain":
            print(f"Aviso: Nenhuma atividade fornecida está presente no log para o filtro '{mode}'. Retornando vazio.")
            return df_event_log.iloc[0:0].copy()
        print(f"Aviso: Nenhuma atividade fornecida está presente no log para o filtro '{mode}'. Ignorando.")
        return df_event_log

    label = "contêm" if mode == "contain" else "excluem"
    retain = mode == "contain"
    print(f"Aplicando filtro de '{mode}': mantendo casos que {label} eventos: {matched_list}")
    return pm4py.filter_event_attribute_values(
        df_event_log,
        attribute_key='concept:name',
        values=matched_list,
        level='case',
        retain=retain,
        case_id_key='case:concept:name'
    )


def _apply_eventually_follows_filter(relation_pairs, df_event_log, retain):
    """Aplica filtro eventually-follows: para cada par (act_A, act_B),
       resolve nomes exatos (ou por substring/aspas) e aplica pm4py.filter_eventually_follows_relation."""
    all_activities = list(df_event_log['concept:name'].unique())
    for act_a, act_b in relation_pairs:
        clean_a = act_a.strip("'\"")
        clean_b = act_b.strip("'\"")

        matched_a = [act for act in all_activities if clean_a.lower() in act.lower()] if clean_a not in all_activities else [clean_a]
        matched_b = [act for act in all_activities if clean_b.lower() in act.lower()] if clean_b not in all_activities else [clean_b]

        if not matched_a or not matched_b:
            ausente = act_a if not matched_a else act_b
            if retain:
                print(
                    f"Aviso: Atividade '{ausente}' não encontrada no log para filtro positivo "
                    f"({act_a} > {act_b}). Retornando vazio."
                )
                return df_event_log.iloc[0:0].copy()
            print(f"Aviso: Atividade '{ausente}' não encontrada no log. Pulando par negativo ({act_a} > {act_b}).")
            continue

        relations = [(a, b) for a in matched_a for b in matched_b]
        action = "mantendo" if retain else "removendo"
        print(f"Aplicando filtro eventually-follows ({action}): {relations}")
        df_event_log = pm4py.filter_eventually_follows_relation(
            df_event_log,
            relations=relations,
            retain=retain,
            activity_key='concept:name',
            timestamp_key='time:timestamp',
            case_id_key='case:concept:name'
        )
    return df_event_log


def _apply_trace_segment_filter(segments, df_event_log, retain):
    """Aplica filtro por segmentos de trace usando pm4py.filter_trace_segments.
    '...' no segmento funciona como wildcard (zero ou mais atividades)."""
    df_event_log = df_event_log.copy()
    df_event_log['concept:name'] = df_event_log['concept:name'].astype(str)

    action = "mantendo" if retain else "removendo"
    for seg in segments:
        clean_seg = [step.strip("'\"") for step in seg]
        print(f"Aplicando filtro de segmentos ({action}): {clean_seg}")
        df_event_log = pm4py.filter_trace_segments(
            df_event_log,
            [clean_seg],
            positive=retain,
            activity_key='concept:name',
            timestamp_key='time:timestamp',
            case_id_key='case:concept:name'
        )
    return df_event_log


def run_filter(args):
    output_csv = resolve_output_file_path(args, args.csv, "analysis/filter", "filtrado", "csv", explicit_override=args.output_csv)
    if not check_overwrite(output_csv, args):
        return

    # Carregar
    print(f"Lendo CSV de logs para filtragem: {args.csv}")
    df, _, df_event_log = prepare_event_log(args.csv, args.events, temp_col=True)
    if df_event_log is None:
        return

    log_len_before = df_event_log['case:concept:name'].nunique()
    print(f"Total de casos (sessões) inicial: {log_len_before}")

def apply_all_filters(df_event_log, args):
    """Aplica a suíte completa de filtros a um DataFrame PM4Py e retorna o df_event_log filtrado."""
    # Aplicar Filtro de Início
    if getattr(args, 'start_activities', None):
        print(f"Aplicando filtro de início para manter apenas casos contendo no início: {args.start_activities}")
        all_acts = list(df_event_log['concept:name'].unique())
        valid = set()
        missing = []
        for raw_a in args.start_activities:
            a = raw_a.strip("'\"")
            if a in all_acts:
                valid.add(a)
            else:
                sub_m = [act for act in all_acts if a.lower() in act.lower()]
                if sub_m:
                    valid.update(sub_m)
                else:
                    missing.append(raw_a)
        valid_list = sorted(list(valid))
        if missing:
            print(f"Aviso: Atividades de início não encontradas no log (ignoradas): {missing}")
        if valid_list:
            df_event_log = pm4py.filter_start_activities(df_event_log, valid_list)
        else:
            print("Aviso: Nenhuma atividade de início fornecida está presente no log. Ignorando filtro de início.")

    # Aplicar Filtro de Fim
    if getattr(args, 'end_activities', None):
        print(f"Aplicando filtro de fim para manter apenas casos contendo no fim: {args.end_activities}")
        all_acts = list(df_event_log['concept:name'].unique())
        valid = set()
        missing = []
        for raw_a in args.end_activities:
            a = raw_a.strip("'\"")
            if a in all_acts:
                valid.add(a)
            else:
                sub_m = [act for act in all_acts if a.lower() in act.lower()]
                if sub_m:
                    valid.update(sub_m)
                else:
                    missing.append(raw_a)
        valid_list = sorted(list(valid))
        if missing:
            print(f"Aviso: Atividades de fim não encontradas no log (ignoradas): {missing}")
        if valid_list:
            df_event_log = pm4py.filter_end_activities(df_event_log, valid_list)
        else:
            print("Aviso: Nenhuma atividade de fim fornecida está presente no log. Ignorando filtro de fim.")

    # Aplicar Filtro de Contém — --contain
    if getattr(args, 'contain', None):
        df_event_log = _apply_contain_filter(args.contain, df_event_log, "contain")

    # Aplicar Filtro de Exclusão — --exclude
    if getattr(args, 'exclude', None):
        df_event_log = _apply_contain_filter(args.exclude, df_event_log, "exclude")

    # Aplicar Filtro Eventually-Follows positivo — --pattern
    if getattr(args, 'pattern', None):
        df_event_log = _apply_eventually_follows_filter(args.pattern, df_event_log, retain=True)

    # Aplicar Filtro Eventually-Follows negativo — --not_pattern
    if getattr(args, 'not_pattern', None):
        df_event_log = _apply_eventually_follows_filter(args.not_pattern, df_event_log, retain=False)

    # Aplicar Filtro de Segmentos (include) — --trace_include
    if getattr(args, 'trace_include', None):
        df_event_log = _apply_trace_segment_filter(args.trace_include, df_event_log, retain=True)

    # Aplicar Filtro de Segmentos (exclude) — --trace_exclude
    if getattr(args, 'trace_exclude', None):
        df_event_log = _apply_trace_segment_filter(args.trace_exclude, df_event_log, retain=False)

    # Aplicar Filtro de Duração
    min_dur = getattr(args, 'min_duration', None)
    max_dur = getattr(args, 'max_duration', None)
    if min_dur is not None or max_dur is not None:
        min_v = min_dur if min_dur is not None else 0
        max_v = max_dur if max_dur is not None else 999999999
        print(f"Aplicando filtro de duração de caso: entre {min_v}s e {max_v}s")
        try:
            df_event_log = pm4py.filter_case_duration(df_event_log, min_v, max_v)
        except Exception as e:
            print(f"Erro ao aplicar filtro de duração: {e}")

    # Aplicar Filtro de Quantidade de Eventos (Trace Length / Tamanho da Sessão)
    min_ev = getattr(args, 'min_events', None)
    max_ev = getattr(args, 'max_events', None)
    if min_ev is not None or max_ev is not None:
        min_v = min_ev if min_ev is not None else 0
        max_v = max_ev if max_ev is not None else 999999999
        print(f"Aplicando filtro de quantidade de eventos por sessão: entre {min_v} e {max_v} eventos")
        try:
            counts = df_event_log['case:concept:name'].value_counts()
            valid_cases = counts[(counts >= min_v) & (counts <= max_v)].index
            df_event_log = df_event_log[df_event_log['case:concept:name'].isin(valid_cases)].copy()
        except Exception as e:
            print(f"Erro ao aplicar filtro de quantidade de eventos: {e}")

    return df_event_log


def run_filter(args):
    output_csv = resolve_output_file_path(args, args.csv, "analysis/filter", "filtrado", "csv", explicit_override=args.output_csv)
    if not check_overwrite(output_csv, args):
        return

    # Carregar
    print(f"Lendo CSV de logs para filtragem: {args.csv}")
    df, _, df_event_log = prepare_event_log(args.csv, args.events, temp_col=True)
    if df_event_log is None:
        return

    log_len_before = df_event_log['case:concept:name'].nunique()
    print(f"Total de casos (sessões) inicial: {log_len_before}")

    df_event_log = apply_all_filters(df_event_log, args)

    # Traduzir o log filtrado de volta para o CSV original
    remaining_session_ids = df_event_log['case:concept:name'].unique()
    df_output = df[df['cd_sessao'].isin(remaining_session_ids)].copy()

    log_len_after = len(remaining_session_ids)
    print(f"Total de casos (sessões) pós-filtragem: {log_len_after}")

    try:
        df_output.to_csv(output_csv, index=False)
        print(f"\nSucesso! Dataset filtrado e salvo em: {os.path.abspath(output_csv)}")
        reduction = ((log_len_before - log_len_after) / log_len_before) * 100 if log_len_before > 0 else 0
        print(f"Casos restantes: {log_len_after} (Redução de {reduction:.1f}%)")
    except Exception as e:
        print(f"Erro ao salvar arquivo filtrado: {e}")
