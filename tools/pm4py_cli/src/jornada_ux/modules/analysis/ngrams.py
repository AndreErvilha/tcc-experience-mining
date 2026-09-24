"""
Módulo de N-Gramas de Navegação.

Extrai e ranqueia as sequências contíguas de N passos que mais se repetem
entre as sessões, revelando micro-padrões de comportamento — do bigrama
mais frequente ("home -> hub_cartoes") a trigramas e além.

Gera também o DFG visual (PNG + .dfg) dos n-gramas para inspeção gráfica.
Com --highlight_ngrams, gera um PNG individual para cada n-grama do top-k
destacando suas arestas no DFG completo.
"""
import os
import pandas as pd
import pm4py
from collections import Counter, defaultdict
from jornada_ux.modules.utils import prepare_event_log, resolve_output_file_path, check_overwrite


def extract_ngrams(sequence, n):
    """Extrai n-gramas contíguos de uma sequência usando janela deslizante."""
    if len(sequence) < n:
        return []
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def format_ngram(ngram_tuple):
    """Formata uma tupla de n-grama como string legível."""
    return " -> ".join(str(s) for s in ngram_tuple)


def build_dfg_from_ngrams(ngram_counter, n):
    """
    Constrói um DFG a partir de um contador de n-gramas.

    Para bigramas (n=2): cada entrada (A, B) -> count vira aresta A->B.
    Para n>2: decompõe cada n-grama em pares consecutivos (A,B), (B,C), ...
    e acumula os pesos.

    Retorna (dfg_dict, start_activities, end_activities).
    """
    dfg = defaultdict(int)

    for ngram, count in ngram_counter.items():
        for i in range(len(ngram) - 1):
            edge = (ngram[i], ngram[i + 1])
            dfg[edge] += count

    sources = set()
    targets = set()
    for src, tgt in dfg:
        sources.add(src)
        targets.add(tgt)

    start_activities = {a: 1 for a in (sources - targets)}
    end_activities = {a: 1 for a in (targets - sources)}

    return dict(dfg), start_activities, end_activities


def _shorten_label(label, max_len=28):
    """Encurta um label de atividade para exibição no grafo."""
    s = label.split("].")[-1] if "]." in label else label
    s = s.replace(".page_view()", ".pv").replace(".click_button(", ".clk(")
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


def save_ngram_highlights(dfg_full, df_out, ngram_counter, n, total_sessions,
                          output_dir, base_name):
    """
    Gera PNGs individuais destacando cada n-grama do top-k no DFG completo.

    Cada PNG mostra o grafo completo com arestas em cinza claro e as arestas
    do n-grama específico destacadas em vermelho com os pesos.
    """
    import graphviz

    # Mapa de IDs curtos para nós (Graphviz não lida bem com labels gigantes)
    node_id_map = {}
    reverse_map = {}
    for i, (src, tgt) in enumerate(dfg_full):
        for node in (src, tgt):
            if node not in node_id_map:
                nid = f"n{len(node_id_map)}"
                node_id_map[node] = nid
                reverse_map[nid] = node

    for idx, (_, row) in enumerate(df_out.iterrows()):
        ngram_str = row['ngram']
        count = row['ocorrencias_total']
        pct = row['pct_sessoes']

        steps = ngram_str.split(" -> ")
        highlight_edges = set()
        for i in range(len(steps) - 1):
            highlight_edges.add((steps[i], steps[i + 1]))

        # Construir Digraph Graphviz
        dot = graphviz.Digraph(
            format='png',
            engine='dot',
        )
        dot.attr(rankdir='TB', splines='polyline', nodesep='0.6', ranksep='0.8')
        dot.attr('node', shape='box', style='filled,rounded', fillcolor='#fafafa',
                 fontname='Helvetica', fontsize='10', penwidth='1.2',
                 color='#888888', margin='0.15,0.08')
        dot.attr('edge', fontname='Helvetica', fontsize='9')

        # Nós
        for node, nid in node_id_map.items():
            label = _shorten_label(node, max_len=32)
            dot.node(nid, label=label)

        # Arestas
        for (src, tgt), weight in dfg_full.items():
            sid = node_id_map.get(src)
            tid = node_id_map.get(tgt)
            if not sid or not tid:
                continue
            if (src, tgt) in highlight_edges:
                dot.edge(sid, tid, label=str(weight),
                         color='#e74c3c', fontcolor='#c0392b',
                         penwidth='2.5', weight='10')
            else:
                dot.edge(sid, tid, color='#cccccc', penwidth='0.8',
                         weight='1', style='dashed')

        # Título do grafo
        short_title = ngram_str
        if len(short_title) > 120:
            short_title = short_title[:117] + '...'
        title_text = (f"#{idx + 1} (n={n})  [{count}x | {pct}% de {total_sessions} sessões]\\n"
                      f"{short_title}")
        dot.attr(label=title_text, labelloc='t', fontsize='11',
                 fontname='Helvetica', fontcolor='#2c3e50')

        safe_name = f"ngram-highlight-{idx + 1:02d}"
        out_path = os.path.join(output_dir, f"{base_name}-{safe_name}")
        dot.render(out_path, cleanup=True)
        print(f"  PNG #{idx + 1} salvo: {os.path.basename(out_path)}.png")


def run_ngrams(args):
    """Ponto de entrada do comando analysis ngrams."""
    output_csv = resolve_output_file_path(
        args, args.csv, "analysis/ngrams", "ngrams", "csv",
        explicit_override=getattr(args, 'output_csv', None)
    )

    df_raw, df_filtered, df_event_log = prepare_event_log(args.csv, args.events)
    if df_event_log is None:
        return

    n = getattr(args, 'n', 2)
    top_k = getattr(args, 'top_k', 20)
    min_count = getattr(args, 'min_count', 1)
    highlight = getattr(args, 'highlight_ngrams', False)

    print(f"\nTotal de registros lidos: {len(df_raw)}")
    print(f"Registros após filtragem de eventos ({args.events}): {len(df_filtered)}")
    print(f"\n=== Extraindo {n}-gramas de navegação ===")

    ngram_counter = Counter()
    total_sessions = 0
    sessions_with_ngram = Counter()
    sessions_skipped = 0

    for case_id, group in df_event_log.groupby('case:concept:name'):
        total_sessions += 1
        events = group.sort_values(by='time:timestamp')['concept:name'].tolist()

        if len(events) < n:
            sessions_skipped += 1
            continue

        ngrams = extract_ngrams(events, n)
        for ng in ngrams:
            ngram_counter[ng] += 1

        for ng in set(ngrams):
            sessions_with_ngram[ng] += 1

    if not ngram_counter:
        print(f"Erro: Nenhum {n}-grama encontrado (todas as sessões têm menos de {n} eventos).")
        return

    rows = []
    for ng, count in ngram_counter.most_common():
        rows.append({
            "ngram": format_ngram(ng),
            "n": n,
            "ocorrencias_total": count,
            "sessoes_com_ngram": sessions_with_ngram[ng],
            "pct_sessoes": round(sessions_with_ngram[ng] / total_sessions * 100, 1),
            "frequencia_media_por_sessao": round(count / total_sessions, 2),
        })

    df_out = pd.DataFrame(rows)

    if min_count > 1:
        df_out = df_out[df_out['ocorrencias_total'] >= min_count].reset_index(drop=True)

    total_distinct_before_topk = len(df_out)
    df_out = df_out.head(top_k).reset_index(drop=True)

    print(f"\nSessões analisadas: {total_sessions} ({sessions_skipped} ignoradas por terem menos de {n} eventos)")
    print(f"Total de {n}-gramas distintos encontrados (antes do corte top_k): {total_distinct_before_topk}")
    print(f"\n--- TOP {len(df_out)} {n}-GRAMAS MAIS FREQUENTES ---")
    for _, row in df_out.iterrows():
        print(f"  [{row['ocorrencias_total']:>4d}x | {row['pct_sessoes']:>5.1f}% sessões]  {row['ngram']}")

    # --- DFG combinado (apenas com os top_k n-gramas) ---
    print(f"\n=== Gerando DFG combinado dos top {len(df_out)} {n}-gramas ===")
    top_ngram_tuples = {}
    if n == 2:
        top_ngram_tuples = {
            tuple(row['ngram'].split(" -> ")): row['ocorrencias_total']
            for _, row in df_out.iterrows()
        }
    if n > 2:
        top_ngram_tuples = {
            ng: count for ng, count in ngram_counter.items()
            if format_ngram(ng) in set(df_out['ngram'].values)
        }
    dfg, start_acts, end_acts = build_dfg_from_ngrams(
        top_ngram_tuples if top_ngram_tuples else ngram_counter, n
    )
    print(f"DFG gerado: {len(dfg)} arestas, peso total = {sum(dfg.values())}")

    png_dfg = resolve_output_file_path(args, args.csv, "analysis/ngrams", "ngrams-dfg", "png")
    dfg_file = resolve_output_file_path(args, args.csv, "analysis/ngrams", "ngrams-dfg", "dfg")

    files_to_check = [output_csv, png_dfg, dfg_file]
    if not check_overwrite(files_to_check, args):
        return

    df_out.to_csv(output_csv, index=False)
    print(f"CSV salvo em: {os.path.abspath(output_csv)}")

    try:
        pm4py.save_vis_dfg(dfg, start_acts, end_acts, png_dfg, rankdir='TB')
        print(f"DFG PNG salvo em: {os.path.abspath(png_dfg)}")
    except Exception as e:
        print(f"Erro ao gerar PNG do DFG: {e}")

    try:
        pm4py.write_dfg(dfg, start_acts, end_acts, dfg_file)
        print(f"DFG (.dfg) salvo em: {os.path.abspath(dfg_file)}")
    except Exception as e:
        print(f"Erro ao exportar .dfg: {e}")

    # --- PNGs individuais com destaque (se --highlight_ngrams) ---
    if highlight:
        print(f"\n=== Gerando PNGs individuais com n-gramas destacados ===")
        # DFG completo (todos os n-gramas, não só os top_k)
        dfg_full, _, _ = build_dfg_from_ngrams(ngram_counter, n)
        highlight_dir = os.path.dirname(png_dfg)
        highlight_base = resolve_output_file_path(args, args.csv, "analysis/ngrams", "ngrams", "")
        highlight_base = os.path.splitext(highlight_base)[0]
        try:
            save_ngram_highlights(
                dfg_full, df_out, ngram_counter, n, total_sessions,
                highlight_dir,
                os.path.basename(highlight_base) if not highlight_base.endswith('-') else os.path.basename(highlight_base)[:-1]
            )
        except ImportError as e:
            print(f"Erro: bibliotecas necessárias não disponíveis ({e}). Instale networkx e matplotlib.")
        except Exception as e:
            print(f"Erro ao gerar PNGs de destaque: {e}")
