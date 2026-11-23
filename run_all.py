# ==============================================================================
# Experience Mining Pipeline - Master Replication Script
# TCC MBA em Gestao de Negocios Digitais e IA - USP/ESALQ (2026)
# Autor: Andre Eduardo Ribeiro Ervilha
# ==============================================================================
import os
import sys
import glob
import json
import time
import argparse
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.resolve()
PYTHON = sys.executable

def run_step(desc, script_name, args):
    print(f"\n{'='*80}\n[*] {desc}\n{'='*80}")
    script_path = ROOT / "src" / script_name
    cmd = [PYTHON, str(script_path)] + args
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"[!] Erro ao executar {script_name} (codigo: {res.returncode})")
        sys.exit(res.returncode)
    dt = time.time() - t0
    print(f"[+] Concluido em {dt:.2f}s")

def consolidate_summary():
    print(f"\n{'='*80}\n[*] CONSOLIDANDO RESULTADOS DOS EXPERIMENTOS\n{'='*80}")
    metrics_files = glob.glob(str(ROOT / "results" / "metrics" / "metrics_*.json"))
    
    rows = []
    for mf in sorted(metrics_files):
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
            exp_name = Path(mf).stem.replace("metrics_", "").upper().replace("_", " ")
            
            silh = data.get("silhouette_score")
            silh_str = f"{silh:.4f}" if silh is not None else "-"
            
            ari_clean = data.get("ari_clean")
            ari_str = f"{ari_clean:.4f}" if ari_clean is not None else "-"
            
            v_meas = data.get("v_measure")
            v_str = f"{v_meas:.4f}" if v_meas is not None else "-"
            
            purity = data.get("purity_real_experience")
            purity_str = f"{purity*100:.2f}%" if purity is not None else "-"
            
            noise = data.get("noise_ratio", 0.0)
            noise_str = f"{noise*100:.1f}%"
            
            rows.append({
                "Experimento": exp_name,
                "Vetorizador": data.get("vectorizer", "-"),
                "Algoritmo": data.get("algorithm", "-"),
                "K": data.get("k_discovered", "-"),
                "Ruido (%)": noise_str,
                "Silhouette": silh_str,
                "ARI (s/ ruido)": ari_str,
                "V-Measure": v_str,
                "Pureza Real": purity_str
            })
            
    df = pd.DataFrame(rows)
    if not df.empty:
        csv_out = ROOT / "results" / "summary_metrics.csv"
        df.to_csv(csv_out, index=False)
        print(f"[+] Sumario exportado para: {csv_out.relative_to(ROOT)}")
        print("\n" + df.to_string(index=False))
    else:
        print("[!] Nenhum arquivo de metrica encontrado para consolidar.")

ROTULOS = "data/sessions_represented.csv"  # rotulo da Tabela 1 revisada (gerar_massas.py -> events2vector.py)
MASSA = "dataset/processed/total-rotulado-v1.1.csv"
MODELOS = [("tf_idf", "data/tfidf_vectors.pkl", "TF-IDF"),
           ("minilm", "data/minilm_embeddings.pkl", "MiniLM"),
           ("gemini", "data/gemini_embeddings.pkl", "Gemini (3072D)")]
ANALISES = ROOT / "analises"  # analises citadas na errata e no README


def run_script(desc, path, args=()):
    """Roda um script fora de src/ (análises da revisão pós-entrega)."""
    print(f"\n{'='*80}\n[*] {desc}\n{'='*80}")
    t0 = time.time()
    res = subprocess.run([PYTHON, str(path), *args], cwd=str(ROOT))
    if res.returncode != 0:
        print(f"[!] Erro ao executar {Path(path).name} (codigo: {res.returncode})")
        sys.exit(res.returncode)
    print(f"[+] Concluido em {time.time() - t0:.2f}s")


def main():
    ap = argparse.ArgumentParser(description="Reproducao do pipeline de Experience Mining.")
    ap.add_argument("--reusar-agrupamentos", action="store_true",
                    help="Nao refaz vetorizacao nem agrupamento: usa data/*.pkl e clusters/*.pkl existentes "
                         "e refaz rotulos, metricas, SHAP, jornadas por grupo e figuras.")
    ap.add_argument("--sem-revisao", action="store_true",
                    help="Pula a comparacao v1.0 x v1.1 (analises/rotulo_revisado.py) e a conferencia do rotulo.")
    ap.add_argument("--rotular-massa", action="store_true",
                    help="Antes de tudo, rotula de novo o dataset/processed/total.csv com o dataset/gerar_massas.py "
                         "(regras da Tabela 1; requer o pm4py-cli). Sem esta opcao, usa o total-rotulado-v1.1.csv existente.")
    ap.add_argument("--sem-shap", action="store_true",
                    help="Pula o SHAP (nao depende dos rotulos; pesos, figuras e beeswarms SHAP ficam como estao).")
    ap.add_argument("--petri", action="store_true",
                    help="Gera tambem as Redes de Petri e DFGs por grupo (pm4py + Graphviz; demorado).")
    args = ap.parse_args()

    print("=" * 80)
    print(" REPRODUCAO COMPLETA DO PIPELINE EXPERIMENTAL DE EXPERIENCE MINING")
    print(" MBA Gestao de Negocios Digitais e IA - USP/ESALQ")
    print("=" * 80)

    # 1. Representacao textual e rotulo das sessoes
    if args.rotular_massa:
        run_script("PASSO 0: Rotulagem da massa pela Tabela 1 (dataset/gerar_massas.py)", ROOT / "dataset" / "gerar_massas.py",
                   ["--skip_discovery_original", "--skip_playout", "--skip_discovery_simulado", "--usar_total_existente",
                    "--saida", os.path.basename(MASSA)])

    # A massa rotulada vem do dataset/gerar_massas.py (regras da Tabela 1: gatilho -> conversao)
    run_step("PASSO 1: Representacao Textual e rotulo das Sessoes (events2vector)", "events2vector.py",
             ["--input", MASSA, "--output", "data/sessions_represented.csv"])

    # 2. Vetorizacoes e agrupamentos (os rotulos nao entram aqui)
    if not args.reusar_agrupamentos:
        run_step("PASSO 2.1: Vetorizacao Lexica Ponderada (TF-IDF)", "tfidf_vectorizer.py",
                 ["--input", "data/sessions_represented.csv", "--output", "data/tfidf_vectors.pkl"])
        run_step("PASSO 2.2: Vetorizacao Neural Local (Sentence-BERT all-MiniLM-L6-v2)", "embeddings_vectorizer.py",
                 ["--input", "data/sessions_represented.csv", "--output", "data/minilm_embeddings.pkl", "--method", "minilm"])
        if not (ROOT / "data" / "gemini_embeddings.pkl").exists():
            print("[!] data/gemini_embeddings.pkl nao encontrado. Pulei a etapa do Gemini.")
            print("    Para gerar (requer GEMINI_API_KEY no ambiente; nunca versione o .env):")
            print("    python src/embeddings_vectorizer.py --method gemini --model-name models/gemini-embedding-001 --output data/gemini_embeddings.pkl")
        for nome, entrada, rotulo in MODELOS:
            if not (ROOT / entrada).exists():
                continue
            run_step(f"PASSO 3: K-Means sobre {rotulo}", "kmeans_cluster.py",
                     ["--input", entrada, "--output", f"clusters/kmeans_{nome}.pkl", "--auto-k"])
            run_step(f"PASSO 3: HDBSCAN sobre {rotulo}", "hdbscan_cluster.py",
                     ["--input", entrada, "--output", f"clusters/hdbscan_{nome}.pkl", "--auto-grid"])
    else:
        print("\n[*] --reusar-agrupamentos: vetorizacao e agrupamento nao refeitos (clusters/*.pkl existentes).")

    # 4. Metricas de avaliacao (rotulo da Tabela 1)
    for nome, _, rotulo in MODELOS:
        for algo in ("kmeans", "hdbscan"):
            pkl = f"clusters/{algo}_{nome}.pkl"
            if (ROOT / pkl).exists():
                run_step(f"PASSO 4: Metricas {algo.upper()} {rotulo}", "eval_metrics.py",
                         ["--input", pkl, "--output", f"results/metrics/metrics_{algo}_{nome}.json", "--labels", ROTULOS])

    # 5. Explicabilidade SHAP
    if not args.sem_shap:
        run_step("PASSO 5: Pesos SHAP de cada grupo, nos 6 modelos (results/jornadas_por_grupo/)", "shap_explainer.py", [])
        run_step("PASSO 5.1: Figuras SHAP (Figuras 4 e 5 do TCC, beeswarm de cada grupo, sensibilidade colsample)", "shap_figures.py", [])

    # 6. Jornadas por grupo (CSVs por grupo, resumos e matrizes grupos x rotulo)
    run_step("PASSO 6: Jornadas por grupo (results/jornadas_por_grupo/)", "jornadas_por_grupo.py", ["--labels", ROTULOS])
    run_step("PASSO 6.1: Tabelas por jornada (results/dados/jornadas_contagem.csv, results/metrics/cobertura_por_jornada.csv)", "jornadas_tabelas.py", [])
    if args.petri:
        run_step("PASSO 6.2: DFG e Redes de Petri por grupo (results/jornadas_por_grupo/<modelo>/grupos/)", "petri_por_grupo.py", [])

    # 7. Comparacao com a versao entregue e conferencia do rotulo (citadas na errata)
    if not args.sem_revisao:
        run_script("PASSO 7.1: Rotulo v1.0 x v1.1 (results/metrics/comparacao_v1.0_v1.1.csv)", ANALISES / "rotulo_revisado.py")
        run_script("PASSO 7.2: Conferencia independente do rotulo (Python puro x pm4py-cli)", ANALISES / "conferir_rotulo_tabela1.py")

    # 8. Consolidacao do sumario de metricas
    consolidate_summary()
    run_step("PASSO 9: Figuras de resultados (results/figures/)", "figuras_tcc.py", [])

    print(f"\n{'='*80}\n[+] REPRODUCAO FINALIZADA COM SUCESSO DE PONTA A PONTA!\n{'='*80}")


if __name__ == "__main__":
    main()
