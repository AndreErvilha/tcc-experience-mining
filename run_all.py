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

def main():
    print("=" * 80)
    print(" REPRODUCAO COMPLETA DO PIPELINE EXPERIMENTAL DE EXPERIENCE MINING")
    print(" MBA Gestao de Negocios Digitais e IA - USP/ESALQ")
    print("=" * 80)
    
    # 1. Extracao e Representacao de Eventos
    run_step(
        "PASSO 1: Representacao Textual das Sessoes (events2vector)",
        "events2vector.py",
        ["--input", "dataset/processed/total-rotulado.csv", "--output", "data/sessions_represented.csv"]
    )
    
    # 2. Vetorizacoes
    run_step(
        "PASSO 2.1: Vetorizacao Lexica Ponderada (TF-IDF)",
        "tfidf_vectorizer.py",
        ["--input", "data/sessions_represented.csv", "--output", "data/tfidf_vectors.pkl"]
    )
    
    run_step(
        "PASSO 2.2: Vetorizacao Neural Local (Sentence-BERT all-MiniLM-L6-v2)",
        "embeddings_vectorizer.py",
        ["--input", "data/sessions_represented.csv", "--output", "data/minilm_embeddings.pkl", "--method", "minilm"]
    )
    
    # 3. Clusterizacao e Avaliacao: TF-IDF
    run_step("PASSO 3.1: K-Means sobre TF-IDF", "kmeans_cluster.py", ["--input", "data/tfidf_vectors.pkl", "--output", "clusters/kmeans_tf_idf.pkl", "--auto-k"])
    run_step("PASSO 3.2: Metricas K-Means TF-IDF", "eval_metrics.py", ["--input", "clusters/kmeans_tf_idf.pkl", "--output", "results/metrics/metrics_kmeans_tf_idf.json"])
    run_step("PASSO 3.3: HDBSCAN sobre TF-IDF", "hdbscan_cluster.py", ["--input", "data/tfidf_vectors.pkl", "--output", "clusters/hdbscan_tf_idf.pkl", "--auto-grid"])
    run_step("PASSO 3.4: Metricas HDBSCAN TF-IDF", "eval_metrics.py", ["--input", "clusters/hdbscan_tf_idf.pkl", "--output", "results/metrics/metrics_hdbscan_tf_idf.json"])
    
    # 4. Clusterizacao e Avaliacao: MiniLM
    run_step("PASSO 4.1: K-Means sobre MiniLM", "kmeans_cluster.py", ["--input", "data/minilm_embeddings.pkl", "--output", "clusters/kmeans_minilm.pkl", "--auto-k"])
    run_step("PASSO 4.2: Metricas K-Means MiniLM", "eval_metrics.py", ["--input", "clusters/kmeans_minilm.pkl", "--output", "results/metrics/metrics_kmeans_minilm.json"])
    run_step("PASSO 4.3: HDBSCAN sobre MiniLM", "hdbscan_cluster.py", ["--input", "data/minilm_embeddings.pkl", "--output", "clusters/hdbscan_minilm.pkl", "--auto-grid"])
    run_step("PASSO 4.4: Metricas HDBSCAN MiniLM", "eval_metrics.py", ["--input", "clusters/hdbscan_minilm.pkl", "--output", "results/metrics/metrics_hdbscan_minilm.json"])
    
    # 5. Clusterizacao e Avaliacao: Gemini (se pre-computado)
    gemini_path = ROOT / "data" / "gemini_embeddings.pkl"
    if gemini_path.exists():
        run_step("PASSO 5.1: K-Means sobre Gemini (3072D)", "kmeans_cluster.py", ["--input", "data/gemini_embeddings.pkl", "--output", "clusters/kmeans_gemini.pkl", "--auto-k"])
        run_step("PASSO 5.2: Metricas K-Means Gemini", "eval_metrics.py", ["--input", "clusters/kmeans_gemini.pkl", "--output", "results/metrics/metrics_kmeans_gemini.json"])
        run_step("PASSO 5.3: HDBSCAN sobre Gemini (3072D)", "hdbscan_cluster.py", ["--input", "data/gemini_embeddings.pkl", "--output", "clusters/hdbscan_gemini.pkl", "--auto-grid"])
        run_step("PASSO 5.4: Metricas HDBSCAN Gemini", "eval_metrics.py", ["--input", "clusters/hdbscan_gemini.pkl", "--output", "results/metrics/metrics_hdbscan_gemini.json"])
    else:
        print("[!] data/gemini_embeddings.pkl nao encontrado. Pulei etapa do Gemini.")
        
    # 6. Explicabilidade SHAP
    run_step("PASSO 6: Explainable AI (SHAP TreeExplainer + Assinaturas)", "shap_explainer.py", [])
    
    # 7. Consolidacao do sumario de metricas
    consolidate_summary()
    
    print(f"\n{'='*80}\n[+] REPRODUCAO FINALIZADA COM SUCESSO DE PONTA A PONTA!\n{'='*80}")

if __name__ == "__main__":
    main()
