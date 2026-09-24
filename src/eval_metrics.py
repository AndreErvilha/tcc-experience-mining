# eval_metrics.py
import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_completeness_v_measure
)

def load_cluster_result(path: str):
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    elif path.endswith('.npy'):
        raw = np.load(path, allow_pickle=True).item()
        return raw
    else:
        raise ValueError(f'Formato desconhecido: {path}')

def calculate_purity_real_experience(cluster_labels, experiences):
    # Metodologia TCC vs1 (Pagina 11):
    # Pureza(C) = max(|success_C|, |attrition_C| + |error_C|) / (|success_C| + |attrition_C| + |error_C|)
    if experiences is None or len(experiences) == 0:
        return None, {}
        
    unique_clusters = [c for c in set(cluster_labels) if c != -1]
    if not unique_clusters:
        return None, {}
        
    cluster_purities = {}
    total_valid_samples = 0
    weighted_purity_sum = 0.0
    
    for c in unique_clusters:
        mask = (cluster_labels == c)
        c_exps = [experiences[i] for i in range(len(experiences)) if mask[i]]
        
        # Filtra 'none' se houver
        valid_exps = [e for e in c_exps if e != 'none']
        if not valid_exps:
            continue
            
        n_success = sum(1 for e in valid_exps if e == 'success')
        n_negative = sum(1 for e in valid_exps if e in ['attrition', 'error'])
        
        total_c = n_success + n_negative
        if total_c == 0:
            continue
            
        purity_c = max(n_success, n_negative) / total_c
        cluster_purities[int(c)] = {
            'size': total_c,
            'success': n_success,
            'negative': n_negative,
            'purity': float(purity_c)
        }
        
        weighted_purity_sum += purity_c * total_c
        total_valid_samples += total_c
        
    global_purity = (weighted_purity_sum / total_valid_samples) if total_valid_samples > 0 else None
    return global_purity, cluster_purities

def evaluate_clustering(cluster_data: dict):
    cluster_labels = np.array(cluster_data['cluster_labels'])
    X = cluster_data.get('X')
    primary_labels = cluster_data.get('primary_labels') or cluster_data.get('labels')
    experiences = cluster_data.get('experiences')
    algorithm = cluster_data.get('algorithm', 'Unknown')
    vectorizer = cluster_data.get('vectorizer_type', 'Unknown')
    
    n_total = len(cluster_labels)
    n_noise = int(np.sum(cluster_labels == -1))
    noise_ratio = float(n_noise / n_total)
    
    unique_clusters = set(cluster_labels) - {-1}
    k_discovered = len(unique_clusters)
    
    metrics = {
        'algorithm': algorithm,
        'vectorizer': vectorizer,
        'n_samples': n_total,
        'k_discovered': k_discovered,
        'noise_count': n_noise,
        'noise_ratio': noise_ratio,
    }
    
    # 1. Silhouette Score
    if X is not None and k_discovered >= 2:
        valid_mask = (cluster_labels != -1)
        if np.sum(valid_mask) > k_discovered:
            metrics['silhouette_score'] = float(silhouette_score(X[valid_mask], cluster_labels[valid_mask]))
        else:
            metrics['silhouette_score'] = None
    else:
        metrics['silhouette_score'] = None
        
    # 2. M?tricas supervisionadas contra os r?tulos de jornada (ARI, NMI, V-Measure)
    if primary_labels is not None:
        primary_labels = np.array(primary_labels)
        # Completo (incluindo ru?do como classe -1)
        metrics['ari_all'] = float(adjusted_rand_score(primary_labels, cluster_labels))
        metrics['nmi_all'] = float(normalized_mutual_info_score(primary_labels, cluster_labels))
        
        # Excluindo ru?do
        valid_mask = (cluster_labels != -1)
        if np.sum(valid_mask) > 1 and len(set(primary_labels[valid_mask])) > 1:
            metrics['ari_clean'] = float(adjusted_rand_score(primary_labels[valid_mask], cluster_labels[valid_mask]))
            metrics['nmi_clean'] = float(normalized_mutual_info_score(primary_labels[valid_mask], cluster_labels[valid_mask]))
            h, c, v = homogeneity_completeness_v_measure(primary_labels[valid_mask], cluster_labels[valid_mask])
            metrics['homogeneity'] = float(h)
            metrics['completeness'] = float(c)
            metrics['v_measure'] = float(v)
        else:
            metrics['ari_clean'] = None
            metrics['nmi_clean'] = None
    else:
        metrics['ari_all'] = None
        metrics['nmi_all'] = None
        
    # 3. Pureza por Experi?ncia Real (Metodologia TCC vs1)
    if experiences is not None:
        global_purity, c_details = calculate_purity_real_experience(cluster_labels, experiences)
        metrics['purity_real_experience'] = global_purity
        metrics['cluster_purity_details'] = c_details
    else:
        metrics['purity_real_experience'] = None
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Calcula metricas de avaliacao de clusterizacao.')
    parser.add_argument('--input', required=True, help='Arquivo de resultados de cluster (.pkl ou .npy)')
    parser.add_argument('--output', default=None, help='Arquivo JSON para salvar o relatorio')
    args = parser.parse_args()
    
    data = load_cluster_result(args.input)
    metrics = evaluate_clustering(data)
    
    out = args.output
    if out is None:
        base = os.path.splitext(args.input)[0]
        out = f'{base}_metrics.json'
        
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    print(f'[+] Metricas de Avaliacao para: {args.input}')
    print('=' * 60)
    for k, v in metrics.items():
        if k != 'cluster_purity_details':
            print(f'  {k:<26}: {v}')
    print('=' * 60)
    print(f'[+] Salvo em: {out}')

if __name__ == '__main__':
    main()
