# hdbscan_cluster.py
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score

def load_data(input_path: str):
    if input_path.endswith('.pkl'):
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        return data
    elif input_path.endswith('.npy'):
        raw = np.load(input_path, allow_pickle=True).item()
        return raw
    else:
        raise ValueError(f'Formato nao suportado: {input_path}')

def grid_search_hdbscan(X, min_cluster_sizes=(5, 10, 15, 20), metric='euclidean'):
    print(f'[*] Grid search HDBSCAN sobre min_cluster_size in {min_cluster_sizes} (metric={metric})...')
    best_params = None
    best_score = -1.0
    best_clusterer = None
    
    for mcs in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric=metric, core_dist_n_jobs=-1)
        labels = clusterer.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.mean(labels == -1)
        
        # Silhouette apenas nos pontos que nao sao ruido
        valid_mask = labels != -1
        if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
            score = silhouette_score(X[valid_mask], labels[valid_mask])
        else:
            score = -1.0
            
        print(f'    min_cluster_size={mcs} -> Clusters: {n_clusters}, Ruido: {noise_ratio*100:.1f}%, Silhouette (sem ruido): {score:.4f}')
        
        # Criterio: maximiza silhouette com penalidade se ruido for excessivo (>70%)
        effective_score = score * (1.0 - 0.5 * noise_ratio)
        if effective_score > best_score:
            best_score = effective_score
            best_params = {'min_cluster_size': mcs, 'metric': metric}
            best_clusterer = clusterer
            
    print(f'[+] Melhores parametros: {best_params}')
    return best_clusterer, best_params

def run_hdbscan(input_path: str, output_path: str, min_cluster_size: int = 5, min_samples: int = None, metric: str = 'euclidean', auto_grid: bool = False):
    print(f'[*] Carregando matriz de recursos de: {input_path}')
    data = load_data(input_path)
    X = data['X']
    
    if auto_grid:
        clusterer, params = grid_search_hdbscan(X, min_cluster_sizes=[5, 10, 15, 20], metric=metric)
        min_cluster_size = params['min_cluster_size']
    else:
        print(f'[*] Executando HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric={metric})...')
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            core_dist_n_jobs=-1
        )
        clusterer.fit(X)
        
    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))
    noise_ratio = float(np.mean(labels == -1))
    
    valid_mask = labels != -1
    sil_score = None
    if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
        sil_score = float(silhouette_score(X[valid_mask], labels[valid_mask]))
        
    print(f'[+] Clusters formados: {n_clusters} | Pontos de ruido: {noise_count} ({noise_ratio*100:.1f}%)')
    if sil_score is not None:
        print(f'[+] Silhouette Score (excluindo ruido): {sil_score:.4f}')
        
    results = {
        'algorithm': 'HDBSCAN',
        'n_clusters': n_clusters,
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'metric': metric,
        'cluster_labels': labels,
        'probabilities': clusterer.probabilities_,
        'noise_ratio': noise_ratio,
        'silhouette_score': sil_score,
        'X': X,
        'session_ids': data.get('session_ids'),
        'primary_labels': data.get('primary_labels') or data.get('labels'),
        'experiences': data.get('experiences'),
        'vectorizer_type': data.get('vectorizer_type', 'unknown')
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    np_path = output_path.replace('.pkl', '.npy') if output_path.endswith('.pkl') else f'{output_path}.npy'
    np.save(np_path, {
        'algorithm': 'HDBSCAN',
        'n_clusters': n_clusters,
        'cluster_labels': labels,
        'noise_ratio': noise_ratio,
        'silhouette_score': sil_score,
        'X': X,
        'labels': results['primary_labels'],
        'experiences': results['experiences']
    })
    
    print(f'[+] Resultados do HDBSCAN salvos em: {output_path} e {np_path}')
    return results

def main():
    parser = argparse.ArgumentParser(description='Clusterizacao HDBSCAN.')
    parser.add_argument('--input', required=True, help='Caminho dos vetores (.pkl ou .npy)')
    parser.add_argument('--output', required=True, help='Caminho de saida (.pkl ou .npy)')
    parser.add_argument('--min-cluster-size', type=int, default=5, help='Tamanho minimo do cluster')
    parser.add_argument('--min-samples', type=int, default=None, help='Amostras minimas de densidade')
    parser.add_argument('--metric', default='euclidean', choices=['euclidean', 'cosine', 'manhattan'], help='Metrica de distancia')
    parser.add_argument('--auto-grid', action='store_true', help='Realizar grid search de parametros')
    args = parser.parse_args()
    
    run_hdbscan(args.input, args.output, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, metric=args.metric, auto_grid=args.auto_grid)

if __name__ == '__main__':
    main()
