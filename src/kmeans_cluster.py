# kmeans_cluster.py
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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

def find_best_k(X, k_min=2, k_max=15, random_state=42):
    print(f'[*] Buscando K otimo via Silhouette Score no intervalo [{k_min}, {k_max}]...')
    best_k = k_min
    best_score = -1.0
    scores = {}
    
    for k in range(k_min, min(k_max + 1, len(X))):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f'    k={k} -> Silhouette: {score:.4f}')
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f'[+] Melhor K selecionado: {best_k} (Silhouette = {best_score:.4f})')
    return best_k, scores

def run_kmeans(input_path: str, output_path: str, n_clusters: int = 7, auto_k: bool = False, random_state: int = 42):
    print(f'[*] Carregando matriz de recursos de: {input_path}')
    data = load_data(input_path)
    X = data['X']
    
    if auto_k:
        k_val, _ = find_best_k(X, k_min=2, k_max=15, random_state=random_state)
    else:
        k_val = n_clusters
        
    print(f'[*] Executando KMeans (k={k_val}, random_state={random_state}, init=k-means++)...')
    kmeans = KMeans(n_clusters=k_val, init='k-means++', random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    sil_score = None
    if len(set(cluster_labels)) > 1:
        sil_score = float(silhouette_score(X, cluster_labels))
        print(f'[+] Silhouette Score: {sil_score:.4f}')
        
    results = {
        'algorithm': 'K-Means',
        'k': k_val,
        'cluster_labels': cluster_labels,
        'cluster_centers': kmeans.cluster_centers_,
        'inertia': float(kmeans.inertia_),
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
        'algorithm': 'K-Means',
        'k': k_val,
        'cluster_labels': cluster_labels,
        'silhouette_score': sil_score,
        'X': X,
        'labels': results['primary_labels'],
        'experiences': results['experiences']
    })
    
    print(f'[+] Resultados do K-Means salvos em: {output_path} e {np_path}')
    return results

def main():
    parser = argparse.ArgumentParser(description='Clusterizacao K-Means.')
    parser.add_argument('--input', required=True, help='Caminho dos vetores (.pkl ou .npy)')
    parser.add_argument('--output', required=True, help='Caminho de saida (.pkl ou .npy)')
    parser.add_argument('--n-clusters', type=int, default=7, help='Numero de clusters K (padrao 7 do baseline)')
    parser.add_argument('--auto-k', action='store_true', help='Determinar K otimo via Silhouette Score')
    parser.add_argument('--seed', type=int, default=42, help='Seed aleatoria')
    args = parser.parse_args()
    
    run_kmeans(args.input, args.output, n_clusters=args.n_clusters, auto_k=args.auto_k, random_state=args.seed)

if __name__ == '__main__':
    main()
