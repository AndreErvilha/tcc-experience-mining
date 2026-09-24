# stream_inference.py
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ExperienceClusterEngine:
    def __init__(self, tfidf_pkl="data/tfidf_vectors.pkl", cluster_pkl="clusters/hdbscan_tf_idf.pkl"):
        print("[*] Carregando modelo base para inferencia continua...")
        with open(tfidf_pkl, "rb") as f:
            self.tfidf_data = pickle.load(f)
        with open(cluster_pkl, "rb") as f:
            self.cluster_data = pickle.load(f)
            
        self.X = self.tfidf_data["X"]
        self.labels = np.array(self.cluster_data["cluster_labels"])
        self.unique_clusters = sorted([c for c in set(self.labels) if c != -1])
        
        # Calcula os centroides normalizados de cada cluster
        self.centroids = {}
        for c in self.unique_clusters:
            mask = (self.labels == c)
            c_mean = np.mean(self.X[mask], axis=0)
            norm = np.linalg.norm(c_mean)
            self.centroids[c] = c_mean / (norm if norm > 0 else 1.0)
            
        print(f"[+] Engine pronta com {len(self.centroids)} centroides calibrados.")
        
    def classify_new_session(self, vector_repr, novelty_threshold=0.30):
        # vector_repr: vetor 1D da sessao
        norm = np.linalg.norm(vector_repr)
        v = vector_repr / (norm if norm > 0 else 1.0)
        
        best_cluster = -1
        best_similarity = -1.0
        
        for c, centroid in self.centroids.items():
            sim = float(np.dot(v, centroid))
            if sim > best_similarity:
                best_similarity = sim
                best_cluster = c
                
        is_novelty = (best_similarity < novelty_threshold)
        
        return {
            "assigned_cluster": int(best_cluster) if not is_novelty else -1,
            "cosine_similarity": float(best_similarity),
            "status": "NOVELTY_ALERT" if is_novelty else "CLASSIFIED"
        }

if __name__ == "__main__":
    engine = ExperienceClusterEngine()
    # Teste com uma sessao de exemplo do Cluster 5
    sample_vec = engine.X[50]
    res = engine.classify_new_session(sample_vec)
    print(f"Resultado da inferencia de teste em streaming:\n{res}")
