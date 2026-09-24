# shap_explainer.py
import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

def explain_clusters(tfidf_pkl="data/tfidf_vectors.pkl", cluster_pkl="clusters/hdbscan_tf_idf.pkl", output_dir="results/xai_shap"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Carregando dados TF-IDF e Clusters...")
    with open(tfidf_pkl, "rb") as f:
        tfidf_data = pickle.load(f)
    with open(cluster_pkl, "rb") as f:
        cluster_data = pickle.load(f)
        
    X = tfidf_data["X"]
    feature_names = np.array(tfidf_data["feature_names"])
    cluster_labels = np.array(cluster_data["cluster_labels"])
    
    # Filtra inst?ncias de ru?do (-1)
    valid_mask = (cluster_labels != -1)
    X_valid = X[valid_mask]
    y_valid = cluster_labels[valid_mask]
    
    unique_classes = sorted(list(set(y_valid)))
    class_map = {c: i for i, c in enumerate(unique_classes)}
    inv_class_map = {i: c for i, c in enumerate(unique_classes)}
    y_mapped = np.array([class_map[c] for c in y_valid])
    
    print(f"[*] Treinando XGBClassifier surrogate sobre {len(X_valid)} sessoes e {len(unique_classes)} clusters...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )
    clf.fit(X_valid, y_mapped)
    
    train_acc = clf.score(X_valid, y_mapped)
    print(f"[+] Acuracia do surrogate XGBoost: {train_acc*100:.2f}%")
    
    print("[*] Calculando valores SHAP exatos via TreeExplainer...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_valid)
    
    cluster_signatures = {}
    
    for cls_idx, original_cluster_id in inv_class_map.items():
        if isinstance(shap_values, list):
            sv_class = shap_values[cls_idx]
        else:
            sv_class = shap_values[:, :, cls_idx] if len(shap_values.shape) == 3 else shap_values
            
        cls_mask = (y_mapped == cls_idx)
        if np.sum(cls_mask) == 0:
            continue
            
        mean_shap_in_cluster = np.mean(sv_class[cls_mask], axis=0)
        
        top_pos_idx = np.argsort(mean_shap_in_cluster)[::-1][:5]
        top_pos = [
            {"token": feature_names[idx], "shap_value": float(mean_shap_in_cluster[idx])}
            for idx in top_pos_idx if mean_shap_in_cluster[idx] > 0.001
        ]
        
        top_neg_idx = np.argsort(mean_shap_in_cluster)[:3]
        top_neg = [
            {"token": feature_names[idx], "shap_value": float(mean_shap_in_cluster[idx])}
            for idx in top_neg_idx if mean_shap_in_cluster[idx] < -0.001
        ]
        
        cluster_signatures[int(original_cluster_id)] = {
            "cluster_id": int(original_cluster_id),
            "size": int(np.sum(cls_mask)),
            "top_positive_features": top_pos,
            "top_negative_features": top_neg
        }
        
    out_json = os.path.join(output_dir, "shap_cluster_signatures.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cluster_signatures, f, indent=2)
        
    print(f"[+] Assinaturas SHAP salvas em {out_json}")
    
    print("\n" + "=" * 80)
    print("ASSINATURAS COMPORTAMENTAIS SHAP POR CLUSTER (HDBSCAN + TF-IDF)")
    print("=" * 80)
    for c_id, sig in sorted(cluster_signatures.items()):
        pos_tokens = [f"{item['token']} (+{item['shap_value']:.2f})" for item in sig["top_positive_features"][:3]]
        print(f"[Cluster {c_id:02d}] (N={sig['size']}): {', '.join(pos_tokens)}")
        
    return cluster_signatures

if __name__ == "__main__":
    explain_clusters()
