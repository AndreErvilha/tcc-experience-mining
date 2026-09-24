# tfidf_vectorizer.py
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_tfidf(input_csv: str, output_path: str, max_features: int = None, min_df: int = 1):
    print(f'[*] Carregando sessoes de: {input_csv}')
    df = pd.read_csv(input_csv)
    
    texts = df['text_representation'].fillna('').astype(str).tolist()
    session_ids = df['cd_sessao'].tolist()
    primary_labels = df['primary_label'].tolist() if 'primary_label' in df.columns else None
    experiences = df['experience'].tolist() if 'experience' in df.columns else None
    
    print(f'[*] Vetorizando com TF-IDF (token_pattern=r"\S+", lowercase=False)...')
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'\S+',
        max_features=max_features,
        min_df=min_df
    )
    
    X_sparse = vectorizer.fit_transform(texts)
    X = X_sparse.toarray().astype(np.float32)
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    print(f'[+] Matriz TF-IDF construida com shape: {X.shape}')
    
    data_dict = {
        'X': X,
        'feature_names': feature_names,
        'session_ids': session_ids,
        'primary_labels': primary_labels,
        'experiences': experiences,
        'vectorizer_type': 'TF-IDF',
        'df': df
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
        
    # Salva tambem como npy para compatibilidade simples
    np_path = output_path.replace('.pkl', '.npy') if output_path.endswith('.pkl') else f'{output_path}.npy'
    np.save(np_path, {'X': X, 'labels': primary_labels, 'experiences': experiences, 'session_ids': session_ids})
    
    print(f'[+] Vetores salvos em: {output_path} e {np_path}')
    return data_dict

def main():
    parser = argparse.ArgumentParser(description='Vetorizacao TF-IDF de sessoes.')
    parser.add_argument('--input', default='data/sessions_represented.csv', help='CSV gerado por events2vector.py')
    parser.add_argument('--output', default='data/tfidf_vectors.pkl', help='Arquivo de saida (.pkl / .npy)')
    parser.add_argument('--max-features', type=int, default=None, help='Numero maximo de features')
    parser.add_argument('--min-df', type=int, default=1, help='Frequencia minima do termo')
    args = parser.parse_args()
    
    vectorize_tfidf(args.input, args.output, args.max_features, args.min_df)

if __name__ == '__main__':
    main()
