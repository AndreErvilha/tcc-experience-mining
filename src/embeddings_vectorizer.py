# embeddings_vectorizer.py
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_minilm_embeddings(texts, batch_size=32, device='cpu'):
    print(f'[*] Carregando modelo local: all-MiniLM-L6-v2...')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print(f'[*] Gerando embeddings para {len(texts)} sessoes...')
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype(np.float32)

def get_gemini_embeddings(texts, api_key=None, model_name='models/gemini-embedding-001', batch_size=50):
    print(f'[*] Configurando Gemini API ({model_name})...')
    import google.generativeai as genai
    
    key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not key:
        raise ValueError('GEMINI_API_KEY ou GOOGLE_API_KEY nao encontrada nas variaveis de ambiente nem fornecida via --api-key.')
        
    genai.configure(api_key=key)
    
    all_embeddings = []
    print(f'[*] Solicitando embeddings para {len(texts)} sessoes via Gemini...')
    for i in tqdm(range(0, len(texts), batch_size), desc='Gemini Batches'):
        batch = texts[i:i+batch_size]
        # Garante textos validos
        batch = [t if t.strip() else 'empty_session' for t in batch]
        response = genai.embed_content(
            model=model_name,
            content=batch,
            task_type='CLUSTERING'
        )
        if hasattr(response, 'embeddings'):
            batch_emb = response.embeddings
        elif isinstance(response, dict) and 'embedding' in response:
            batch_emb = response['embedding']
        else:
            batch_emb = response
            
        # Converte lista de dicts ou floats
        if isinstance(batch_emb[0], dict) and 'values' in batch_emb[0]:
            batch_vecs = [item['values'] for item in batch_emb]
        else:
            batch_vecs = batch_emb
            
        all_embeddings.extend(batch_vecs)
        
    X = np.array(all_embeddings, dtype=np.float32)
    # Normaliza L2
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def vectorize_embeddings(input_csv: str, output_path: str, method: str = 'minilm', api_key: str = None, batch_size: int = 32, model_name: str = 'models/gemini-embedding-001'):
    print(f'[*] Carregando sessoes de: {input_csv}')
    df = pd.read_csv(input_csv)
    
    texts = df['text_representation'].fillna('empty').astype(str).tolist()
    session_ids = df['cd_sessao'].tolist()
    primary_labels = df['primary_label'].tolist() if 'primary_label' in df.columns else None
    experiences = df['experience'].tolist() if 'experience' in df.columns else None
    
    if method == 'minilm':
        X = get_minilm_embeddings(texts, batch_size=batch_size)
        vec_name = 'Embedding-MiniLM-L6-v2'
    elif method == 'gemini':
        X = get_gemini_embeddings(texts, api_key=api_key, model_name=model_name, batch_size=batch_size)
        vec_name = 'Embedding-Gemini'
    else:
        raise ValueError(f'Metodo desconhecido: {method}. Use minilm ou gemini.')
        
    print(f'[+] Embeddings gerados com shape: {X.shape}')
    
    data_dict = {
        'X': X,
        'session_ids': session_ids,
        'primary_labels': primary_labels,
        'experiences': experiences,
        'vectorizer_type': vec_name,
        'df': df
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
        
    np_path = output_path.replace('.pkl', '.npy') if output_path.endswith('.pkl') else f'{output_path}.npy'
    np.save(np_path, {'X': X, 'labels': primary_labels, 'experiences': experiences, 'session_ids': session_ids})
    
    print(f'[+] Salvo em: {output_path} e {np_path}')
    return data_dict

def main():
    parser = argparse.ArgumentParser(description='Gerador de embeddings semanticos para sessoes.')
    parser.add_argument('--input', default='data/sessions_represented.csv', help='CSV com sessoes')
    parser.add_argument('--output', default=None, help='Caminho do arquivo de saida (.pkl / .npy)')
    parser.add_argument('--method', choices=['minilm', 'gemini'], default='minilm', help='Modelo de embedding')
    parser.add_argument('--api-key', default=None, help='Chave Gemini API se method=gemini')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamanho do batch')
    parser.add_argument('--model-name', default='models/gemini-embedding-001', help='Modelo Gemini (o TCC usou models/gemini-embedding-001, 3072D)')
    args = parser.parse_args()
    
    out = args.output
    if out is None:
        out = f'data/{args.method}_embeddings.pkl'
        
    vectorize_embeddings(args.input, out, method=args.method, api_key=args.api_key, batch_size=args.batch_size, model_name=args.model_name)

if __name__ == '__main__':
    main()
