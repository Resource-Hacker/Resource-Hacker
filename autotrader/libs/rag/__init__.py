import faiss, numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384
index = faiss.IndexFlatIP(DIM)
docs: List[str] = []

def add(texts:List[str]): 
    global docs
    embs = _model.encode(texts, normalize_embeddings=True)
    index.add(np.asarray(embs,dtype="float32"))
    docs.extend(texts)

def query(q:str,k:int=5)->List[str]:
    if index.ntotal==0: return []
    emb = _model.encode([q], normalize_embeddings=True)
    D,I = index.search(np.asarray(emb,dtype="float32"),k)
    return [docs[i] for i in I[0] if i < len(docs)]
