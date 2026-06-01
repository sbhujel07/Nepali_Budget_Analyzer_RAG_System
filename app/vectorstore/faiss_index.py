import faiss
import numpy as np
def build_faiss(docs):
    embeddings = np.array([doc["Chunk_Embedding"] for doc in docs]).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])

    #add embedding to index
    index.add(embeddings)

    return index