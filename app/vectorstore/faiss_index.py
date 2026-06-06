import faiss
import numpy as np
def build_faiss(docs,save_file):
    embeddings = np.array([doc["chunk_embeddings"] for doc in docs]).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])

    #add embedding to index
    index.add(embeddings)

    #save index
    faiss.write_index(index,save_file)

    print(f"Saved the faiss indexes in {save_file}")
    return index