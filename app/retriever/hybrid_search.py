import json
import faiss
import numpy as np
from app.retriever.topic_map import group_by_topic
from config.config import CHUNKS_EMBEDDINGS
from app.embeddings.model_embeddings import model
from app.retriever.topic_detect import detect_topic
from app.retriever.bm25 import build_bm25
from app.vectorstore.faiss_index import build_faiss



#Now hybrid Search bm25 + Faiss
def hybrid_search(user_query,topic_map,model,top_k=5,alpha=0.5):
    #detect topic
    topic = detect_topic(user_query,topic_map)
    
    #gets the document of that topic (tyo topic ko sabai documents ligxa)
    docs = topic_map.get(topic,[])  #if that topic vetyo vani chai value dinxa

    if len(docs) == 0:
        return []

    #BM25 search
    bm25 = build_bm25(docs)
    tokenized_query = user_query.split()
    bm25_score = bm25.get_scores(tokenized_query)

    #Normalize Bm25 scores(0 to 1)
    bm25_scores = np.array(bm25_score)
    bm25_scores = bm25_scores / (bm25_scores.max() + 1e-8)


    #Faisss Search
    index = faiss.read_index(f"indexes/{topic}.faiss")

    user_query_vector = model.encode([user_query]).astype("float32")
    
    distance,indices = index.search(user_query_vector,len(docs))

    #convert distance to similarity
    faiss_scores = 1/(1+distance[0])

    #normalize FAISS scores
    faiss_scores = faiss_scores/(faiss_scores.max()+1e-8)

    #combine Scores 
    hybrid_scores = alpha * bm25_scores + (1-alpha)*faiss_scores
    #Rank top Results
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    results = [docs[i] for i in top_indices]

    return results


def read_file(file):
    with open(file,"r",encoding="utf-8") as f:
        data = json.load(f)
    return data
    


if __name__ == "__main__":

    user_query = "आगामी आर्थिक वर्षको लागि अनुमान गरिएको कुल सरकारी खर्च (बजेटको कुल आकार) कति हो र त्यसमध्ये चालु खर्चको हिस्सा कति प्रतिशत छ?"
    chunk_file = read_file(CHUNKS_EMBEDDINGS)
    topic_map = group_by_topic(chunk_file)
    retrieve_chunk = hybrid_search(user_query,topic_map,model)

    #print the retrieve chunks
    for item in retrieve_chunk:
        print("Text:",item["text"])
        print("TOPIC:",item["metadata"]["topic"])
        print("*"*50)
