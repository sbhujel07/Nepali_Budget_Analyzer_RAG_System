import json
import faiss
import numpy as np
from app.embeddings.model_embeddings import model
from app.retriever.topic_detect import detect_topic

from app.loader import TOPIC_MAP,FAISS_INDEX,BM25_INDEX,GLOBAL_DOCS

#Now hybrid Search bm25 + Faiss
def hybrid_search(user_query,model,top_k=5,alpha=0.5):
    print("\n================ HYBRID SEARCH START ================")
    print("Query:", user_query)

    #detect topic
    topic = detect_topic(user_query,TOPIC_MAP)
    print("Detected Topic:", topic)
    
    if topic == "other" or topic not in TOPIC_MAP:
        #fallback to global
        docs = GLOBAL_DOCS
        bm25 = BM25_INDEX["global"]
        index = FAISS_INDEX["global"]

    else:
        docs = TOPIC_MAP[topic]
        bm25 = BM25_INDEX[topic]
        index = FAISS_INDEX[topic]

    #bm25 search
    tokenized_query = user_query.split()
    bm25_score = bm25.get_scores(tokenized_query)

    #Normalize Bm25 scores(0 to 1)
    bm25_scores = np.array(bm25_score)
    bm25_scores = bm25_scores / (bm25_scores.max() + 1e-8)
    print("BM25 Scores (sample):", bm25_scores[:5])



    #Faisss Search

    user_query_vector = model.encode([user_query]).astype("float32")
    
    distance,indices = index.search(user_query_vector,len(docs))

    #convert distance to similarity
    faiss_scores = 1/(1+distance[0])

    #normalize FAISS scores
    faiss_scores = faiss_scores/(faiss_scores.max()+1e-8)
    print("FAISS Scores (sample):", faiss_scores[:5])


    #combine Scores 
    hybrid_scores = alpha * bm25_scores + (1-alpha)*faiss_scores
    print("Hybrid Scores (sample):", hybrid_scores[:5])

    #Rank top Results
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    results = [docs[i] for i in top_indices]

    print("============= HYBRID SEARCH END =============\n")


    return results





if __name__ == "__main__":

    user_query = "आगामी आर्थिक वर्षको लागि अनुमान गरिएको कुल सरकारी खर्च (बजेटको कुल आकार) कति हो र त्यसमध्ये चालु खर्चको हिस्सा कति प्रतिशत छ?"
    retrieve_chunk = hybrid_search(user_query,model)

    #print the retrieve chunks
    for item in retrieve_chunk:
        print("Text:",item["text"])
        print("TOPIC:",item["metadata"]["topic"])
        print("*"*50)
