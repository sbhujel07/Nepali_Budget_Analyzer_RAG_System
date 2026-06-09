import json
import faiss
import numpy as np
from app.embeddings.model_embeddings import model
from app.retriever.topic_detect import detect_topic

from app.loader import TOPIC_MAP,FAISS_INDEX,BM25_INDEX

#Now hybrid Search bm25 + Faiss
def hybrid_search(user_query,model,top_k=5,alpha=0.5):
    print("\n================ HYBRID SEARCH START ================")
    print("Query:", user_query)

    #detect topic
    topic = detect_topic(user_query,TOPIC_MAP)
    print("Detected Topic:", topic)
    
    #gets the document of that topic (tyo topic ko sabai documents ligxa)
    topic_docs = TOPIC_MAP.get(topic,[])  #if that topic vetyo vani chai value dinxa
    

    #Build global docs
    #global docs ma topic_map को nested lists लाई flatten गरेर single list बनाउँछ like this  global_docs = [item1, item2, item3, item4, item5]
    global_docs = [item for docs in TOPIC_MAP.values() for item in docs] 

    ##Candidate selection 
    if len(topic_docs) == 0:
        Candidates = global_docs

    else:
        #combine topic docs + global_docs since sometimes multitopics in user query
        Candidates = topic_docs + global_docs
    
    #remove duplicate in candidates
    seen = set()
    unique_candidates = []

    for doc in Candidates:
        key = doc["text"]
        if key not in seen:
            unique_candidates.append(doc)
            seen.add(key)
    docs = unique_candidates
    if len(docs)== 0:
        print("Cannot find documents.")
        return []

    #BM25 search ,already loaded so
    bm25 = BM25_INDEX.get(docs)
    
    tokenized_query = user_query.split()
    bm25_score = bm25.get_scores(tokenized_query)

    #Normalize Bm25 scores(0 to 1)
    bm25_scores = np.array(bm25_score)
    bm25_scores = bm25_scores / (bm25_scores.max() + 1e-8)
    print("BM25 Scores (sample):", bm25_scores[:5])



    #Faisss Search
    index = FAISS_INDEX.get(docs)

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
