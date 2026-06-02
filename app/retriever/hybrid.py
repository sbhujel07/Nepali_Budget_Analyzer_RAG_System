#Now hybrid Search bm25 + Faiss
def hybrid_search(user_query,topic_map,model,top_k=5,alpha=0.5):
    #detect topic
    topic = detect_topic(user_query,topic_map)
    
    #gets the document of that topic (tyo topic ko sabai documents ligxa)
    docs = topic_map.get(topic,[])

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
    index = build_faiss(docs)

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
    