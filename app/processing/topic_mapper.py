from app.processing.keyword_map import keyword_map,keyword_matching


#for embedding score
def embedding_score(sentence, topic, topics_list, topic_embeddings,sentence_embedding):
    """Get similarity score for a specific topic"""
    # sentence_embedding = model.encode([sentence])
    topic_index = topics_list.index(topic)
    similarity = cosine_similarity(sentence_embedding, [topic_embeddings[topic_index]])[0][0]
    return similarity

def hybrid_score(sentence, topic, keyword_map, topics_list, topic_embeddings,sentence_embedding):
    """Calculate combined keyword and embedding score for a specific topic"""
    keyword_scores = keyword_matching(sentence, topic, keyword_map)
    embedding_scores = embedding_score(sentence, topic, topics_list, topic_embeddings,sentence_embedding)
    
    # Normalize keyword scores to 0-1 range
    max_keywords = max(len(keywords) for keywords in keyword_map.values())
    normalized_keyword = (keyword_scores / max_keywords) if max_keywords > 0 else 0
    
    combined_score = 0.7 * normalized_keyword + 0.3 * embedding_scores
    return combined_score

#Now return the topic with highest score for each sentence
def classify_sentence(sentence, keyword_map, topics_list, topic_embeddings,sentence_embedding):
    """Classify sentence to the topic with highest score"""
    scores = {}
    # Calculate score for each topic
    for topic in keyword_map.keys():
        scores[topic] = hybrid_score(sentence, topic, keyword_map, topics_list, topic_embeddings,sentence_embedding)
    
    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]
    
    if best_score < 0.1:  # threshold for classification
        return "अन्य", best_score
    
    return best_topic, best_score