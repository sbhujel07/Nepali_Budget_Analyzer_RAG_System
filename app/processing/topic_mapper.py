from app.processing.keyword_map import keyword_map,keyword_matching


#for embedding score
def embedding_score(sentence, topic, topics_list, topic_embeddings):
    """Get similarity score for a specific topic"""
    sentence_embedding = model.encode([sentence])
    topic_index = topics_list.index(topic)
    similarity = cosine_similarity(sentence_embedding, [topic_embeddings[topic_index]])[0][0]
    return similarity


if __name__ == "main" :

    #embedding model ma pathauney vanda aagadi hamley list lai string ma lanuparxa so
    topic_sentences = {topic:" ".join(keywords) for topic, keywords in keyword_map.items()}

    topic_embeddings = model.encode(list(topic_sentences.values()))

    # Store the list of topics to use for indexing
    topics_list = list(keyword_map.keys())
        