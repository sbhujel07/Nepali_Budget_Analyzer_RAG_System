#topic ko embeddings and store the list of topics

from app.embeddings import model
from app.processing import keyword_map

def build_topic_embeddings(keyword_map, model):
    topic_sentences = {
        topic: " ".join(keywords)
        for topic, keywords in keyword_map.items()
    }

    topic_embeddings = model.encode(list(topic_sentences.values()))

    topics_list = list(keyword_map.keys())

    return topics_list, embeddings