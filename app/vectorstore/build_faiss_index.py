import json
from config.config import CHUNKS_EMBEDDINGS
from app.retriever.topic_map import group_by_topic
from app.vectorstore.faiss_index import build_faiss


def read_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)



if __name__ == "__main__" :
    chunks = read_file(CHUNKS_EMBEDDINGS)
    topic_map = group_by_topic(chunks)

    #save index of faiss according to topic
    for topic,docs in topic_map.items():
        build_faiss(
            docs,f"indexes/{topic}.faiss"
        )