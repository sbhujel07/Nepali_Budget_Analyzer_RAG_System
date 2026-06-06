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

    #Save Topic_map also 
    with open("indexes/topic_map.json","w",encoding="utf-8") as f:
        json.dump(topic_map,f,ensure_ascii=False)

    #save index of faiss according to topic
    for topic,docs in topic_map.items():
        build_faiss(
            docs,f"indexes/{topic}.faiss"
        )

        
    ##Just for check if topic map is correct
    # print(type(topic_map))
    # print(len(topic_map))

    # for k in list(topic_map.keys())[:5]:
    #     print(k)

    # with open("indexes/topic_map.json","r",encoding="utf-8") as f:
    #     data = json.load(f)
    # print(data.keys())