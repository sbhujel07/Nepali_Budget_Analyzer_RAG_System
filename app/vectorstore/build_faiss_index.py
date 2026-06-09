import json
from config.config import CHUNKS_EMBEDDINGS
from app.retriever.topic_map import group_by_topic
from app.vectorstore.faiss_index import build_faiss
from app.retriever.bm25 import build_bm25


def read_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)



if __name__ == "__main__" :
    chunks = read_file(CHUNKS_EMBEDDINGS)
    topic_map = group_by_topic(chunks)

    #Save Topic_map also 
    with open("indexes/topic_map.json","w",encoding="utf-8") as f:
        json.dump(topic_map,f,ensure_ascii=False,indent=2)

    #yeuta global docs banauney jasma topic map ko sab values list ma store garney ani global faiss index and global bm25 index banauney
    global_docs = [item for docs in  topic_map.values() for item in docs] 

    #save index of faiss according to topic
    for topic,docs in topic_map.items():
        #save faiss
        build_faiss(
            docs,f"indexes/{topic}.faiss"
        )

        #save the Bm25 search also
        build_bm25(
            docs,f"indexes/{topic}.bm25.pkl"
        )

    #build global faiss and bm25 index
    build_faiss(global_docs,"indexes/global.faiss")
    build_bm25(global_docs,"indexes/global.bm25.pkl")

    print("All index built successfully!")





    ##Just for check if topic map is correct
    # print(type(topic_map))
    # print(len(topic_map))

    # for k in list(topic_map.keys())[:5]:
    #     print(k)

    # with open("indexes/topic_map.json","r",encoding="utf-8") as f:
    #     data = json.load(f)
    # print(data.keys())