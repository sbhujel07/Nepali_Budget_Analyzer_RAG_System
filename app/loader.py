import json 
import faiss 

#load topic map
with open("indexes/topic_map.json","r",encoding="utf-8") as f:
    TOPIC_MAP = json.load(f)


#load faiss indexes
FAISS_INDEX = {}

for topic in TOPIC_MAP.keys():
    FAISS_INDEX[topic] = faiss.read_index(
        f"indexes/{topic}.faiss"
    )

print("All index loaded")