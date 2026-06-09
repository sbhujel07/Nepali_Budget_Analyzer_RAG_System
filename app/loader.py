import json 
import faiss 
import pickle

#load topic map
with open("indexes/topic_map.json","r",encoding="utf-8") as f:
    TOPIC_MAP = json.load(f)


#load faiss indexes
FAISS_INDEX = {}

for topic in TOPIC_MAP.keys():
    FAISS_INDEX[topic] = faiss.read_index(
        f"indexes/{topic}.faiss"
    )

#global faiss index
FAISS_INDEX["global"] = faiss.read_index(
    "indexes/global.faiss"
)

print("All faiss index loaded")

#load Bm25 
BM25_INDEX = {}

for topic in TOPIC_MAP.keys():
    with open(f"indexes/{topic}.bm25.pkl","rb")as f:
        BM25_INDEX[topic] = pickle.load(f)

#global bm25 index
with open("indexes/global.bm25.pkl","rb") as f:
    BM25_INDEX["global"] = pickle.load(f)

print("All bm25 index loaded")



# yesto hunxa output =>
# BM25_INDEX = {
#     "अर्थतन्त्र": <BM25Okapi object>,
#     "कृषि": <BM25Okapi object>,
#     "शिक्षा": <BM25Okapi object>,
# }