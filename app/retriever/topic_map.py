#At first lets make Topic map for token 
#सबै sentences लाई उनीहरूको topic अनुसार छुट्याएर groups बनाउँछ

import json
from collections import defaultdict
from config.config import CHUNKS_EMBEDDINGS

def read_file(file):
    with open(file,"r",encoding="utf-8") as f:
        data = json.load(f)
    return data

def group_by_topic(data):
    topic_map = defaultdict(list)

    for item in data:
        topics = item["metadata"].get("topic",[])

        #if there are no topics
        if len(topics) == 0:
            topic = "other"
        else:
            topic = topics[0]  #Primary topics

        topic_map[topic].append(item)

    return topic_map


##after this output topic map be like :
#     {
#   "economy": [doc1, doc3],
#   "health": [doc2],
#   "education": [doc4]
#     }


if __name__ == "__main__" :
    chunk_file = read_file(CHUNKS_EMBEDDINGS)
    topic_map = group_by_topic(chunk_file)
    #print

