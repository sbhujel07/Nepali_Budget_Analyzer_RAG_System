#first read the json file and group the similar topics

import json
from collections import defaultdict


def read_file(file):
    with open(file,"r",encoding="utf-8") as f:
        data = json.load(f)

    return data


#सबै sentences लाई उनीहरूको topic अनुसार छुट्याएर groups बनाउँछ

def group_by_topic(data):
    topic_map = defaultdict(list)

    for item in data:
        topics = item["metadata"].get("topics",[])

        #if there are no topics
        if len(topics) == 0:
            topic = "other"
        else:
            topic = topics[0]  #Primary topics

        topic_map[topic].append(item)

    return topic_map

#chunking using the sliding window
def create_topic_chunks(topic_map, chunk_size=3, overlap=1):
    all_chunks = []
    step = chunk_size - overlap

    for topic, sentences in topic_map.items():

        for i in range(0, len(sentences), step):

            chunk_items = sentences[i:i + chunk_size]

            chunk_text = " ".join([x["text"] for x in chunk_items])

            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "topic": topic,
                    "page_numbers": list(set([x["metadata"]["page"] for x in chunk_items])),
                    "sentence_ids": [x["metadata"]["sentence_id"] for x in chunk_items],
                    "source": "Budget Report 2081"
                }
            })

    return all_chunks



if __name__ == "__main__" :
    
    file_path = "data/processed/cleaned_budget_data_2081.json"
    read_file_data = read_file(file_path)
    topic_map = group_by_topic(read_file_data)

    # print(topic_map)

    #chunking
    Chunks = create_topic_chunks(topic_map)
    print(Chunks[1])

