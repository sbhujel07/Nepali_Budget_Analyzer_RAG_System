#first read the json file and group the similar topics

import json
from collections import defaultdict
from config.config import CLEAN_BUDGET_DATA
from config.config import CHUNKS_FILE


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


    #filter the chunks less than 5 words and must have atleast one sentence
def filter_chunks(Chunks,minimum_word):

    filtered_chunks = [chunk for chunk in Chunks
        if chunk.get("text")
        and chunk["text"].strip()
        and len(chunk["text"].split()) >= minimum_word  #chunks has atleast 5 word
        and chunk.get("metadata") is not None ]  #must have atleast 1 sentence

    return filtered_chunks

#save the chunking to the json file
def save_to_json(file_path,data):
     with open(file_path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

     print(f"file saved to {file_path} successfully.")





if __name__ == "__main__" :
    

    read_file_data = read_file(CLEAN_BUDGET_DATA)
    topic_map = group_by_topic(read_file_data)

    # print(topic_map)

    #chunking
    Chunks = create_topic_chunks(topic_map)
    # print(Chunks[1])

    #now filter the chunks
    filtered_chunks = filter_chunks(Chunks,5)
    # print(filtered_chunks[1])
    # print(len(Chunks))
    # print(len(filtered_chunks))

    #save filtered_chunk to file for the accessibility
    # file_path = "data/processed/final_chunks_2081.json"
    save_to_json(CHUNKS_FILE,filtered_chunks)

   


