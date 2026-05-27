
import json
from app.embeddings.model_embeddings import model
from config.config import CHUNKS_FILE
from config.config import CHUNKS_EMBEDDINGS


#read the chunking from chunk json

def read_chunks(file):

    with open(file,"r",encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks

#embedding the chunks
def chunk_embeds(data,model):
    items = [item["text"] for item in data]
    chunk_embedding = model.encode(items)

    for item,embeds in zip(data,chunk_embedding):
        item["chunk_embeddings"] = embeds.tolist()
    
    return data

#save embeddings to the json file
def save_vectors_to_json(file,data):
    with open(file,"w" ,encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

    print(f"file saved to {file} successfully")


if __name__ == "__main__" :
    # file_path = "data/processed/final_chunks_2081.json"
    chunks = read_chunks(CHUNKS_FILE)
    chunks_embedding = chunk_embeds(chunks,model)
    # print(chunks_embedding[1])

    #now save vector to the json file
    # file_to_save = "data/processed/chunks_embeddings.json"
    save_vectors_to_json(CHUNKS_EMBEDDINGS,chunks_embedding)



