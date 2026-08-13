# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

import numpy as np
from app.setting import HF_TOKEN
from huggingface_hub import InferenceClient

class EmbeddingModel:
    def __init__(self):
        self.client = InferenceClient(provider="hf-inference",api_key=HF_TOKEN)
        self.model_name = ("sentence-transformers/"
                            "paraphrase-multilingual-MiniLM-L12-v2"
        )


    def encode(self,texts):
        #for single strings
        if isinstance(texts,str):
            embedding = self.client.feature_extraction(texts,model=self.model_name)
            return np.asrray(embedding,dtype=np.float32)

        
        #for list of texts
        embeddings=[]

        for text in texts:
            embedding=self.client.feature_extraction(text,model=self.model_name)
            embeddings.append(np.asarray(embedding, dtype=np.float32))
        

        return np.asarray(embeddings, dtype=np.float32)


model = EmbeddingModel()