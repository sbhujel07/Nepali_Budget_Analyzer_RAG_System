from rank_bm25 import BM25Okapi
import numpy as np
import pickle

def build_bm25(docs,save_file):

    tokenized_doc = [item["text"].split() for item in docs]
    bm25 = BM25Okapi(tokenized_doc)

    # save bm25 file
    with open(save_file, "wb") as f:
        pickle.dump(bm25, f)
    
    print(f"Saved the Bm25 in {save_file}")

    return bm25