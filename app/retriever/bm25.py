from rank_bm25 import BM25Okapi
import numpy as np

def build_bm25(docs):

    tokenized_doc = [item["text"].split() for item in docs]
    bm25 = BM25Okapi(tokenized_doc)

    return bm25