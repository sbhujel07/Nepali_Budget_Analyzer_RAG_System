#Here add the metadata on the basis of topic
from scripts.build_index import cleaned_text_data,is_clean_sentence
from app.utils.splitter import split_nepali_sentences
from scripts.ingest_data import ingest_documents
from app.processing.keyword_map import keyword_map
from app.embeddings.topic_embedding_builder import build_topic_embeddings
from app.embeddings.sentence_embeddings import sentence_embedding_func
from app.embeddings.model_embeddings import model
from app.processing.topic_mapper import classify_sentence
from app.processing.metadata_cleaner import clean_metadata
import json
from config.config import CLEAN_BUDGET_DATA


def build_metadata(
    cleaned_text_data,
    keyword_map,
    topics_list,
    topic_embeddings,
    sentence_embeddings
):

    add_metadata_text = []
    #pointer for mapping embeddings
    idx = 0

    for item in cleaned_text_data:

        page = item["page_number"]
        sentences = item["sentences"] 

        for i, sentence in enumerate(sentences):

            sentence_embedding = sentence_embeddings[idx]
            idx += 1
            # classify topic
            topic, score = classify_sentence(
                sentence,
                keyword_map,
                topics_list,
                topic_embeddings,
                sentence_embedding
            )

            add_metadata_text.append({

                "text": sentence,
                "metadata": {

                    "page": page,
                    "sentence_id":
                        f"p{page}_s{i+1}",
                        "language": "ne",
                        "source":
                        "Budget Report 2081",
                        "topics": topic,
                        "scores":
                        round(score, 3)
                        }
            })

    return add_metadata_text


if __name__ == "__main__":

    text_data = ingest_documents()
    cleaned_data = cleaned_text_data(
        text_data,
        split_nepali_sentences,
        is_clean_sentence)

    
    topic_list,topic_embeddings = build_topic_embeddings(model,keyword_map)
    all_sentence,sentence_embeddings = sentence_embedding_func(cleaned_data,model)
    add_metadata_text = build_metadata(cleaned_data,keyword_map,topic_list,topic_embeddings,sentence_embeddings)
    print(add_metadata_text[1])

    #clean the metadata
    
    MIN_SCORE       = 0.20
    MIN_TEXT_LEN    = 15
    GARBAGE_TOKENS  = {"0 0", "0?", "॥०॥", "0० 0", "०?", "0 0 0"}

    cleaned_metadata = clean_metadata(add_metadata_text,MIN_SCORE,MIN_TEXT_LEN,GARBAGE_TOKENS)

    #now save to text file 
    #file path to save

    with open(CLEAN_BUDGET_DATA, "w", encoding="utf-8") as f:
        json.dump(cleaned_metadata, f, ensure_ascii=False, indent=2)

    print(f"file Saved to {CLEAN_BUDGET_DATA}")

