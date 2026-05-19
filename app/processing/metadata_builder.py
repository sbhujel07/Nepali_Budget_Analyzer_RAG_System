#Here add the metadata on the basis of topic
from scripts.build_index import cleaned_text_data,is_clean_sentence
from app.utils.splitter import split_nepali_sentences
from scripts.ingest_data import ingest_documents
from processing.keyword_map import keyword_map
from app.embeddings.topic_embedding_builder import build_topic_embeddings
from app.embeddings.model_embeddings import model


def build_metadata(
    cleaned_text_data,
    keyword_map,
    topics_list,
    topic_embeddings
):

    add_metadata_text = []

    for item in cleaned_text_data:

        page = item["Page_number"]

        sentences = item["Sentences"]

        for i, sentence in enumerate(sentences):

            # classify topic
            topic, score = classify_sentence(
                sentence,
                keyword_map,
                topics_list,
                topic_embeddings
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


if __name__ == "main":


    
    text_data = ingest_documents()
    cleaned_data = cleaned_text_data(
        text_data,
        split_nepali_sentences,
        is_clean_sentence)

    
    topic_list,topic_embeddings = build_topic_embeddings(model,keyword_map)
    add_metadata_text = build_metadata(cleaned_data,)
