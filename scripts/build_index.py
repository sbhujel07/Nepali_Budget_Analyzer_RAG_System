#split text and remove the uneccesary words in budget speech

from app.utils.helper import split_nepali_sentences
from scripts.ingest_data import ingest_documents

text_data = ingest_documents()



REMOVE_KEYWORDS = [
    "सम्माननीय",
    "सभामुख",
    "अध्यक्ष",
    "शुभकामना",
    "श्रद्धाञ्जली",
    "शहीद",
    "उपस्थित भएको छु"
]


def is_clean_sentence(sentence):
    for word in REMOVE_KEYWORDS:
        if word in sentence:
            return False
    return True


def cleaned_text_data(data, split_fn, clean_fn):
    """
    Splits sentences, removes unwanted ones, and enriches metadata.
    """

    cleaned_data = []

    for item in data:

        sentences = split_fn(item["text"])

        cleaned_sentences = [
            s for s in sentences if clean_fn(s)
        ]

        cleaned_data.append({
            "page_number": item["page_number"],
            "text": item["text"],
            "sentences": cleaned_sentences,
            "sentence_count": len(cleaned_sentences)
        })

    return cleaned_data


cleaned_data = cleaned_text_data(
    text_data,
    split_nepali_sentences,
    is_clean_sentence)

print(cleaned_data[0])
