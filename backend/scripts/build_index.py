#split text and remove the uneccesary words in budget speech
import logging
import json
from app.utils.splitter import split_nepali_sentences
from config.config import BUDGET_SPEECH_JSON_FILE
from config.logging_config import setup_logging



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


def cleaned_and_add_metadata( data,split_fn, clean_fn):
    """
    Splits sentences, removes unwanted ones, and enriches metadata.
    """
    logger.info("Started builder Pipeline")
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

    logger.info("Pipeline Complited, cleaned text, add metadata like text,sentence etc")

    return cleaned_data

def read_file(file):
    logger.info(f"Reading the file: {file}")
    with open(file,"r",encoding="utf-8") as f:
        data =  json.load(f)
    logger.info("completed reading")
    return data




if __name__ == "__main__":

    setup_logging() #initialize the logging func
    logger = logging.getLogger(__name__)

    #reading file 
    text_data = read_file(BUDGET_SPEECH_JSON_FILE)

    cleaned_data = cleaned_and_add_metadata(
        text_data,
        split_nepali_sentences,
        is_clean_sentence)
    #print the data
    print(cleaned_data[0])
