# each sentence ko embedding garney

def sentence_embedding_func(cleaned_data, model):

    all_sentences = []

    for item in cleaned_data:
        all_sentences.extend(item["sentences"])

    sentence_embeddings = model.encode(
        all_sentences,
        batch_size=32,
        show_progress_bar=True
    )

    return all_sentences, sentence_embeddings