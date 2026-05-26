# load + chunk documents
import re
from pathlib import Path
from config.config import BUDGET_SPEECH_FILE

def clean_text(text):
    # remove page markers like --- Page 1 ---
    text = re.sub(r"--- Page \d+ ---", "", text)
    
    # remove standalone page numbers (lines with only numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    
    # remove OCR garbage like .१10.80४.110
    text = re.sub(r"\.\d*[०-९\d./]+", "", text)
    
    # remove multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)
    
    # remove extra spaces
    text = re.sub(r"\s{2,}", " ", text)
    
    return text.strip()





def load_processed_pages(file_path: str):

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    text_data = []
    pattern = r'--- Page (\d+) ---\s*(.*?)(?=--- Page \d+ ---|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    for page_num, page_text in matches:
        text_data.append({
            "page_number": int(page_num),
            "text": clean_text(page_text)
        })

    return text_data


def ingest_documents():

    return  load_processed_pages(BUDGET_SPEECH_FILE)




if __name__ == "__main__":

    data = ingest_documents()
    # Check result
    print(f"Total pages: {len(data)}")
    print(data[0])
    print(data[-1])


    