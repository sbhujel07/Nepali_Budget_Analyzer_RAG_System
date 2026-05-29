# settings (model, chunk size etc.)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


#Budget Speech file
BUDGET_SPEECH_FILE = BASE_DIR / "data"/"processed"/"budget_speech_text.txt"

#open and add text file
BUDGET_SPEECH_JSON_FILE = BASE_DIR / "data"/"processed"/"budget_speech_txt.json"

#Clean budget data 
CLEAN_BUDGET_DATA = BASE_DIR / "data"/"processed"/"cleaned_budget_data_2081.json"

#Chunks file
CHUNKS_FILE = BASE_DIR / "data"/"processed"/"final_chunks_2081.json"

#Chunks Embeddings
CHUNKS_EMBEDDINGS = BASE_DIR / "data"/"processed"/"chunks_embeddings.json"



