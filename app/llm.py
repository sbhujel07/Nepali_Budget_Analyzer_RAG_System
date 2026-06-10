#set the environment variable for Groq API key
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing")

llm = ChatGroq(model = "llama-3.1-8b-instant",temperature = 0.1)
# response = llm.invoke("नेपालमा सबैभन्दा अग्लो हिमाल कुन हो??").content
# print(response)
