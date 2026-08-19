# #set the environment variable for Groq API key
# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")

# if not groq_api_key:
#     raise ValueError("GROQ_API_KEY is missing")

# llm = ChatOpenAI(model = "gpt-5.4-mini",temperature = 0.1)
# response = llm.invoke("नेपालमा सबैभन्दा अग्लो हिमाल कुन हो??").content
# print(response)


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
base_url="https://api-direct.apikey.cloud/openai/v1"

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing")

llm = ChatOpenAI(
    model="gpt-5.4",
    temperature=0.1,
    api_key=openai_api_key,
    base_url=base_url,
    max_retries=0,
)

# response = llm.invoke("नेपालमा सबैभन्दा अग्लो हिमाल कुन हो??").content
# print(response)