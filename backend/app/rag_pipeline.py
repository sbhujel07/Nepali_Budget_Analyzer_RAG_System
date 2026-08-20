from langchain_core.messages import SystemMessage,HumanMessage
from app.retriever.hybrid_search import hybrid_search
from app.embeddings.model_embeddings import model
from app.prompts.prompt_template import prompt_formatter
from app.llm.llm_setup import chat_with_memory
from app.llm.groq_llm import llm

def rag_pipeline(user_query,session_id):
    retrieved_chunks = hybrid_search(user_query)
    prompt = prompt_formatter(user_query,retrieved_chunks)

    config = {
         "configurable":{
             "session_id" : session_id
         }
     }

    response = chat_with_memory.invoke(
         [SystemMessage(content= prompt),
         HumanMessage(content= user_query)
         ],
         config=config
     )

    return response.content

    # print("\n================ PROMPT SENT TO LLM ================\n")
    # print(prompt)
    # print("\n======================================================\n")

    # response = llm.invoke(prompt)
    # print(response.content)
    # return response.content