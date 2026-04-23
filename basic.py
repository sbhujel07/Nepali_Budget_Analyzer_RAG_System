from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# store memory per user/session
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# base model
llm = ChatOllama(model="llama3.2", temperature=0.0)

# wrap with memory
chat_with_memory = RunnableWithMessageHistory(
    llm,
    get_session_history,
)

config = {"configurable": {"session_id": "user1"}}

while True:
    question = input("Enter your question: ")
    if question.lower() == "exit":
        break

    response = chat_with_memory.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            ("user", question)
        ],
        config=config
    )

    print("AI:", response.content)