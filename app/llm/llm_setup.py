
#llm + memory integration
from app.llm.groq_llm import llm
from app.memory.session_memory import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory

chat_with_memory = RunnableWithMessageHistory(
    llm,get_session_history
)