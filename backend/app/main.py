# #api entrypoint (FastAPI)

# from app.rag_pipeline import rag_pipeline


# def main():
#     print("RAG Chatbot Started \n")

#     session_id = input("Input the session id/user_name: ")

#     while True:
#         user_query = input(f"\n {session_id}:  ")

#         if user_query.lower() in ["exit","quit"]:
#             print("Thankyou! GoodDay")
#             break

#         else:
#             response = rag_pipeline(user_query,session_id)

#             print("\n Bot:", response)

# if __name__ == "__main__":
#     main()

from fastapi import FastAPI, Depends,HTTPException
from sqlalchemy import select
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.db_connection import Base,get_db,engine,Sessionlocal
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.users import router as user_router
from app.api.auth import signup_user,login_user
from app.database.schemas import UserCreate
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#for the access not to be blocked when sending request from frontend to backend 
app.add_middleware(
    CORSMiddleware,
    #for frontend request
    allow_origins =  "http://localhost:5173",
                   
    #for cookies and all
    allow_credentials = True,
    #for https methods post,get etc
    allow_methods=["*"],
    allow_headers=["*"],
)

#create startup
@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


#users router
app.include_router(user_router)

#auth router
app.include_router(auth_router)

#chat router
app.include_router(chat_router)



