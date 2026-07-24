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



#create user
@app.post("/users")
async def create_user(user: UserCreate,db: AsyncSession = Depends(get_db)):
    #schemas through user ley post gareko name and email chai db users tables ma store hunxa
    new_user = User(name = user.name,email = user.email)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    #return the object -> sqlalchemy object
    return new_user


#auth router
app.include_router(auth_router)



