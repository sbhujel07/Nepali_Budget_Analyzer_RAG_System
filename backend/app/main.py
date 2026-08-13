
from fastapi import FastAPI, Depends,HTTPException
from sqlalchemy import select
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.db_connection import Base,get_db,engine,Sessionlocal
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.users import router as user_router
from app.api.conversation import router as conversation_router
from app.api.auth import signup_user,login_user
from app.database.schemas import UserCreate
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from app.core.exception_handler import http_exception_handler,validation_exception_handler,global_exception_handler
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#for the access not to be blocked when sending request from frontend to backend 
app.add_middleware(
    CORSMiddleware,
    #for frontend request
    allow_origins =  "http://localhost:5173",
                    "https://nepali-budget-analyzer-rag-system.vercel.app",
                   
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

#conversation history router
app.include_router(conversation_router)


#Exception handler
app.add_exception_handler(HTTPException,http_exception_handler)

app.add_exception_handler(RequestValidationError,validation_exception_handler)

app.add_exception_handler(Exception,global_exception_handler)




