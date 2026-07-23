import os
from dotenv import load_dotenv
import asyncio
from sqlalchemy.ext.asyncio  import create_async_engine,AsyncSession
from sqlalchemy.orm import sessionmaker,declarative_base
from app.setting import DATABASE_URL

# load_dotenv()
# DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)

#we use sessionlocal for not to db connection again and again
Sessionlocal = sessionmaker(
    bind = engine,
    class_= AsyncSession,
    expire_on_commit=False

)
#this is base class,database know creating tables,columns and rows inside tables through this
Base = declarative_base()

#database session
async def get_db():
    #establish a db session
    #query garna ko lagi db session connection chahinxa so we need session
    async with Sessionlocal() as session:
        #route and end session using yield
        yield session
