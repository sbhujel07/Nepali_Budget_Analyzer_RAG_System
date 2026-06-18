import os
from dotenv import load_dotenv
import asyncio
from sqlalchemy.ext.asyncio  import create_async_engine,AsyncSession
from sqlalchemy.orm import sessionmaker,declarative_base

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)

#we use sessionlocal for not to db connection again and again
Sessionlocal = sessionmaker(
    bind = engine,
    class_= AsyncSession,
    expire_on_commit=False

)
#this is base class,database know creating tables,columns and rows inside tables through this
Base = declarative_base()

async def test_connection():
    async with engine.connect() as conn:
        print("Database connected successfully")


if __name__ == "__main__" :
   asyncio.run(test_connection())