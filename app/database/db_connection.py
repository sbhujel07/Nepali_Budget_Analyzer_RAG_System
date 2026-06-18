import os
from dotenv import load_dotenv
import asyncio
from sqlalchemy.ext.asyncio  import create_async_engine

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)

async def test_connection():
    async with engine.connect() as conn:
        print("Database connected successfully")


if __name__ == "__main__" :
   asyncio.run(test_connection())