from sqlalchemy import Column,Integer,String,func,DateTime
from app.database.db_connection import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer,primary_key=True,index=True)
    name = Column(String,nullable=False)
    #must be unique and not null
    email = Column(String,unique=True,nullable=False) 
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
