from datetime import datetime
from sqlalchemy import Column,Integer,String,func,DateTime,ForeignKey
from app.database.db_connection import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer,primary_key=True,index=True)
    name = Column(String,nullable=False)
    #must be unique and not null
    email = Column(String,unique=True,nullable=False) 
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


#for the conversation message
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer,primary_key=True,index=True,)
    user_id = Column(Integer,ForeignKey("users.id", ondelete="CASCADE"),nullable=False,index=True,)
    title = Column(String(255),nullable=False,)
    created_at = Column(DateTime,default=datetime.utcnow,nullable=False,)
    #for the latest search
    updated_at = Column(DateTime,default=datetime.utcnow,onupdate=datetime.utcnow,nullable=False,)