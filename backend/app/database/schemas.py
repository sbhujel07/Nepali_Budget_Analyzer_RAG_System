from pydantic import BaseModel, ConfigDict,EmailStr,Field
from datetime import date, datetime

class UserCreate(BaseModel):
    name : str = Field(min_length=3,max_length=50)
    email : EmailStr

class UserOut(BaseModel):
    id : int
    name : str
    email : EmailStr

    model_config = ConfigDict(from_attributes=True)

class UserUpdate(BaseModel):
    name : str=Field(min_length=3,max_length=50)
    email : EmailStr



# Signup Request
class SignupRequest(BaseModel):
    name: str = Field(min_length=3,max_length=50)
    email: EmailStr
    password: str = Field(min_length=8)


# Login Request
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

#input chat request
class ChatRequest(BaseModel):
    question: str


#output chat request
class ChatResponse(BaseModel):
    answer: str


#for user conversation
class ConversationCreate(BaseModel):
    title: str


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True