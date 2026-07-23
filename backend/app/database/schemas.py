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
    name: str
    email: EmailStr
    password: str


# Login Request
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

