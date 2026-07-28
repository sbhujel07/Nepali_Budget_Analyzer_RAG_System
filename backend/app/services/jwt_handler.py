from datetime import datetime,date,timezone,timedelta
from app.setting import SECRET_KEY,ALGORITHM,ACCESS_TOKEN_EXPIRE_MINUTES
from fastapi.security import OAuth2PasswordBearer
from jose import jwt

def create_access_token(data: dict):
    to_encode = data.copy() #copy the dict data

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    #put the expiry time in the dict data
    to_encode.update({"exp": expire})
    #here generate access key using the secret key algorithm and expiry time
    encoded_jwt = jwt.encode(to_encode,SECRET_KEY,algorithm=ALGORITHM)
    return encoded_jwt


