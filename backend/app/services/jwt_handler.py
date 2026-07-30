from datetime import datetime,date,timezone,timedelta
from fastapi import  Depends,HTTPException,status
from sqlalchemy import select
from jose import JWTError
from app.database.tables import User
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.db_connection import get_db
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


def verify_access_token(token: str):
    payload = jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
    return payload


#make oauth schema -> kun endpoint bata token fetch garney vanera
#frontend bata localstorage ma save vako token+bearer hunxa so oauth ley token matra nikalxa like yesto; Authorization: Bearer token_key
oauth2_schema = OAuth2PasswordBearer(tokenUrl="/auth/login")



async def get_current_user(token: str = Depends(oauth2_schema),db: AsyncSession = Depends(get_db)):
    try:
        #verify the jwt token
        payload = verify_access_token(token)

        #extract email from token
        email = payload.get("sub")

        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
            )

        #find user in database
    result = await db.execute(select(User).where(User.email==email))

    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user
