from sqlalchemy.ext.asyncio import AsyncSession
from app.database.schemas import SignupRequest,LoginRequest
from app.database.hashing import hashed_password,verify_password
from app.services.jwt_handler import create_access_token
from app.database.tables import User
from sqlalchemy import select
from fastapi import HTTPException



async def signup_user(request: SignupRequest, db: AsyncSession):
    # check if mail already exits
    result = await db.execute(select(User).where(User.email == request.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="This email is already registered"
        )


    #hash the password
    hash_password = hashed_password(request.password)

    new_user = User(name=request.name,email=request.email,hashed_password=hash_password)
    #save to db
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return {
        "message": "User created successfully"
    }

async def login_user(request: LoginRequest,db: AsyncSession):
    #verify email from user and select all if email is already signup
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    #verify the password
    password_is_valid = verify_password(request.password,user.hashed_password)
    if not password_is_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    #create access token
    access_token = create_access_token(data={"sub": user.email})
    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer"
    }