from sqlalchemy.ext.asyncio import AsyncSession
from app.database.schemas import SignupRequest,LoginRequest
from app.database.hashing import hashed_password,verify_password
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
    user = await db.execute(select(User).where(User.email == request.email))
    valid_email = user.scalar_one_or_none()
    if valid_email is None:
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

    