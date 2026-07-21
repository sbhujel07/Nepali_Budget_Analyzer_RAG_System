from sqlalchemy.ext.asyncio import AsyncSession
from app.database.schemas import SignupRequest,LoginRequest
from app.database.hashing import hash_password
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
    hashed_password = hash_password(request.password)

    new_user = User(name=request.name,email=request.email,hash_password=hashed_password)
    #save to db
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return {
        "message": "User created successfully"
    }

