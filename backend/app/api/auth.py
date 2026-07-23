from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db_connection import get_db
from app.database.schemas import SignupRequest,LoginRequest
from app.services.auth_service import signup_user

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post("/signup")
async def signup(
    request: SignupRequest,
    db: AsyncSession = Depends(get_db)
):
    return await signup_user(request, db)



@router.post("/login")
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    return await login_user(request,db)