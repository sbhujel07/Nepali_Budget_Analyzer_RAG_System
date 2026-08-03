from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.db_connection import get_db
from app.database.tables import User
from app.services.jwt_handler import get_current_user
from app.database.schemas import ConversationResponse,ConversationCreate
from app.database.crud_conversation import create_conversation,get_conversation




router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)

#create a new conversation and save to db -> like user_id and title etc for user history
@router.post("",response_model=ConversationResponse)
async def create_new_conversation(request:ConversationCreate,db: AsyncSession = Depends(get_db),current_user: User = Depends(get_current_user)):
    return await create_conversation(request=request,user_id=current_user.id,db=db)


#get the user conversation history
@router.get("",response_model=list[ConversationResponse])
async def get_all_conversations(db: AsyncSession=Depends(get_db),current_user: User=Depends(get_current_user)):
    return await get_conversation(user_id=current_user.id,db=db)
