from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.tables import Conversation
from app.database.schemas import ConversationCreate


async def create_conversation(request: ConversationCreate,user_id: int,db: AsyncSession):
    conversation = Conversation(user_id = user_id,title = request.title)
    db.add(conversation)

    await db.commit()

    await db.refresh(conversation)

    return conversation



#get the conversation history -> title of user
async def get_conversation(user_id: int,db: AsyncSession):
    result = await db.execute(select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc()))
    conversations = result.scalars().all()

    return conversations