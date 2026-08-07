from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
from app.database.tables import Conversation
from app.database.schemas import ConversationCreate
from sqlalchemy.exc import IntegrityError,SQLAlchemyError

async def create_conversation(request: ConversationCreate,user_id: int,db: AsyncSession):
    conversation = Conversation(user_id = user_id,title = request.title)
    db.add(conversation)
    try:
        await db.commit()

        await db.refresh(conversation)

        return conversation

    #execption handling for race condition
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Conversation could not be created"
        )
    #database error handle
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Database error occured"
        )


#get the conversation history -> title of user
async def get_conversation(user_id: int,db: AsyncSession):
    result = await db.execute(select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc()))
    conversations = result.scalars().all()

    return conversations