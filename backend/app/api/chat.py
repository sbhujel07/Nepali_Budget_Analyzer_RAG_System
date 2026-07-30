from fastapi import APIRouter,Depends
from app.services.jwt_handler import oauth2_schema
from app.rag_pipeline import rag_pipeline
from app.services.jwt_handler import get_current_user
from app.database.schemas import ChatRequest,ChatResponse




router = APIRouter(prefix="/chat",tags=["Chat"])

@router.post("")
#Depend ley suru ma yo endpoint run huda Depend vitra ko func call gar vanxa
async def chat(request:ChatRequest,current_user: str = Depends(get_current_user)):
    
    answer = rag_pipeline(request.question,session_id=str(current_user.id))
    return ChatResponse(answer=answer)