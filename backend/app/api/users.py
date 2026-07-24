from fastapi import APIRouter,Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.db_connection import get_db
from app.database.tables import User 
from app.database.schemas import UserCreate,UserOut,UserUpdate
from sqlalchemy import select

router = APIRouter(
    prefix="/user",
    tags=["Users"]
)


#create user
@router.post("/users")
async def create_user(user: UserCreate,db: AsyncSession = Depends(get_db)):
    #schemas through user ley post gareko name and email chai db users tables ma store hunxa
    new_user = User(name = user.name,email = user.email)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    #return the object -> sqlalchemy object
    return new_user




#get only one user
@router.get("/users/{user_id}",response_model = UserOut)
async def get_user(user_id:int ,db: AsyncSession = Depends(get_db)):
    results = await db.execute(select(User).where(User.id == user_id))
    user = results.scalar_one_or_none()
    return user



#get the  all users
@router.get("/users",response_model= list[UserOut]) #for the validation like password hash haru sab client lai return hunxa if userout model rakhena vani
async def get_users(db: AsyncSession = Depends(get_db)):
    results = await db.execute(select(User))
    user = results.scalars().all()
    return user



#update the users in db
@router.put("/users/{user_id}",response_model=UserOut)
async def update_user(user_id: int,user_update:UserUpdate,db: AsyncSession = Depends(get_db)):
    results = await db.execute(select(User).where(User.id == user_id))
    user = results.scalar_one_or_none() #if one return else none
    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found."
        )
    #update
    user.name = user_update.name
    user.email = user_update.email

    #save
    await db.commit()
    await db.refresh(user)

    return user

