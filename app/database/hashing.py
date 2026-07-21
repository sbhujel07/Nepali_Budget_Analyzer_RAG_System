from passlib.context import CryptContext




pwd_context = CryptContext(
    schemes= ["bcrypt"],
    deprecated="auto"
)

# hash password from plain text to encrypt
def hash_password(password: str)  -> str:
    return pwd_context.hash(password)

#change hasing pw to plain and verify
def verify_password(plain_password: str,hashed_password: str) -> bool:
    return pwd_context.verify(plain_password,hashed_password)


