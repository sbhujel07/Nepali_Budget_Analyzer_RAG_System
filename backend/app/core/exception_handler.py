from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

#handle the error we write on a  code like httpexecption-> user not found etc
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
        },
    )

#for validation handler like email 
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation Error",
            "errors": exc.errors(),
        },
    )

# handle the error which may occurs that we dont defined on our code like -> x = 10 / 0  yesto bug aayo aaba yesley app crash huna bata bachauxa ani internal server error dinxa
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal Server Error",
        },
    )