from fastapi import FastAPI, Path
from enum import Enum

app = FastAPI()


@app.get("/")
async def hello_world():
    return {"hello": "World"}


class UserType(str, Enum):
    STANDARD = "standard"
    ADMIN = "admin"


@app.get("/users/{type}/{id}/")
async def get_user(type: UserType, id: int):
    return {"type": type, "id": id}


# Page 13
@app.get("/license-plates/{license}")
async def get_license_plate(license: str = Path(..., regex=r"^\d{5}-\d{3}-\d{2}$")):
    return {"license": license}


# @app.get("/users/{id}")
# async def get_user(id: int):
#     return {"id": id}
