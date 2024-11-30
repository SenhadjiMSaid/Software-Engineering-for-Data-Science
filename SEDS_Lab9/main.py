import json
from typing import List
from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Header,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from enum import Enum

from fastapi.templating import Jinja2Templates
import pandas as pd

app = FastAPI()


@app.get("/")
async def hello_world():
    return {"hello": "world"}


@app.get("/users/{id}")
async def get_user(id: int = Path(..., ge=1)):
    return {"id": id}


class UserType(str, Enum):
    STANDARD = "standard"
    ADMIN = "admin"


@app.get("/users/{type}/{id}/")
async def get_user(type: UserType, id: int):
    return {"type": type, "id": id}


## 5. String Validation with Regex (Slide 13)
@app.get("/license-plates/{license}")
async def get_license_plate(license: str = Path(..., regex=r"^\d{5}-\d{3}-\d{2}$")):
    return {"license": license}


## 6. Query Parameters (Slide 14-15)
@app.get("/users")
async def get_user(page: int = Query(1, gt=0), size: int = Query(10, le=100)):
    return {"page": page, "size": size}


## 7. Body Parameters (Slide 16)
@app.post("/users")
async def create_user(name: str = Body(...), age: int = Body(...)):
    return {"name": name, "age": age}


## 8. Form Data (Slide 17)
@app.post("/createUser")
async def create_user(name: str = Form(...), age: int = Form(...)):
    return {"name": name, "age": age}


## 9. File Upload (Slide 18-20)
@app.post("/uploadfile")
async def upload_file(file: UploadFile = File(...)):
    return {"file_name": file.filename, "content_type": file.content_type}


@app.post("/uploadmultiplefiles")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    return [
        {"file_name": file.filename, "content_type": file.content_type}
        for file in files
    ]


## 10. Headers and Cookies (Slide 21-24)
@app.get("/getheader")
async def get_header(user_agent: str = Header(...)):
    return {"user_agent": user_agent}


@app.get("/setcookie")
async def set_cookie(response: Response):
    response.set_cookie("Name", "Said", max_age=86400)
    return {"hello": "world"}


## 11.HTTP Exceptions (Slide 25)
@app.post("/password")
async def check_password(password: str = Form(...), password_confirm: str = Form(...)):
    if password != password_confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Passwords don't match."
        )
    return {"message": "Passwords match."}


## 12.Building a custom HTMLresponse (Slide 26)
templates = Jinja2Templates(directory="templates")


@app.get("/reply")
async def reply(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


## Dataframes in Templates (Slides 27-28)


@app.get("/houseprices")
async def house_prices(request: Request):
    df = pd.read_csv("data/house_pricing.csv", nrows=25)
    data = json.loads(df.to_json(orient="records"))
    return templates.TemplateResponse(
        "houseprices.html", {"request": request, "house_prices": data}
    )
