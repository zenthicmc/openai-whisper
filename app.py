from fastapi import FastAPI
from pydantic import BaseModel
from transcriber import transcribe_tiktok

app = FastAPI()

class Request(BaseModel):
    url: str

@app.post("/transcribe")
def transcribe(req: Request):
    result = transcribe_tiktok(req.url)
    return {
        "status": "success",
        "data": result
    }