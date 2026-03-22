from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transcriber import stream_transcribe_tiktok

app = FastAPI()

class Request(BaseModel):
    url: str

@app.post("/transcribe")
async def transcribe_stream(req: Request):

    async def event_generator():
        for chunk in stream_transcribe_tiktok(req.url):
            yield {
                "event": "message",
                "data": chunk
            }

    return EventSourceResponse(event_generator())