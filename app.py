import asyncio
import json
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse

from transcriber import stream_transcribe

# --- Logging ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Concurrency & queue control ---
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "5"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "20"))
semaphore = asyncio.Semaphore(MAX_CONCURRENT)
active_requests = 0
queued_requests = 0

# --- FastAPI app ---
app = FastAPI(
    title="Whisper Transcription API",
    description="Production-ready audio transcription API with SSE streaming and queue system",
    version="1.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request model ---
class TranscribeRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        if len(v) > 2048:
            raise ValueError("URL too long (max 2048 characters)")
        return v


# --- Endpoints ---
@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {
        "status": "ok",
        "active_requests": active_requests,
        "queued_requests": queued_requests,
        "max_concurrent": MAX_CONCURRENT,
        "max_queue_size": MAX_QUEUE_SIZE,
    }


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    """
    Transcribe audio from a URL via SSE stream.
    
    Supports TikTok, YouTube, and any URL supported by yt-dlp.
    Results are streamed as Server-Sent Events with the following event types:
    - queued: request is waiting in queue (includes position)
    - status: progress updates (downloading, transcribing)
    - transcript: individual transcribed segments with text, start, end times
    - done: final event with language info and total segments
    - error: error details if something goes wrong
    """
    global active_requests, queued_requests
    request_id = str(uuid.uuid4())[:8]

    async def event_generator():
        global active_requests, queued_requests

        # Reject if queue is also full (prevent unbounded growth)
        if queued_requests >= MAX_QUEUE_SIZE:
            logger.warning(f"[{request_id}] Rejected: queue full ({queued_requests}/{MAX_QUEUE_SIZE})")
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": f"Server overloaded. Queue full ({queued_requests}/{MAX_QUEUE_SIZE}). Try again later.",
                }),
            }
            return

        # If all slots are busy, enter queue
        is_queued = semaphore.locked()
        if is_queued:
            queued_requests += 1
            position = queued_requests
            logger.info(f"[{request_id}] Queued at position {position} for: {req.url}")
            yield {
                "event": "queued",
                "data": json.dumps({
                    "position": position,
                    "message": f"All {MAX_CONCURRENT} slots busy. You are #{position} in queue.",
                }),
            }

        try:
            # Semaphore will wait here until a slot opens
            async with semaphore:
                if is_queued:
                    queued_requests -= 1
                    logger.info(f"[{request_id}] Dequeued, starting processing")

                active_requests += 1
                logger.info(f"[{request_id}] Started transcription for: {req.url} (active: {active_requests}/{MAX_CONCURRENT}, queued: {queued_requests})")

                try:
                    async for chunk in stream_transcribe(req.url, request_id=request_id):
                        event_type = chunk.get("event", "message")
                        data = chunk.get("data", "")

                        yield {
                            "event": event_type,
                            "data": json.dumps(data) if isinstance(data, (dict, list)) else str(data),
                        }
                except Exception as e:
                    logger.exception(f"[{request_id}] Stream error")
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": str(e)}),
                    }
                finally:
                    active_requests -= 1
                    logger.info(f"[{request_id}] Finished (active: {active_requests}/{MAX_CONCURRENT}, queued: {queued_requests})")

        except Exception as e:
            if is_queued:
                queued_requests = max(0, queued_requests - 1)
            logger.exception(f"[{request_id}] Queue error")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}),
            }

    return EventSourceResponse(event_generator())


# --- Run directly ---
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False, workers=1)