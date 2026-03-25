import asyncio
import uuid
import os
import logging
from typing import AsyncGenerator, Dict, Any

logger = logging.getLogger(__name__)

# --- Configuration via environment variables ---
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "120"))
COOKIES_PATH = os.getenv("YTDLP_COOKIES")

# --- Lazy-loaded Whisper model (singleton) ---
_model = None
_model_lock = asyncio.Lock()


async def _get_model():
    """Load the Whisper model once, thread-safe via asyncio.Lock."""
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:  # double-check after acquiring lock
                logger.info(f"Loading Whisper model: size={MODEL_SIZE}, compute_type={COMPUTE_TYPE}")
                from faster_whisper import WhisperModel
                _model = await asyncio.to_thread(
                    WhisperModel, MODEL_SIZE, compute_type=COMPUTE_TYPE
                )
                logger.info("Whisper model loaded successfully")
    return _model


def _transcribe_sync(audio_path: str):
    """Run transcription synchronously (called inside thread pool)."""
    from faster_whisper import WhisperModel
    # Access the already-loaded global model
    segments, info = _model.transcribe(audio_path)
    results = []
    for segment in segments:
        results.append({
            "text": segment.text.strip(),
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
        })
    return results, info


async def stream_transcribe(url: str, request_id: str = None) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Download audio from URL, transcribe with Whisper, and yield results as async generator.

    SSE event types (all data is JSON):
    - status:     { step, total_steps, stage, message, progress_pct, request_id }
    - transcript: { index, total, text, start, end }
    - done:       { language, language_probability, total_segments, duration, full_text, request_id }
    - error:      { code, message, request_id }
    """
    import time
    import re

    if not request_id:
        request_id = str(uuid.uuid4())[:8]

    # Strip tracking parameters dari URL YouTube (?si=, &si=)
    url = re.sub(r'[?&]si=[^&]+', '', url).rstrip('?&')

    uid = str(uuid.uuid4())
    audio_path = f"/tmp/whisper_{uid}.wav"
    start_time = time.time()

    # Gunakan path absolut yt-dlp dari venv agar berjalan benar di PM2
    YTDLP_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv/bin/yt-dlp")
    if not os.path.exists(YTDLP_BIN):
        YTDLP_BIN = "yt-dlp"  # fallback ke system yt-dlp

    def _status(step: int, stage: str, message: str, progress_pct: int = 0):
        return {
            "event": "status",
            "data": {
                "step": step,
                "total_steps": 5,
                "stage": stage,
                "message": message,
                "progress_pct": progress_pct,
                "request_id": request_id,
            },
        }

    def _error(code: str, message: str):
        return {
            "event": "error",
            "data": {
                "code": code,
                "message": message,
                "request_id": request_id,
            },
        }

    try:
        # --- Step 1: Connecting ---
        yield _status(1, "connecting", "Connecting to video source...", 5)

        # --- Step 2: Download + convert to WAV ---
        cookies_part = f'--cookies "{COOKIES_PATH}"' if COOKIES_PATH else ""

        cmd = (
            f'"{YTDLP_BIN}" {cookies_part} '
            f'--no-js-runtimes --js-runtimes node '
            f'--extractor-args "youtube:player_client=web" '
            f'-f "140-1/251-1/bestaudio/best" '
            f'--no-warnings --no-progress -o - "{url}" | '
            f'ffmpeg -nostdin -y -i pipe:0 -vn -f wav -acodec pcm_s16le -ar 16000 -ac 1 '
            f'{audio_path} 2>/dev/null'
        )

        logger.info(f"[{request_id}] Downloading audio from: {url}")
        yield _status(2, "downloading", "Downloading and extracting audio...", 15)

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=DOWNLOAD_TIMEOUT
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.error(f"[{request_id}] Download timed out after {DOWNLOAD_TIMEOUT}s")
            yield _error("DOWNLOAD_TIMEOUT", f"Download timed out after {DOWNLOAD_TIMEOUT} seconds")
            return

        if process.returncode != 0 or not os.path.exists(audio_path):
            err_msg = stderr.decode(errors="ignore").strip() if stderr else "Unknown error"
            logger.error(f"[{request_id}] Download failed: {err_msg}")
            yield _error("DOWNLOAD_FAILED", f"Failed to download audio: {err_msg}")
            return

        file_size = os.path.getsize(audio_path)
        if file_size < 1000:
            logger.error(f"[{request_id}] Audio file too small: {file_size} bytes")
            yield _error("EMPTY_AUDIO", "Downloaded audio file is empty or corrupt")
            return

        logger.info(f"[{request_id}] Audio downloaded: {file_size} bytes")

        # --- Step 3: Converting / validating audio ---
        file_size_mb = round(file_size / (1024 * 1024), 2)
        yield _status(3, "processing", f"Audio ready ({file_size_mb} MB). Preparing for transcription...", 40)

        # --- Step 4: Loading model (if first time) ---
        yield _status(4, "loading_model", "Loading AI transcription model...", 50)
        model = await _get_model()
        yield _status(4, "transcribing", "Transcribing audio with Whisper AI...", 60)

        results, info = await asyncio.to_thread(_transcribe_sync, audio_path)
        total_segments = len(results)

        logger.info(f"[{request_id}] Transcription complete: {total_segments} segments, lang={info.language}")
        yield _status(5, "streaming", f"Transcription complete. Streaming {total_segments} segments...", 85)

        # --- Step 5: Stream individual segments ---
        full_text_parts = []
        for i, segment in enumerate(results):
            full_text_parts.append(segment["text"])
            yield {
                "event": "transcript",
                "data": {
                    "index": i + 1,
                    "total": total_segments,
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                },
            }

        # --- Final: done with full text ---
        elapsed = round(time.time() - start_time, 2)
        full_text = " ".join(full_text_parts).strip()
        duration = round(results[-1]["end"], 2) if results else 0

        yield {
            "event": "done",
            "data": {
                "request_id": request_id,
                "language": info.language,
                "language_probability": round(info.language_probability, 2),
                "total_segments": total_segments,
                "duration": duration,
                "processing_time": elapsed,
                "full_text": full_text,
            },
        }

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error")
        yield _error("INTERNAL_ERROR", str(e))

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.debug(f"[{request_id}] Cleaned up {audio_path}")