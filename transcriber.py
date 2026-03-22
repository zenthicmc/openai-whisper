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
    
    Supports TikTok, YouTube, and any URL supported by yt-dlp.
    Uses async subprocess to avoid blocking the event loop.
    """
    if not request_id:
        request_id = str(uuid.uuid4())[:8]

    uid = str(uuid.uuid4())
    audio_path = f"/tmp/whisper_{uid}.wav"

    try:
        # --- 1. Download + convert to WAV via async subprocess ---
        # bestaudio/best: fallback to best video if audio-only not available (fixes TikTok)
        cmd = (
            f'yt-dlp -f "bestaudio/best" --no-warnings --no-progress -o - "{url}" | '
            f'ffmpeg -nostdin -y -i pipe:0 -vn -f wav -acodec pcm_s16le -ar 16000 -ac 1 '
            f'{audio_path} 2>/dev/null'
        )

        logger.info(f"[{request_id}] Downloading audio from: {url}")
        yield {"event": "status", "data": "downloading"}

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
            yield {"event": "error", "data": f"Download timed out after {DOWNLOAD_TIMEOUT} seconds"}
            return

        if process.returncode != 0 or not os.path.exists(audio_path):
            err_msg = stderr.decode(errors="ignore").strip() if stderr else "Unknown download error"
            logger.error(f"[{request_id}] Download failed: {err_msg}")
            yield {"event": "error", "data": f"Failed to download audio: {err_msg}"}
            return

        file_size = os.path.getsize(audio_path)
        if file_size < 1000:  # less than 1KB = probably empty/corrupt
            logger.error(f"[{request_id}] Audio file too small: {file_size} bytes")
            yield {"event": "error", "data": "Downloaded audio file is empty or corrupt"}
            return

        logger.info(f"[{request_id}] Audio downloaded: {file_size} bytes")

        # --- 2. Transcribe in thread pool (non-blocking) ---
        yield {"event": "status", "data": "transcribing"}

        model = await _get_model()
        results, info = await asyncio.to_thread(_transcribe_sync, audio_path)

        logger.info(f"[{request_id}] Transcription complete: {len(results)} segments, lang={info.language}")

        # --- 3. Stream results ---
        for segment in results:
            yield {
                "event": "transcript",
                "data": segment,
            }

        yield {
            "event": "done",
            "data": {
                "language": info.language,
                "language_probability": round(info.language_probability, 2),
                "total_segments": len(results),
            },
        }

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error")
        yield {"event": "error", "data": str(e)}

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.debug(f"[{request_id}] Cleaned up {audio_path}")