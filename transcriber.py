import subprocess
import uuid
import os
from faster_whisper import WhisperModel

MODEL_SIZE = "small"
model = WhisperModel(MODEL_SIZE, compute_type="int8")

def stream_transcribe_tiktok(url):
    uid = str(uuid.uuid4())
    audio_path = f"/tmp/{uid}.wav"

    try:
        # 1. Download + convert langsung ke wav (NO MP4)
        cmd = f"""
        yt-dlp -f bestaudio -o - "{url}" |
        ffmpeg -i pipe:0 -f wav -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}
        """

        subprocess.run(cmd, shell=True, check=True)

        # 2. Transcribe dari file (STABLE)
        segments, info = model.transcribe(audio_path)

        for segment in segments:
            yield {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            }

        yield {
            "event": "done",
            "language": info.language
        }

    except Exception as e:
        yield {"error": str(e)}

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)