import subprocess
import uuid
import os
from faster_whisper import WhisperModel

MODEL_SIZE = "small"

model = WhisperModel(MODEL_SIZE, compute_type="int8")

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def transcribe_tiktok(url):
    uid = str(uuid.uuid4())
    video = f"/tmp/{uid}.mp4"
    audio = f"/tmp/{uid}.wav"

    # 1. Download video
    run(f"yt-dlp '{url}' -o {video}")

    # 2. Extract audio (16k mono)
    run(f"ffmpeg -i {video} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio}")

    # 3. Transcribe
    segments, info = model.transcribe(audio)

    text = " ".join([seg.text for seg in segments])

    # Cleanup
    os.remove(video)
    os.remove(audio)

    return {
        "text": text,
        "language": info.language
    }