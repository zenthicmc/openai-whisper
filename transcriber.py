import subprocess
from faster_whisper import WhisperModel

MODEL_SIZE = "small"
model = WhisperModel(MODEL_SIZE, compute_type="int8")

def stream_transcribe_tiktok(url):

    ytdlp_cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-o", "-",
        "--no-playlist",
        "--quiet",
        url
    ]

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "pipe:0",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "pipe:1"
    ]

    ytdlp = subprocess.Popen(
        ytdlp_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdin=ytdlp.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    segments, info = model.transcribe(ffmpeg.stdout)

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