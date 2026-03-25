module.exports = {
  apps: [
    {
      name: "whisper-api",
      script: "venv/bin/python",
      args: "-m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 3",
      cwd: "/var/www/openai-whisper",
      interpreter: "none",

      exec_mode: "fork",
      instances: 1,

      autorestart: true,
      max_memory_restart: "4G",

      env: {
        WHISPER_MODEL_SIZE: "small",
        WHISPER_COMPUTE_TYPE: "int8",

        // 🔥 ini yang penting
        MAX_CONCURRENT_TRANSCRIPTIONS: "3",
        MAX_QUEUE_SIZE: "50",

        DOWNLOAD_TIMEOUT_SECONDS: "120",
        LOG_LEVEL: "INFO",

        YTDLP_COOKIES: "/var/www/openai-whisper/cookies.txt"
      },
    },
  ],
};