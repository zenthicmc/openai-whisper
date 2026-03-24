module.exports = {
  apps: [
    {
      name: "whisper-api",
      script: "venv/bin/python",
      args: "-m uvicorn app:app --host 0.0.0.0 --port 8000",
      cwd: "/var/www/openai-whisper",
      interpreter: "none",

      // 🚀 cluster mode
      exec_mode: "cluster",
      instances: 2, // jangan langsung max!

      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 5000,

      // 🔥 limit RAM biar ga jebol
      max_memory_restart: "3G",

      env: {
        WHISPER_MODEL_SIZE: "small",
        WHISPER_COMPUTE_TYPE: "int8",

        // ⚠️ turunin concurrency per instance
        MAX_CONCURRENT_TRANSCRIPTIONS: "2",
        MAX_QUEUE_SIZE: "50",

        DOWNLOAD_TIMEOUT_SECONDS: "120",
        LOG_LEVEL: "INFO",
      },
    },
  ],
};