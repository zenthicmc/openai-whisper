module.exports = {
    apps: [
        {
            name: "whisper-api",
            script: "venv/bin/python",
            args: "-m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 10",
            cwd: "/var/www/openai-whisper",
            interpreter: "none",
            autorestart: true,
            watch: false,
            max_restarts: 10,
            restart_delay: 5000,
            env: {
                WHISPER_MODEL_SIZE: "small",
                WHISPER_COMPUTE_TYPE: "int8",
                MAX_CONCURRENT_TRANSCRIPTIONS: "50",
                MAX_QUEUE_SIZE: "100",
                DOWNLOAD_TIMEOUT_SECONDS: "120",
                LOG_LEVEL: "INFO",
            },
        },
    ],
};
