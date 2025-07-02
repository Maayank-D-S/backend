import os
from celery_app import celery
import subprocess
from dotenv import load_dotenv

load_dotenv()
@celery.task
def run_voice_agent(room, identity, user_id, session_id):
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    print(url," ",api_key," ",api_secret)

    # Validate presence
    if not all([url, api_key, api_secret]):
        raise ValueError("Missing one or more required LiveKit environment variables.")
    cmd = [
        "python",
        "voice_agent/voice_bot.py",
        "connect",
        "--url", url,
        "--api-key", api_key,
        "--api-secret", api_secret,
        "--room", room,
        "--participant-identity", identity
    ]

    env = os.environ.copy()
    env.update({
        "USER_ID": str(user_id),
        "SESSION_ID": str(session_id)
    })

    subprocess.run(cmd, env=env)
