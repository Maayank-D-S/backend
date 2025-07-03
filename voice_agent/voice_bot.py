"""
voice_agent/voice_bot.py
Rev: 2025-07-03 – adds retry-on-participant-missing logic
"""

import asyncio
import logging
import os
import sys
import time

# ──────────────────────────────────────────────
# Optional: make sure the project-local venv’s site-packages are importable
venv_site_packages = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "venv", "Lib", "site-packages")
)
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# ─── Third-party imports ───────────────────────
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, stt
from livekit.api.twirp_client import TwirpError  # for retry detection

# ─── Local project imports ─────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import AIMessage
from app import app
from database import db
from livekit.plugins import deepgram
from Chatbot.bot import generate_response

# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────
load_dotenv()
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

logger = logging.getLogger("voice_bot")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

logger.debug("Python executable: %s", sys.executable)
logger.debug("DEEPGRAM_API_KEY starts with: %s", deepgram_api_key[:6] if deepgram_api_key else "None")

user_id    = os.getenv("USER_ID")
session_id = os.getenv("SESSION_ID")

# ──────────────────────────────────────────────
# Helper: call LangChain bot + persist chat
# ──────────────────────────────────────────────
def fetch_response(user_input: str, user_id: str, session_id: str) -> str:
    logger.debug("[fetch_response] User: %s | Session: %s | Text: %s", user_id, session_id, user_input)
    try:
        with app.app_context():
            # persist user message
            db.session.add(AIMessage(user_id=user_id, session_id=session_id,
                                     role="user", message=user_input))

            # last 19 previous → history (this + 1 = 20)
            history_rows = (
                AIMessage.query.filter_by(user_id=user_id, session_id=session_id)
                .order_by(AIMessage.timestamp.desc())
                .limit(19)
                .all()[::-1]
            )
            history = [{"role": r.role, "content": r.message} for r in history_rows]
            history.append({"role": "user", "content": user_input})

            result = generate_response("Krupal Habitat", history, True)

            # persist AI answer
            db.session.add(AIMessage(user_id=user_id, session_id=session_id,
                                     role="ai", message=result["text"]))
            db.session.commit()
            logger.debug("[fetch_response] Bot reply: %s", result)
            return result["text"]

    except Exception:
        logger.exception("❌ Exception inside fetch_response")
        return "Sorry, I couldn't answer that."

# ──────────────────────────────────────────────
# LiveKit Agent entrypoint
# ──────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    logger.info("🚀 Voice agent joined room: %s", ctx.room.name)

    participant_ready = asyncio.Event()

    # Deepgram STT / TTS
    stt_impl = deepgram.STT(model="nova-3", api_key=deepgram_api_key)
    tts_impl = deepgram.TTS(
        model="aura-2-andromeda-en",
        encoding="linear16",
        sample_rate=24000,
        api_key=deepgram_api_key,
    )

    audio_src   = rtc.AudioSource(sample_rate=24000, num_channels=1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("bot-tts", audio_src)

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        logger.info("🎙️ Transcribing %s", participant.identity)
        audio_stream = rtc.AudioStream(track)
        stt_stream   = stt_impl.stream()

        async def pump_audio():
            async for ev in audio_stream:
                stt_stream.push_frame(ev.frame)

        async def handle_transcripts():
            async for ev in stt_stream:
                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    user_text = ev.alternatives[0].text
                    logger.info("🙋 %s said: %s", participant.identity, user_text)

                    bot_text = fetch_response(user_text, user_id, session_id)
                    logger.info("🤖 Bot: %s", bot_text)

                    synth_stream = tts_impl.synthesize(bot_text)
                    async for chunk in synth_stream:
                        await audio_src.capture_frame(chunk.frame)

        await asyncio.gather(pump_audio(), handle_transcripts())

    # LiveKit event hooks
    @ctx.room.on("track_subscribed")
    def _(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            participant_ready.set()
            asyncio.create_task(transcribe_track(participant, track))

    # connect & wait
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("🔌 Connected, awaiting audio...")
    await participant_ready.wait()

    await ctx.room.local_participant.publish_track(audio_track)
    logger.info("📢 TTS track published – ready to chat.")

# ──────────────────────────────────────────────
# Robust startup wrapper
# ──────────────────────────────────────────────
_RETRIES      = 10
_RETRY_DELAY  = 1.0  # seconds

def run_worker_with_retry():
    """
    LiveKit-agents fails fast when the participant is not yet in the room.
    We retry a few times so the browser can finish its join+publish handshake.
    """
    for attempt in range(1, _RETRIES + 1):
        try:
            cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
            return  # normal exit
        except TwirpError as err:
            if getattr(err, "code", "") == "not_found":
                logger.warning("[startup] Participant not found (attempt %d/%d) – retrying in %.1fs", attempt, _RETRIES, _RETRY_DELAY)
                time.sleep(_RETRY_DELAY)
                continue
            raise  # other LiveKit errors → bubble up
        except Exception:
            logger.exception("Unhandled exception during worker startup")
            raise
    logger.error("Aborting – participant never appeared after %d retries.", _RETRIES)

# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_worker_with_retry()
