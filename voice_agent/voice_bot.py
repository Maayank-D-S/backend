import asyncio
import logging
import os
import sys

# Inject virtualenv site-packages (optional)
venv_site_packages = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "venv", "Lib", "site-packages"))
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt,
)

# Local app imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import AIMessage
from app import app
from database import db
from livekit.plugins import deepgram
from Chatbot.bot import generate_response

# Load environment variables
load_dotenv()
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
print("Python executable used in Celery:", sys.executable)
print("DEEPGRAM_API_KEY =", deepgram_api_key)

logger = logging.getLogger("transcribe")
logging.basicConfig(level=logging.DEBUG)

user_id = os.getenv("USER_ID")
session_id = os.getenv("SESSION_ID")

def fetch_response(user_input, user_id, session_id):
    logger.debug(f"[fetch_response] User input: {user_input}")
    logger.debug(f"[fetch_response] User ID: {user_id}, Session ID: {session_id}")
    try:
        with app.app_context():
            user_row = AIMessage(user_id=user_id, session_id=session_id, role="user", message=user_input)
            db.session.add(user_row)

            history_rows = (
                AIMessage.query
                .filter_by(user_id=user_id, session_id=session_id)
                .order_by(AIMessage.timestamp.desc())
                .limit(19)
                .all()[::-1]
            )
            history = [{"role": r.role, "content": r.message} for r in history_rows]
            history.append({"role": "user", "content": user_input})

            result = generate_response("Krupal Habitat", history, True)

            ai_row = AIMessage(user_id=user_id, session_id=session_id, role="ai", message=result["text"])
            db.session.add(ai_row)
            db.session.commit()

            logger.debug(f"[fetch_response] Response from bot: {result}")
            return result["text"]
    except Exception:
        logger.exception("❌ Exception in fetch_response")
        return "Sorry, I couldn’t answer that."


async def entrypoint(ctx: JobContext):
    logger.info(f"🚀 Starting transcriber for room: {ctx.room.name}")

    participant_ready = asyncio.Event()

    # ─── Deepgram STT / TTS ──────────────────────────────────────────────
    stt_impl = deepgram.STT(model="nova-3", api_key=deepgram_api_key)
    tts = deepgram.TTS(
        model="aura-2-andromeda-en",
        encoding="linear16",
        sample_rate=24000,
        api_key=deepgram_api_key,
    )

    audio_src  = rtc.AudioSource(sample_rate=24000, num_channels=1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("bot-tts", audio_src)

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        logger.info(f"🎙️ Starting transcription for: {participant.identity}")
        audio_stream = rtc.AudioStream(track)
        stt_stream = stt_impl.stream()

        async def _handle_audio_stream():
            logger.debug("🎧 Listening to incoming audio frames...")
            async for ev in audio_stream:
                stt_stream.push_frame(ev.frame)

        async def _handle_transcription_output():
            logger.debug("🧠 Waiting for transcription results...")
            async for ev in stt_stream:
                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    user_query = ev.alternatives[0].text
                    logger.info(f"🙋 User query: {participant.identity}: {user_query}")
                    response_text = fetch_response(user_query, user_id, session_id)
                    logger.info(f"🤖 Bot response: {response_text}")

                    synth_stream = tts.synthesize(response_text)
                    logger.debug("🔊 Synthesizing response...")
                    async for chunk in synth_stream:
                        await audio_src.capture_frame(chunk.frame)

        await asyncio.gather(_handle_audio_stream(), _handle_transcription_output())

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        logger.info(f"✅ [track_subscribed] {participant.identity} | {track.kind}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            participant_ready.set()
            asyncio.create_task(transcribe_track(participant, track))

    @ctx.room.on("participant_joined")
    def on_participant_joined(p):
        logger.info(f"👤 participant_joined: {p.identity}")

    @ctx.room.on("track_published")
    def on_track_published(pub, p):
        logger.info(f"📡 track_published from {p.identity} ({pub.kind})")

    @ctx.room.on("*")
    def on_any(evt, *args, **kw):
        logger.debug(f"[EVENT] {evt} {args} {kw}")

    # ─── Connect & wait for audio ────────────────────────────────────────
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("🔌 Connected – waiting for first audio track...")
    await participant_ready.wait()
    logger.info("🎉 Remote audio track seen, publishing TTS track")

    await ctx.room.local_participant.publish_track(audio_track)
    logger.info("📢 Bot track published")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
