import asyncio
import logging
import os
import sys

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import AIMessage
from app import app
from database import db
print("Python executable used in Celery:", sys.executable)

from livekit.plugins import deepgram
from Chatbot.bot import generate_response





# Load env
load_dotenv()
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
print("DEEPGRAM_API_KEY =", os.getenv("DEEPGRAM_API_KEY")) 
logger = logging.getLogger("transcribe")
logging.basicConfig(level=logging.DEBUG)
user_id = os.getenv("USER_ID")
session_id = os.getenv("SESSION_ID")
# room = os.getenv("ROOM")
# participant_identity = os.getenv("PARTICIPANT_IDENTITY")
# url = os.getenv("LIVEKIT_URL")
# api_key = os.getenv("LIVEKIT_API_KEY")
# api_secret = os.getenv("LIVEKIT_API_SECRET")
def fetch_response(user_input, user_id, session_id):
    logger.debug(f"[fetch_response] User input: {user_input}")
    logger.debug(f"[fetch_response] User ID: {user_id}, Session ID: {session_id}")
    try:
        with app.app_context():
            # 💾 Save the user message
            user_row = AIMessage(
                user_id=user_id,
                session_id=session_id,
                role="user",
                message=user_input
            )
            db.session.add(user_row)

            # 🧠 Get previous messages
            history_rows = (
                AIMessage.query
                .filter_by(user_id=user_id, session_id=session_id)
                .order_by(AIMessage.timestamp.desc())
                .limit(19)
                .all()[::-1]
            )
            history = [{"role": r.role, "content": r.message} for r in history_rows]
            history.append({"role": "user", "content": user_input})
            print("history is", history)
            logger.debug("[fetch_response] Calling generate_response")
            result = generate_response("Krupal Habitat", history, True)

            # 💾 Save the AI response
            ai_row = AIMessage(
                user_id=user_id,
                session_id=session_id,
                role="ai",
                message=result["text"]
            )
            db.session.add(ai_row)

            # ✅ Commit both entries
            db.session.commit()

            logger.debug(f"[fetch_response] Response from bot: {result}")
            return result["text"]
    except Exception as e:
        logger.exception("❌ Exception in fetch_response")
        return "Sorry, I couldn’t answer that."
    logger.debug(f"[fetch_response] User input: {user_input}")
    logger.debug(f"[fetch_response] User ID: {user_id}, Session ID: {session_id}")
    try:
        with app.app_context():
            history_rows = (
                AIMessage.query
                .filter_by(user_id=user_id, session_id=session_id)
                .order_by(AIMessage.timestamp.desc())
                .limit(19)
                .all()[::-1]
            )
            history = [{"role": r.role, "content": r.message} for r in history_rows]
            history.append({"role": "user", "content": user_input})
            print("history is",history)

        logger.debug("[fetch_response] Calling generate_response")
        result = generate_response("Krupal Habitat", history, True)
        logger.debug(f"[fetch_response] Response from bot: {result}")
        return result["text"]
    except Exception as e:
        logger.exception("❌ Exception in fetch_response")
        return "Sorry, I couldn’t answer that."


async def entrypoint(ctx: JobContext):
    logger.info(f" Starting transcriber for room: {ctx.room.name}")

    # Init Deepgram STT and TTS early
    stt_impl = deepgram.STT(model="nova-3", api_key=deepgram_api_key)
    tts = deepgram.TTS(
        model="aura-2-andromeda-en",
        encoding="linear16",
        sample_rate=24000,
        api_key=deepgram_api_key,
    )

    audio_src = rtc.AudioSource(sample_rate=24000, num_channels=1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("bot-tts", audio_src)

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        logger.info(f" Starting transcription for: {participant.identity}")
        audio_stream = rtc.AudioStream(track)
        stt_stream = stt_impl.stream()

        async def _handle_audio_stream():
            logger.debug(" Listening to incoming audio frames...")
            async for ev in audio_stream:
                # logger.debug(" Received audio frame")
                stt_stream.push_frame(ev.frame)

        async def _handle_transcription_output():
            logger.debug(" Waiting for transcription results...")
            async for ev in stt_stream:
                # logger.debug(f"[Deepgram Event] {ev}")
                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    user_query = ev.alternatives[0].text
                    logger.info(f" User query: {participant.identity}: {user_query}")

                    response_text = fetch_response(user_query, user_id, session_id)
                    logger.info(f" Bot response: {response_text}")

                    synth_stream = tts.synthesize(response_text)
                    logger.debug(" Synthesizing response...")
                    async for chunk in synth_stream:
                        logger.debug(" Sending TTS audio chunk")
                        await audio_src.capture_frame(chunk.frame)

        await asyncio.gather(
            _handle_audio_stream(),
            _handle_transcription_output(),
        )

    # ───────────── Event Handlers ─────────────
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(
            f"✅ [track_subscribed] Got track from {participant.identity} | Kind: {track.kind}"
        )
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(
                f" [track_subscribed] Subscribing to AUDIO from {participant.identity}"
            )
            asyncio.create_task(transcribe_track(participant, track))

    @ctx.room.on("participant_joined")
    def on_participant_joined(participant: rtc.RemoteParticipant):
        logger.info(f" [participant_joined] {participant.identity} joined the room")

    @ctx.room.on("track_published")
    def on_track_published(
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(
            f" [track_published] From {participant.identity}, kind: {publication.kind}"
        )

    @ctx.room.on("*")
    def on_any_event(event_name, *args, **kwargs):
        logger.debug(f" [EVENT] {event_name} | args={args}, kwargs={kwargs}")

    # Connect and publish TTS track after registering handlers
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(" Connected to room with AUDIO_ONLY subscription")
    logger.info(" Deepgram STT initialized")
    await ctx.room.local_participant.publish_track(audio_track)
    logger.info(" TTS Audio track published to room")
    logger.info(" Deepgram TTS initialized")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

