from flask import Blueprint, request, jsonify
from models  import AIMessage
from database import db
from Chatbot.bot import generate_response

ai_bp = Blueprint("ai_routes", __name__)

# ──────────────────────────────────────────────────────────────────────────────
@ai_bp.route("/new_query", methods=["POST"])
def new_query():
    data = request.get_json(force=True)

    user_id      = data.get("user_id")
    session_id   = data.get("session_id")
    project_name = data.get("project_name", "Krupal Habitat")  # default
    user_msg     = (data.get("message") or "").strip()

    if not all([user_id, session_id, user_msg]):
        return jsonify(error="user_id, session_id, message required"), 400

    # save user message
    user_row = AIMessage(user_id=user_id, session_id=session_id,
                        role="user", message=user_msg)
    db.session.add(user_row)

    # pull last 19 previous msgs (so + current user = 20)
    history_rows = (
        AIMessage.query
        .filter_by(user_id=user_id, session_id=session_id)
        .order_by(AIMessage.timestamp.desc())
        .limit(19)
        .all()[::-1]
    )
    history = [{"role": r.role, "content": r.message} for r in history_rows]
    history.append({"role": "user", "content": user_msg})

    # LLM
    bot = generate_response(project_name, history)
    ai_row = AIMessage(user_id=user_id, session_id=session_id,
                    role="ai", message=bot["text"])
    db.session.add(ai_row)
    db.session.commit()
    print(f"AI Response: {bot['image_url']}")
    return jsonify(user=user_row.to_dict(), ai=ai_row.to_dict(),
                image_url=bot["image_url"]), 200

# ──────────────────────────────────────────────────────────────────────────────
@ai_bp.route("/get_messages/<string:user_id>/<string:session_id>", methods=["GET"])
def get_messages(user_id, session_id):
    rows = (AIMessage.query
            .filter_by(user_id=user_id, session_id=session_id)
            .order_by(AIMessage.timestamp)
            .all())
    return jsonify([r.to_dict() for r in rows]), 200
