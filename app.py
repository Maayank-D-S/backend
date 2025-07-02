# app.py
from flask import Flask
from flask_cors import CORS
from database import db
from routes.customer_routes import customer_bp
from routes.ai_message_route import ai_bp
from flask import request, jsonify
from livekit.api import AccessToken,VideoGrants
import os
import datetime
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///customers.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 💡 CORS for everything coming from the React dev server
from flask_cors import CORS

CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])

db.init_app(app)

# Blueprints
app.register_blueprint(customer_bp, url_prefix="/customers")
app.register_blueprint(ai_bp, url_prefix="/ai")

@app.route("/api/get-livekit-token", methods=["POST"])
def get_livekit_token():
    try:
        data = request.get_json()
        identity = data["identity"]
        room = data["room"]

        token = AccessToken(
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"]
        ).with_identity(identity).with_grants(
            VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True
            )
        )

        token.ttl = datetime.timedelta(minutes=30)

        return jsonify({"token": token.to_jwt()})

    except Exception as e:
        print("❌ Error generating token:", str(e))
        return jsonify({"error": "Token generation failed"}), 500

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    # use a fixed port so the frontend URL stays http://localhost:5000
    app.run(debug=True, port=5000)
