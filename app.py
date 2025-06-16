# app.py
from flask import Flask
from flask_cors import CORS
from database import db
from routes.customer_routes import customer_bp
from routes.ai_message_route import ai_bp

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///customers.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 💡 CORS for everything coming from the React dev server
CORS(
    app,
    origins="http://localhost:3000",
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

db.init_app(app)

# Blueprints
app.register_blueprint(customer_bp, url_prefix="/customers")
app.register_blueprint(ai_bp, url_prefix="/ai")

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    # use a fixed port so the frontend URL stays http://localhost:5000
    app.run(debug=True, port=5000)
