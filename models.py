from database import db

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    project_id = db.Column(db.String(100), nullable=True)  # ✅ NEW FIELD

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "project_id": self.project_id
        }

class AIMessage(db.Model):
    __tablename__ = "ai_message"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.String(36),  nullable=False)
    session_id = db.Column(db.String(36),  nullable=False, index=True)
    role       = db.Column(db.String(10),  nullable=False)       # "user" | "ai"
    message    = db.Column(db.Text,      nullable=False)
    timestamp  = db.Column(db.DateTime,  server_default=db.func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }