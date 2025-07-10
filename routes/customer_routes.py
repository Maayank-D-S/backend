# routes/customer_routes.py
from flask import Blueprint, request, jsonify
from models import Customer
from database import db

from google.oauth2 import service_account
from googleapiclient.discovery import build

SPREADSHEET_ID = '1ZsHAcjTV2XbT8BB9VoY4TxEdx7NvmQWt3diUKZ7ym38'
RANGE_NAME = 'Sheet1!A1'  # Or A2 if A1 has headers
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
creds = service_account.Credentials.from_service_account_file(
    'credentials.json', scopes=SCOPES
)
sheets_service = build('sheets', 'v4', credentials=creds)

def append_to_google_sheet(customer):
    values = [[
        customer.name,
        customer.email,
        customer.phone or '',
        str(customer.project_id) if customer.project_id else ''
    ]]
    body = {'values': values}
    sheets_service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_NAME,
        valueInputOption='USER_ENTERED',
        body=body
    ).execute()

customer_bp = Blueprint("customer_routes", __name__)

@customer_bp.route("", methods=["POST"])  #  ←  NO leading slash
def add_customer():
    data = request.get_json(force=True)

    if not data.get("name") or not data.get("email"):
        return jsonify(error="Name and email are required"), 400

    if Customer.query.filter_by(email=data["email"]).first():
        return jsonify(error="Email already exists"), 400

    new_customer = Customer(
        name=data["name"],
        email=data["email"],
        phone=data.get("phone"),
        project_id=data.get("project_id")
    )
    db.session.add(new_customer)
    db.session.commit()

    try:
        append_to_google_sheet(new_customer)
    except Exception as e:
        print(f"[Google Sheets Error] {e}")  # Logging only, don't fail the request

    return jsonify(new_customer.to_dict()), 201
