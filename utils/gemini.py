import os
from dotenv import load_dotenv
import google.generativeai as genai

#Use LLM to generate response
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")


genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash") 

def query_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Error querying Gemini:", str(e))
        return "Sorry, I couldn't process your request."
