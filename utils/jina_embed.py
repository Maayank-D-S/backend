import os
import requests
from dotenv import load_dotenv

#used to create the embeddings using Jina embeddings.
load_dotenv() 

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_EMBEDDING_ENDPOINT = "https://api.jina.ai/v1/embeddings"

def get_embedding(text: str):
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY not found in environment variables.")
    #JINA API KEY is used for Authorization
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    #Text is stored in data
    data = {
        "input": text,
        "model": "jina-embeddings-v2-base-en"
    }

    try:
        #API is hit and the data is converted into embeddings
        response = requests.post(JINA_EMBEDDING_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("data")[0].get("embedding")
    except Exception as e:
        print("Error fetching embedding from Jina:", str(e))
        return None
