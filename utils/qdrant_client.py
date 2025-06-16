import os
import requests
from dotenv import load_dotenv

load_dotenv()
#used to save embeddings into qdrant and also for searching for top K matches.
QDRANT_URL = os.getenv("QDRANT_URL")  
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "news_articles")


#this function searches for top 5 matches related to a query
def query_qdrant(embedding: list, top_k: int = 5, collection_name: str =QDRANT_COLLECTION_NAME ):
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing Qdrant config in environment variables.")

    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }

    payload = {
        "vector": embedding,
        "top": top_k,
        "with_payload": True
    }

    url = f"{QDRANT_URL}/collections/{collection_name}/points/search"

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()["result"]

        return [{"id": r["id"], "score": r["score"], "text": r["payload"].get("text", "")} for r in results]

    except Exception as e:
        print("Error querying Qdrant:", str(e))
        return []
#function to upload embeddings to qdrant cloud
def upload_to_qdrant(documents: list):
    """
    Upload a list of documents to Qdrant collection.
    Each document is a dict with keys: 'id', 'embedding' (list of floats), 'text'.
    """

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Missing Qdrant config in environment variables.")

    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }

    points = []
    for doc in documents:
        points.append({
            "id": doc["id"],
            "vector": doc["embedding"],
            "payload": {
                "text": doc["text"],
                "title": doc.get("title", ""),
                "link": doc.get("link", ""),
                "published": doc.get("published", "")
            }
        })

    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION_NAME}/points"

    try:
        response = requests.put(url, headers=headers, json={"points": points})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print("Error uploading to Qdrant:", e)
        print("Response content:", response.content.decode())
        return None
    except Exception as e:
        print("Error uploading to Qdrant:", str(e))
        return None
