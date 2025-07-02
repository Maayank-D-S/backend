import os
from dotenv import load_dotenv
from collections import deque
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
import os
import re

#  If the query mentions one of these: {image_keywords}, end your answer with:
#  IMAGE: <room name>
# this for image return will be added later
# -- If the user asks about pricing or cost, include: IMAGE: payment plan
#  If asked for pricing → **include: IMAGE: payment plan**
# - If asked for **layout** or **masterplan** or **plot details** -> **include: IMAGE: masterplan**


# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Shared LLM and Embeddings
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# Prompt templates
KRUPAL_PROMPT = """
You are a helpful and persuasive real estate sales agent for Krupal Habitat in Dholera, Gujarat — a premium plotting project in a high-growth smart city zone.

Use the context provided to answer the user query in under 5 bullet points. Follow these rules:
- Use a confident and human tone — like a friendly sales agent.
- Never say "I don't know". Offer help or a next step.
- If the question is about Dholera (development, infrastructure, city growth, etc.), use general knowledge confidently.
- For questions about hospitals, roads, or civic services without naming Krupal Habitat, assume the query is about Dholera.
- If asked for map or location → include:
  📍 [View on Google Maps](https://maps.app.goo.gl/jMBMpq5tEcDVi8ZNA)
- Don’t include the location link unless specifically asked.

🏡 For Krupal Habitat-specific queries:
- Total project size: 22,000 sq. yards
- Plot sizes: 213–742 sq. yards (mention only range if asked)
- BSP: ₹8,000/sq yd, Dev Charges: ₹1,500/sq yd
- PLC (for park-facing/corner plots): 10% of BSP
- Payment Plan:
  • 10% of BSP on booking  
  • 20% of BSP on BBA  
  • 70% of BSP + other charges on registry
- Super area includes 35% common dev space  
  • Carpet area = 0.65 × super area  
  • Buildable area = 60% of carpet area

💬 If asked about cost:
- Show a clear price breakdown: BSP, Dev Charges, Total  
- Mention PLC only if relevant

🛠 If asked about layout:
- Mention: entrance gate, internal roads, street lights, drainage, power

🌿 If asked about amenities:
- Mention: clubhouse, swimming pool, landscaped parks, community spaces

📄 confirm:
- All legal documents are available for review



Avoid long explanations. Be short, clear, and convincing.

CONTEXT:
{context}

USER:
{query}

ANSWER:
- Limit your reply to 4–5 bullet points max.
- before starting points just give **one line** at the top like here are the main points or some other line that seems suitable.
- Use natural language like a real person speaking.
- Be short, clear, and persuasive — avoid repeating details.
- Speak like a friendly human sales executive — not like a robot.
"""

RAMVAN_PROMPT = """
You are a helpful and persuasive real estate sales agent for Ramvan Villas in Uttarakhand — a premium gated plotting project near Jim Corbett National Park.

Use the context provided to answer the user query in under 5 bullet points. Follow these rules:
- Use a confident and human tone — like a friendly sales agent.
- Never say "I don't know". Offer help or a next step.
-  If asked for map/location → include:
  📍 [View on Google Maps](https://maps.app.goo.gl/Q5y5SKGX82QnLHPE6?g_st=iw)
  dont return location everytime only when asked about the it specifically.

Avoid long explanations. Be short, clear, and convincing.

CONTEXT:
{context}

USER:
{query}

ANSWER:
- Limit your reply to 4–5 bullet points max.
- before starting points just give **one line** at the top like here are the main points or some other line that seems suitable.
- Use natural language like a real person speaking.
- Be short, clear, and persuasive — avoid repeating details.
- Speak like a friendly human sales executive — not like a robot.
"""


FIREFLY_PROMPT = """

You are a helpful and persuasive real estate sales agent for Firefly Homes in Uttarakhand.

Use the context provided to answer the user query in under 5 bullet points. Follow these rules:
- Use a confident and human tone — like a friendly sales agent.
- Never say "I don't know". Offer help or a next step.


Avoid long explanations. Be short, clear, and convincing.

CONTEXT:
{context}

USER:
{query}

ANSWER:
- Limit your reply to 4–5 bullet points max.
- before starting points just give **one line** at the top like here are the main points or some other line that seems suitable.
- Use natural language like a real person speaking.
- Be short, clear, and persuasive — avoid repeating details.
- Speak like a friendly human sales executive — not like a robot.


"""

LEGAL_PROMPT = """
You are a helpful real estate consultant for our customer.Clear thier doubts regarding property purchase questions and any legal queiries.
Be very **breif**.Make sure the process is clear to them and in the end they feel confident 

CONTEXT:
{context}

USER:
{query}

ANSWER:
- Limit your reply to 4–5 bullet points max.
- before starting points just give **one line** at the top like here are the main points or some other line that seems suitable.
- Use natural language like a real person speaking.
- Be short, clear, and persuasive — avoid repeating details.
- Speak like a friendly human real estate consultant— not like a robot.

"""
# Project configuration loader


# ──────────────────────────────────────────────────────────────────────────────
def _project_cfg(name: str):
    print("Testing 3")
    if name == "Krupal Habitat":
        print("Testing 8")
        print("🟡 Trying to load FAISS index")
        try:
            vec = FAISS.load_local(
                os.path.join(BASE_DIR, "krupaldb_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            )
            print("✅ FAISS loaded")
        except Exception as e:
            print(f"❌ Failed to load FAISS: {e}")
            raise
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "krupaldb_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            images={
                "gated community": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749903023/gatedcommunity_gpjff4.jpg",
                "house": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749903025/house_cu9on6.jpg",
                "clubhouse": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749903022/clubhouse_opxfdz.jpg",
                "krupal habitat": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749903024/krupalhabitat_ywpcpp.jpg",
                "payment plan": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749922165/krupal_payment_soj4mc.jpg",
            },
            tpl=KRUPAL_PROMPT,
        )
    print("Testing 4")
    if name == "Ramvan Villas":
        print("Testing 9")
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "ramvan_villas_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            images={
                "bedroom": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749903320/bedroom_rnp54b.jpg",
                "living room": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749903327/livingroom_xdpba4.jpg",
                "dining room": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749903321/diningroom_xezi1c.jpg",
                "villa": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749903321/house_rceotg.jpg",
                "kitchen": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749903321/diningroom_xezi1c.jpg",
                "payment plan": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749922062/ramvan_payment_ychisk.jpg",
                "masterplan": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1750164243/Layout_qwbaun.jpg",
            },
            tpl=RAMVAN_PROMPT,
        )
    print("Testing 4")
    if name == "Firefly Homes":
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "firefly_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            images={
                "clubhouse": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1749902620/clubhouse_og4dc2.jpg"
            },
            tpl=FIREFLY_PROMPT,
        )
    print("Testing 4")
    if name == "Legal Consultant":
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "legal_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            images={},  # likely not needed unless you want legal diagrams or infographics
            tpl=LEGAL_PROMPT,
        )
    print("Unknown Project")
    raise ValueError("Unknown project")


# ──────────────────────────────────────────────────────────────────────────────
# tiny helper for LLM calls with explicit history
def _ask_llm(prompt: str, history: list[dict]):
    messages = []
    for h in history:
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        else:
            messages.append(AIMessage(content=h["content"]))
    messages.append(HumanMessage(content=prompt))
    print("Sending messages to LLM:")
    for msg in messages:
        print(">", msg)
    return llm.invoke(messages).content.strip()


# wrapper filters -------------------------------------------------------------
def _violates_policy(text: str, history):
    pol_prompt = f"""You are a content-filter. Reply ONLY "BLOCK" or "ALLOW".
Text: "{text}" """
    return _ask_llm(pol_prompt, history).upper() == "BLOCK"


def _is_greeting(text: str, history):
    g_prompt = (
        f"""Reply "GREETING" if "{text}" is just a greeting/ vague, else "QUERY":"""
    )
    return _ask_llm(g_prompt, history).upper() == "GREETING"


# ──────────────────────────────────────────────────────────────────────────────
def generate_response(project: str, history: list[dict],voice_mode: bool):
    """
    history: full chat so far, **last item must be the latest USER msg**.
    Returns {text:str, image_url:str|None}
    """
    print("Testing")
    cfg = _project_cfg(project)
    print("Testing: after cfg")
    user_input = history[-1]["content"]

    # 1 early exits -----------------------------------------------------------
    if _is_greeting(user_input, history):
        print("Yo")
        return dict(
            text=f"Hi! I'm your assistant for {project}. Ask me anything!",
            image_url=None,
        )
    # if _violates_policy(user_input, history):
    #     return dict(text="Query blocked due to policy.", image_url=None)

    # 2 vector context --------------------------------------------------------
    docs = cfg["vector"].similarity_search(user_input, k=5)
    context = "\n".join(d.page_content for d in docs)
    VOICE_PROMPT_TEMPLATE=""
    if(voice_mode):
        VOICE_PROMPT_TEMPLATE = """

You are speaking aloud to a human in voice mode.

⭑ Tone & Emotion
- Match the user's sentiment: friendly if excited, calm if unsure, concise if rushed.
- Use a human, conversational tone — not robotic or overly formal.

⭑ Style & Delivery
- Expand abbreviations: say "square yard", not "sq. yd".
- Speak full numbers: say "twenty thousand", not "20,000".
- Use punctuation for natural pauses.

⭑ Answering Strategy
- Always respond in **1 natural paragraph**, not bullets.
- Summarize everything in **around 40 words only**.
- Do not explain every detail — highlight the most important points.
- Always end with a **follow-up question** to keep the conversation going.
- If the question is simple or factual (e.g., distance, direction, yes/no), answer it **briefly** — ideally 1 sentence.
- If the question asks for full project details or comparisons, summarize it in **under 40 words**, in 1 short paragraph.
- Always ask a relevant follow-up question to continue the conversation.
"""
    

    # 3 main prompt -----------------------------------------------------------
    prompt =(
        "Analyze the user's emotional tone and respond accordingly.\n\n"
        +cfg["tpl"].format(
        context=context,
        query=user_input,
        image_keywords="",  # ", ".join(cfg["images"].keys()),
        )+VOICE_PROMPT_TEMPLATE
    )
    try:
        answer = _ask_llm(prompt, history)
    except Exception as e:
        print(f"[ERROR] generate_response failed during LLM call: {e}")
        raise
    print(f"[DEBUG] Generated answer: {answer}")

    # 4 policy check on answer ------------------------------------------------
    # if _violates_policy(answer, history):
    #     return dict(text="Response blocked due to policy.", image_url=None)

    # 5 optional image tag parsing -------------------------------------------
    # img_url = None
    # match = re.search(r"image:\s*(\w[\w\s]*)", answer, re.IGNORECASE)
    # if match:
    #     keyword = match.group(1).strip().lower()
    #     img_url = cfg["images"].get(keyword)
    #     if img_url:
    #         answer = re.sub(
    #             r"image:\s*[\w\s]*", "", answer, flags=re.IGNORECASE
    #         ).strip()
    #     else:
    #         print(f"[WARN] No image found for keyword: '{keyword}'")

    return dict(text=answer, image_url=None)
