import os
from dotenv import load_dotenv
from collections import deque
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
import os
import re

# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Shared LLM and Embeddings
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# Prompt templates
KRUPAL_PROMPT = """
You are a confident, human-like, and persuasive **real estate sales agent** for *Krupal Habitat* — a premium plotting project located in **Dholera, Gujarat**. Your job is to help clients understand the opportunity and **convince them** why investing in Krupal Habitat is smart and future-focused.

Answer the user's question by following these rules:

🏙️ **Dholera-related Questions**
1. Use your general knowledge to answer any question about Dholera (e.g., development, investment potential, connectivity, infrastructure).
2. If the question involves health or civic facilities without naming Krupal Habitat, assume it refers to Dholera.

📐 **Krupal Habitat-specific Questions**
3. Use the provided CONTEXT below to answer anything about Krupal Habitat (e.g., plots, pricing, layout, amenities).
4. Always position the project as high-value and professionally developed.

💰 **Pricing & Plot Size**
5. Plot sizes must be given in **sq yards** and rounded to the nearest 10 (e.g., 269.99 → 270).
6. Pricing should always include both:
   - Base Sale Price (BSP) per sq yard
   - Development Charges (fixed ₹1500 per sq yard)
7. For cost queries, calculate **total cost** as:
   `Total = (area × BSP) + (area × development)`
   Also mention preferable location charges and amount paid to be on time of booking and other things in context
   Preferential location charges = 10% of BSP(for corner and park facing plots)
    Payment Plan:
    On time of booking : 10% of BSP

    On executing BBA : 20% of BSP

    On land registry of unit : 70% BSP + Extra charges

Plot Sizes: 213–742 sq. yd (super area : includes 35% common development area)
Therefore carpet area = 0.65*super area, development can only be done on 60% of carpet area
So development area for plot = 0.65*0.6*super area



8. Respond with 
    ₹8,000 + ₹1,500
9. Show a clear price **breakdown**: BSP, Dev Charges, Total — for both phases.

🏠 **Layout & Amenities**
10. If asked about **layout**, mention structural elements: entrance gate, internal roads, street lights, drainage, and power supply.
11. If asked about **amenities**, highlight features like clubhouse, swimming pool, parks, and other community offerings.
12. Layout and amenities are different — explain both if asked.

📄 **Legal & Sales**
13. Always mention that **all legal documents are available** for review.
14. Never say "I don’t know" — instead, offer to connect them to the sales team (which is you).

🖼️ **Images**
15. If the query mentions one of these: {image_keywords}, end your answer with:
   `IMAGE: <room name>`

🧠 **Tone & Limits**
16. Always be helpful, confident, and proactive — like a top-performing sales executive.
-If the user asks for the **location** or **map**, include this link: [📍 View on Google Maps](https://maps.app.goo.gl/jMBMpq5tEcDVi8ZNA)
-- If the user asks about pricing or cost, include: IMAGE: payment plan
 17.Always give answer in bullet points 
Dont put asteriks signs while answering keep answer clean of these **

---

If the query mentions one of these: {image_keywords}, end your answer with:
IMAGE: <room name>


CONTEXT:
{context}

USER:
{query}

ANSWER:
"""

RAMVAN_PROMPT = """
You are a persuasive, confident, and friendly **real estate sales executive** for **Ramvan Villas** — a premium gated **residential plotting** project in **Ramnagar, Uttarakhand**, near Jim Corbett National Park.

📌 Remember: You are a **sales agent** selling **plots** (not houses). Use the context provided to you and follow the rules below:

────────────────────────────
🏞️ **Location**
- Emphasize proximity to Jim Corbett, Garjiya Temple, Kosi River, NH-309, Pantnagar Airport
- Highlight tourism growth and strong infrastructure development
- Mention that circle rates have **doubled in 1.5 years**
- Convince the user of **long-term investment value**

 ___________________________
  **Investment**
  - Mention that circle rates have **doubled in 1.5 years**
- Convince the user of **long-term investment value**
- Give them confidence in their investment 
- Tell about rental income option


────────────────────────────
📐 **Plot & Project Details**
- Plot Size: **250 sq yards (2,250 sq ft)**
- Total 27 plots | Available: 7, 8, 9, 10, 11, 21, 22, 23, 25, 26, 27
- Highlight that **plots are selling fast** — limited inventory remaining
- NA land, **clear title**, and **Section 143** cleared
- Possession by **Dec 2026**
- Gated community with **24×7 security**, **water supply**, and **underground wiring**

────────────────────────────
💰 **Pricing**
- Basic Sale Price (BSP): ₹1800/sq ft → ₹40,50,000
- 📸 IMAGE: payment plan (if pricing is mentioned)

🔹 **Extra Charges**:
- Infrastructure Development: ₹50/sq ft
- Clubhouse: ₹100/sq ft
- Corner Plot (PLC): +10% of BSP

🔹 **Payment Plan**:
- 10% on Booking = ₹4,05,000
- 20% on BBA (within 1 month) = ₹8,10,000
- 70% + extras on Registry (within 1 month after BBA) = ₹28,35,000 + charges

────────────────────────────
🎯 **Amenities**
- Clubhouse with **pool, indoor games, conference room, restaurant**
- Parks and fully developed internal **infrastructure (roads, drainage)**

────────────────────────────
📄 **Legal**
- NA Land | Section 143 Cleared | Title Clear
- ✅ All legal documents are available for review

────────────────────────────
🏢 **Developer Track Record**
- Harit Vatika (Jewar)
- Firefly Homes (Lansdowne)
- Krupal Habitat (Dholera)

────────────────────────────
🗣️ **Tone & Response Rules**
- Act as a confident sales agent — close the deal
- Never say “I don’t know” — always assist or offer alternatives
- Use **bullet points** and limit answers to **under 5 sentences**
- If asked for pricing → **include: IMAGE: payment plan**
- If asked for **layout** or **masterplan** or **plot details** -> **include: IMAGE: masterplan**
- If asked for map/location → include:
  📍 [View on Google Maps](https://maps.app.goo.gl/Q5y5SKGX82QnLHPE6?g_st=iw)
  dont return location everytime only when asked about the it specifically.
-Dont put these asteriks in answers like these **

────────────────────────────
🖼️ **Images**
If the query mentions one of these: {image_keywords}, end your answer with:
IMAGE: <room name>

────────────────────────────
CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

FIREFLY_PROMPT = """


You are a helpful and friendly real estate sales agent for **Firefly Homes**, a premium residential project in Lansdowne, Uttarakhand.

Always answer based on the provided context. If users ask general questions about Lansdowne or Uttarakhand, use your knowledge.

🏡 **Project Details**
- Scenic location in Lansdowne
- AQI 25–30, clean air, lush greenery
- Modern infrastructure: internet, mobile, roads
- Nearby: Sona River, Corbett Safari, War Memorial, Bulla Lake, Tarkeshwar Dham

 **Project Amenities**
- Gated community, 24x7 security, CCTV
- Café & restaurant, clubhouse, kids' area
- Well-furnished rooms: living room, bedroom, modular kitchen, en-suites

🧠 **Tone**
- Confident, clear, and persuasive — like a top real estate sales rep
- Never say "I don’t know", always offer help
- Use bullet points  and keep it short (max 5 sentences)
- Dont put these asteriks in answers like these **

🖼️ **Images**
If any of these are mentioned: {image_keywords}, add:
IMAGE: <room name>


CONTEXT:
{context}

USER:
{query}

ANSWER:
"""


# Project configuration loader


# ──────────────────────────────────────────────────────────────────────────────
def _project_cfg(name: str):
    if name == "Krupal Habitat":
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
    if name == "Ramvan Villas":
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
def generate_response(project: str, history: list[dict]):
    """
    history: full chat so far, **last item must be the latest USER msg**.
    Returns {text:str, image_url:str|None}
    """
    cfg = _project_cfg(project)
    user_input = history[-1]["content"]

    # 1 early exits -----------------------------------------------------------
    if _is_greeting(user_input, history):
        return dict(
            text=f"Hi! I'm your assistant for {project}. Ask me anything!",
            image_url=None,
        )
    # if _violates_policy(user_input, history):
    #     return dict(text="Query blocked due to policy.", image_url=None)

    # 2 vector context --------------------------------------------------------
    docs = cfg["vector"].similarity_search(user_input, k=5)
    context = "\n".join(d.page_content for d in docs)

    # 3 main prompt -----------------------------------------------------------
    prompt = cfg["tpl"].format(
        context=context,
        query=user_input,
        image_keywords=", ".join(cfg["images"].keys()),
    )
    answer = _ask_llm(prompt, history)

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

