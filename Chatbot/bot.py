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
# If the query mentions one of these: {image_keywords} and wherever image is applicable end your answer with:
# IMAGE: <room name>


# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Shared LLM and Embeddings
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2, openai_api_key=OPENAI_API_KEY)
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
  📍 [View on Google Maps](https://maps.app.goo.gl/yna6fJTuMJsDfoS19?g_st=ipc)
- Don’t include the location link unless specifically asked.


📄 confirm:
- All legal documents are available for review
project is not rera approved dont mention it.

-if they ask for contact number give this number- +91 9311474661





Avoid long explanations. Be short, clear, and convincing.

CONTEXT:
{context}

USER:
{query}

ANSWER:
- Answer in  minimum number of points
- before starting points just give **one line** at the top like here are the main points or some other line that seems suitable.
- Use natural language like a real person speaking.
-for project legalaties and very specific details strictly use only the context.

- Be short, clear, and persuasive — avoid repeating details.
- Speak like a friendly human conversational sales executive — not like a robot.
"""

RAMVAN_PROMPT = """
You are a helpful and persuasive real estate sales agent for Ramvan Villas in Uttarakhand — a premium gated plotting project near Jim Corbett National Park.

 Follow these rules:
- Use a confident and human tone — like a friendly sales agent.
- Never say "I don't know". Offer help or a next step.
-  If asked for map/location → include:
  📍 [View on Google Maps](https://maps.app.goo.gl/Q5y5SKGX82QnLHPE6?g_st=iw)
  dont return location everytime only when asked about the it specifically.

Avoid long explanations. Be short, clear, and convincing.
-if they ask for contact number give this number- +91 9971659153


CONTEXT:
{context}

USER:
{query}

ANSWER:
- Answer in minimum number of points
- before starting points just give **one line** at the top like here are the main points or some other line that seems suitable.
- Use natural language like a real person speaking.
-for project legalaties and very specific details strictly use only the context.
- Be short, clear, and persuasive — avoid repeating details.
- Speak like a friendly human converdational sales executive — not like a robot.
"""


FIREFLY_PROMPT = """

You are a helpful and persuasive real estate sales agent for Firefly Homes in Uttarakhand.

Use the context provided to answer the user query in under 5 bullet points. Follow these rules:
- Use a confident and human tone — like a friendly sales agent.
- Never say "I don't know". Offer help or a next step.
-if they ask for contact number give this number- +91 9311474661


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
REAL_ESTATE_PROMPT = """

You are a helpful, confident, and persuasive real estate sales agent for luxury and lifestyle properties.

Your job is to assist potential buyers by providing accurate, relevant answers strictly based on the given context. Follow these rules:

- Use a natural, human tone — like a friendly sales executive.
- NEVER hallucinate — if something is not in the context, offer to provide it later or suggest contacting the team.
- Be short, clear, and convincing — not robotic or overly formal.
- If the user asks for a contact number, give this number: +91 9311474661
- Mention the project name naturally if it's present in the context.
- Never say “I don’t know” — instead offer a next step or suggest reaching out.

FORMAT:

CONTEXT:
{context}

USER:
{query}

ANSWER:
- Start with a short, friendly one-liner like “Here’s what you need to know” or “Let me break it down for you”.
- Then give 4–5 concise bullet points only.
- Use persuasive but realistic language.
- Always respond **only using details from the context**.
- Do NOT make up prices, amenities, locations, or builder names that are not in the context.
- Avoid long paragraphs or repetition.

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
                os.path.join(BASE_DIR, "krupalfinal_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            )
            print("✅ FAISS loaded")
        except Exception as e:
            print(f"❌ Failed to load FAISS: {e}")
            raise
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "krupalfinal_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            images={
                "amenities": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570131/amenities_yxwsos.png",
                "clubhouse1": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570130/clubhouse1_ftj3be.png",
                "clubhouse2": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570132/clubhouse2_c9rpm1.png",
                "entrance gate": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570129/entrance_gate_fqvj9g.png",
                "garden": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570128/garden_bbapb0.png",
                "gym1": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570129/gym1_u80pin.png",
                "gym2": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570127/gym2_zy4wkk.png",
                "gym3": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570125/gym3_y0bgss.png",
                "plot size": "",
                "layout1": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570124/Layout_1_ls7niu.png",
                "layout2": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570131/Layout_2_rw6gs2.png",
                "location map": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570123/location_map_svrwg3.png",
                "masterplan page": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570122/masterplan_page_xolaj0.png",
                "site office": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570123/Site_office_wy2nqq.png",
                "surrounding developments": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570123/surrounding_developments_h5rolw.png",
                "theatre": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752570130/theatre_u6nmdz.png",
            },
            tpl=KRUPAL_PROMPT,
        )
    print("Testing 4")
    if name == "Ramvan Villas":
        print("Testing 9")
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "ramvan_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            images={
                "amenities": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577372/amenities_bwjrvi.png",
                "bathroom": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577372/Bathroom_akkylf.png",
                "bedroom": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577371/Bedroom_vqcje1.png",
                "kitchen": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577369/Kitchen_lsacjy.png",
                "layout": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577374/layout_vqoqsm.png",
                "living room": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577367/Living_room_edz5pe.png",
                "nearby tourist attractions": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577365/nearby_tourist_attractions_gqhhyk.png",
                "payment plan": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577366/payment_plan_ptx03d.png",
                "progress1": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577364/progress1_xqo54b.png",
                "progress2": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577370/progress2_hyt2hr.png",
                "progress3": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577361/progress3_vsk3xz.jpg",
                "ramvan map and nearby cities": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577361/ramvan_map_and_nearby_cities_hcz8hd.png",
                "sample villa1": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577361/sample_villa1_t5r1xm.jpg",
                "sample villa2": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577358/sample_villa2_dhex89.jpg",
                "sample villa3": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577358/sample_villa3_c9y6po.jpg",
                "sample villa4": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577357/sample_villa4_k80fvq.jpg",
                "sample villa5": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577356/sample_villa5_u49opi.jpg",
                "sample villa6": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577930/sample_villa6_rd8upo.jpg",
                "sample villa7": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577927/sample_villa7_ux1u1m.jpg",
                "sample villa8": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577926/sample_villa8_lgvxwv.jpg",
                "sample villa9": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577356/sample_villa9_qefin3.jpg",
                "sample villa10": "https://res.cloudinary.com/dqlrfkgt0/image/upload/v1752577355/sample_villa10_u70edd.jpg",
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
    if name == "Sobha Central":
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "sobha_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            tpl=REAL_ESTATE_PROMPT,
        )
    if name == "Samana Portofonio":
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "samana_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            tpl=REAL_ESTATE_PROMPT,
        )
    if name == "Marriot Residencies Jumeirah Lake Towers":
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "marriot_jlt_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            tpl=REAL_ESTATE_PROMPT,
        )
    if name == "Damac Riverside":
        return dict(
            vector=FAISS.load_local(
                os.path.join(BASE_DIR, "riverside_faiss"),
                embedding,
                allow_dangerous_deserialization=True,
            ),
            tpl=REAL_ESTATE_PROMPT,
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

    return llm.invoke(messages).content.strip()


# wrapper filters -------------------------------------------------------------
def _violates_policy(text: str, history):
    pol_prompt = f"""You are a content-filter. Reply ONLY "BLOCK" or "ALLOW".
Text: "{text}" """
    return _ask_llm(pol_prompt, history).upper() == "BLOCK"


# def _is_greeting(text: str, history):
#     g_prompt = (
#         f"""Reply "GREETING" if "{text}" is just a greeting/ vague, else "QUERY":"""
#     )
#     return _ask_llm(g_prompt, history).upper() == "GREETING"


# ──────────────────────────────────────────────────────────────────────────────
def generate_response(project: str, history: list[dict], voice_mode: bool):
    """
    history: full chat so far, **last item must be the latest USER msg**.
    Returns {text:str, image_url:str|None}
    """
    print("Testing")
    cfg = _project_cfg(project)
    print("Testing: after cfg")
    user_input = history[-1]["content"]

    # 1 early exits -----------------------------------------------------------
    # if _is_greeting(user_input, history):
    #     print("Yo")
    #     return dict(
    #         text=f"Hi! I'm your assistant for {project}. Ask me anything!",
    #         image_url=None,
    #     )
    # if _violates_policy(user_input, history):
    #     return dict(text="Query blocked due to policy.", image_url=None)

    # 2 vector context --------------------------------------------------------
    docs = cfg["vector"].similarity_search(user_input, k=3)
    context = "\n".join(d.page_content for d in docs)
    VOICE_PROMPT_TEMPLATE = ""
    if voice_mode:
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
    prompt = (
        "Analyze the user's emotional tone and respond accordingly.\n\n"
        + cfg["tpl"].format(
            context=context,
            query=user_input,
            image_keywords=", ".join(cfg.get("images", {}).keys()),
        )
        + VOICE_PROMPT_TEMPLATE
    )
    answer = _ask_llm(prompt, history)
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
