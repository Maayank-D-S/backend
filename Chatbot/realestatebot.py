import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4.1", temperature=0.2, api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
BASE_DIR = os.getcwd()

PROJECTS = {
    "krupal habitat": {
        "vector": FAISS.load_local(
            os.path.join(BASE_DIR, "krupalfinal_faiss"),
            embedding,
            allow_dangerous_deserialization=True,
        )
    },
    "ramvan villas": {
        "vector": FAISS.load_local(
            os.path.join(BASE_DIR, "ramvan_villas_faiss"),
            embedding,
            allow_dangerous_deserialization=True,
        )
    },
}


def extract_project_names(user_query):
    prompt = f"""
Extract only the names of real estate projects mentioned in this query. Available projects: Krupal Habitat, Ramvan Villas. If none, return "None".

Query: "{user_query}"
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().lower()


def get_faiss_context(project_names, user_query):
    if not project_names or project_names.lower() == "none":
        return None

    names = [name.strip().lower() for name in project_names.split(",")]
    context_chunks = []

    for name in names:
        project = PROJECTS.get(name)
        if project:
            docs = project["vector"].similarity_search(user_query, k=3)
            context_text = (
                "\n".join([doc.page_content for doc in docs])
                if docs
                else "No additional details."
            )
            context_chunks.append(context_text)

    return "\n".join(context_chunks) if context_chunks else None


def build_prompt(faiss_context, user_input):
    positivity_prompt = """
You are an expert real estate consultant who always speaks positively about Dubai, Uttarakhand, and Dholera real estate markets.

 For Dubai:
- Highlight world-class infrastructure, high rental yields, luxury living, and strong global investor interest.

 For Uttarakhand:
- Emphasize natural beauty, tourism growth, rising demand for vacation homes, and peaceful living.

For Dholera:
- Focus on India's first smart city, rapid development, government support, and long-term investment potential.

 General Guidance:
- Always provide helpful, confident, and optimistic real estate advice.
- Never mention risks, problems, delays, or anything negative.
- Subtly suggest our projects (Krupal Habitat in Dholera, Ramvan Villas in Uttarakhand) where appropriate.

 Project-Specific Guidance:
- If project details are provided, confidently highlight features, pricing, and benefits in a friendly tone.

 Response Style:
- Keep answers  short and crisp:answer in minimum number of points.
- Avoid long explanations or unnecessary details.

---
Project Information:
{faiss_context}

User Question:
{user_input}

Answer:
"""
    return positivity_prompt.format(
        faiss_context=(
            faiss_context if faiss_context else "No specific project details available."
        ),
        user_input=user_input,
    )


def _ask_llm(prompt, history):
    messages = []
    for h in history:
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        else:
            messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)
    return response.content.strip().replace("**", "")


def chatbot_loop():
    print("🏡 Welcome to the Real Estate Chatbot!")
    print("Ask me anything")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break

        history.append({"role": "user", "content": user_input})

        project_names = extract_project_names(user_input)
        faiss_context = (
            get_faiss_context(project_names, user_input)
            if project_names != "none"
            else None
        )

        prompt = build_prompt(faiss_context, user_input)
        response_text = _ask_llm(prompt, history)

        history.append({"role": "assistant", "content": response_text})

        print("\nBot:", response_text, "\n")


if __name__ == "__main__":
    chatbot_loop()
