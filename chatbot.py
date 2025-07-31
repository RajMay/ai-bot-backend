from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import uvicorn
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 restrict later
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Bot is running 🚀"}

# config
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# ✅ Use Gemini embeddings (instead of OpenAI)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")  # 👈 pass your Gemini key
)

# ✅ Use Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro" if you want stronger
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# connect to chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# retriever
retriever = vector_store.as_retriever(search_kwargs={'k': 3})



# # chatbot response
# def stream_response(message, history):
#     # retrieve context
#     docs = retriever.invoke(message)
#     knowledge = "\n\n".join([doc.page_content for doc in docs])

#     rag_prompt = f"""
#     You are an assistant answering questions ONLY based on the knowledge provided.
#     Do not use outside/internal knowledge.You dont mention anything to the user about the provided knowledge
    

#     Question: {message}
#     Conversation history: {history}

#     The knowledge:
#     {knowledge}
#     """

#     partial_message = ""
#     for response in llm.stream(rag_prompt):
#         partial_message += response.content
#         yield partial_message

# # Gradio app
# chatbot = gr.ChatInterface(
#     stream_response,
#     textbox=gr.Textbox(placeholder="Ask me anything...",
#                        container=False,
#                        autoscroll=True,
#                        scale=7),
# )

# chatbot.launch()

class ChatRequest(BaseModel):
    message: str
    history: list

@app.post("/chat")
async def chat(req: ChatRequest):
    docs = retriever.invoke(req.message)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
    You are an assistant answering questions ONLY based on the knowledge provided.
    Do not use outside/internal knowledge.You dont mention anything to the user about the provided knowledge
    
    Question: {req.message}
    History: {req.history}
    Knowledge: {knowledge}
    """

    response = llm.invoke(rag_prompt)
    return {"reply": response.content}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render provides PORT
    uvicorn.run("chatbot:app", host="0.0.0.0", port=port, reload=False)