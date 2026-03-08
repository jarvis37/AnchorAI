import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Import the retrieve function from retrieval.py
try:
    from retrieval import retrieve
except ImportError:
    st.error("Failed to import retrieval.py. Ensure you are running this from the AnchorAI root directory.")
    st.stop()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

# We use the standard OpenAI client which automatically picks up OPENAI_API_KEY from environment
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_MESSAGE = """
You are a helpful assistant that answers questions based on the provided context retrieved from the user's personal markdown knowledge base.
If the answer is not in the context, clearly state that the answer is not provided to you in the retrieved notes.
Keep your answers concise, clear, and highly relevant to the provided context.
"""

st.set_page_config(
    page_title="AnchorAI",
    page_icon="⚓",
    layout="wide"
)

# Custom Styling for Premium Feel
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Core Chat Logic
# ---------------------------------------------------------------------------

def generate_response(user_inquiry, history):
    """
    1. Retrieve relevant text chunks from the RAG pipeline.
    2. Format the retrieved context alongside the user's question.
    3. Generate the response using OpenAI's API.
    """
    
    # RAG Retrieval
    with st.spinner("Retrieving relevant notes..."):
        try:
            rag_context = retrieve(user_inquiry)
        except Exception as e:
            return f"🚨 Failed to search the knowledge base: {str(e)}"
    
    # Construct Context Payload
    if rag_context:
        # Convert list of Document objects to string
        context_string = "\\n\\n---\\n\\n".join([doc.page_content for doc in rag_context])
        user_content = f"CONTEXT FROM NOTES:\\n{context_string}\\n\\nQUESTION: {user_inquiry}"
    else:
        user_content = user_inquiry

    # Build Message List
    api_messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    
    # Append last 5 messages for conversation context (to avoid giant token payloads)
    for msg in history[-5:]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    api_messages.append({"role": "user", "content": user_content})

    # Call LLM
    try:
        response = client.chat.completions.create(
            model=MODEL, 
            messages=api_messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"🚨 API Generation Error: {str(e)}"


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/anchor.png", width=80)
    st.markdown("## ⚓ AnchorAI")
    st.caption("Your Personal Knowledge Base Assistant")
    
    st.divider()
    
    st.markdown("### How it works")
    st.markdown("""
    1. **Ingest Notes**: Run `python ingestion.py` to embed your Markdown files.
    2. **Retrieve Context**: When you ask a question, AnchorAI queries a local `ChromaDB` for the most relevant chunks.
    3. **Generate Answer**: The retrieved chunks are sent to the LLM to generate an accurate, grounded response.
    """)
    
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# Main Window
st.markdown("<h1 class='main-title'>AnchorAI Chat</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask questions directly against your stored Markdown notes.</p>", unsafe_allow_html=True)


# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Handler
prompt = st.chat_input("Ask about your notes...")
if prompt:
    # Immediately render User Prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and Render Assistant Reply
    with st.chat_message("assistant"):
        ai_reply = generate_response(prompt, st.session_state.messages[:-1])
        st.markdown(ai_reply)
        
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
