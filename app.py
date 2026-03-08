import streamlit as st
import streamlit.components.v1 as components
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
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Premium Feel (Antigravity Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Background - Force Light Mode */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    /* Animation Canvas fixes */
    #canvas {
        z-index: 0 !important;
    }
    
    /* Bring content forward over canvas */
    .block-container {
        z-index: 1 !important;
    }

    /* Hide Streamlit Header & Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main Title Styling (Antigravity-inspired) */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        color: #111827;
        text-align: center;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-top: 3rem;
        margin-bottom: 1rem;
    }
    
    .hero-title .gradient-text {
        background: linear-gradient(90deg, #4285f4, #ea4335, #fbbc05, #34a853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #4b5563;
        font-size: 1.1rem;
        text-align: center;
        font-weight: 400;
        margin-bottom: 3rem;
        max-width: 650px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.5;
    }

    /* Clean Chat Messages */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 1rem 0rem !important;
    }

    /* Input Box Styling */
    .stChatInputContainer {
        border-radius: 9999px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
        padding-left: 1rem !important;
        max-width: 700px;
        margin: 0 auto;
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

# Inject Particle Animation
st.components.v1.html("""
<canvas id="canvas" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 9999;"></canvas>
<script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    let particlesArray = [];
    const colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853'];

    const mouse = {
        x: null,
        y: null,
        radius: 100
    }

    window.addEventListener('mousemove', function(event) {
        mouse.x = event.x;
        mouse.y = event.y;
    });

    window.addEventListener('resize', function() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });

    class Particle {
        constructor(x, y, dx, dy, size, color) {
            this.x = x;
            this.y = y;
            this.dx = dx;
            this.dy = dy;
            this.size = size;
            this.color = color;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2, false);
            ctx.fillStyle = this.color;
            ctx.fill();
        }

        update() {
            if (this.x + this.size > canvas.width || this.x - this.size < 0) {
                this.dx = -this.dx;
            }
            if (this.y + this.size > canvas.height || this.y - this.size < 0) {
                this.dy = -this.dy;
            }

            this.x += this.dx;
            this.y += this.dy;

            // Interactivity
            let dx = mouse.x - this.x;
            let dy = mouse.y - this.y;
            let distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < mouse.radius) {
                const forceDirectionX = dx / distance;
                const forceDirectionY = dy / distance;
                const maxDistance = mouse.radius;
                const force = (maxDistance - distance) / maxDistance;
                const directionX = forceDirectionX * force * 5;
                const directionY = forceDirectionY * force * 5;
                
                this.x -= directionX;
                this.y -= directionY;
            }

            this.draw();
        }
    }

    function init() {
        particlesArray = [];
        let numberOfParticles = (canvas.height * canvas.width) / 9000;
        for (let i = 0; i < numberOfParticles; i++) {
            let size = (Math.random() * 3) + 1;
            let x = (Math.random() * ((innerWidth - size * 2) - (size * 2)) + size * 2);
            let y = (Math.random() * ((innerHeight - size * 2) - (size * 2)) + size * 2);
            let dx = (Math.random() - 0.5) * 2;
            let dy = (Math.random() - 0.5) * 2;
            let color = colors[Math.floor(Math.random() * colors.length)];
            particlesArray.push(new Particle(x, y, dx, dy, size, color));
        }
    }

    function animate() {
        requestAnimationFrame(animate);
        ctx.clearRect(0, 0, innerWidth, innerHeight);
        for (let i = 0; i < particlesArray.length; i++) {
            particlesArray[i].update();
        }
    }

    init();
    animate();
</script>
""", height=800)

st.markdown("<h1 class='hero-title'>Experience liftoff with your<br>next-generation <span class='gradient-text'>assistant</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A highly intuitive Retrieval-Augmented Generation system that securely grounds every conversation directly in your personal knowledge base.</p>", unsafe_allow_html=True)


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
