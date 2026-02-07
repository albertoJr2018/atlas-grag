"""
Atlas-GRAG Streamlit Dashboard.

Interactive web interface for querying the supply chain knowledge base
with hybrid retrieval and multi-hop graph reasoning.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.database.graph_db import GraphDatabaseManager
from src.database.vector_db import VectorDatabaseManager
from src.retriever.hybrid import HybridRetriever
from src.llm.chains import ReasoningChain

logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Atlas-GRAG",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for premium look
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Info boxes */
    .graph-path-box {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(0, 212, 255, 0.1));
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* Status indicators */
    .status-online {
        color: #10b981;
    }
    .status-offline {
        color: #ef4444;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(22, 33, 62, 0.95);
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_retriever() -> HybridRetriever:
    """Get cached hybrid retriever instance."""
    try:
        return HybridRetriever()
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        return None


@st.cache_resource
def get_reasoning_chain() -> ReasoningChain:
    """Get cached reasoning chain instance."""
    try:
        return ReasoningChain()
    except Exception as e:
        logger.error(f"Failed to create reasoning chain: {e}")
        return None


def check_system_health() -> dict:
    """Check health of all system components."""
    health = {
        "neo4j": False,
        "chromadb": False,
        "ollama": False
    }
    
    # Check Neo4j
    try:
        with GraphDatabaseManager() as graph:
            health["neo4j"] = graph.is_healthy()
    except Exception:
        pass
    
    # Check ChromaDB
    try:
        vector = VectorDatabaseManager()
        health["chromadb"] = vector.is_healthy()
    except Exception:
        pass
    
    # Check Ollama (simple ping)
    try:
        import httpx
        config = get_config().ollama
        response = httpx.get(f"{config.base_url}/api/tags", timeout=5)
        health["ollama"] = response.status_code == 200
    except Exception:
        pass
    
    return health


def render_sidebar():
    """Render the sidebar with system status and settings."""
    st.sidebar.title("🌐 Atlas-GRAG")
    st.sidebar.markdown("*Multi-Hop Graph Reasoning*")
    
    st.sidebar.markdown("---")
    
    # System Health
    st.sidebar.subheader("⚡ System Status")
    
    health = check_system_health()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if health["neo4j"]:
            st.success("Neo4j ✓")
        else:
            st.error("Neo4j ✗")
        
        if health["chromadb"]:
            st.success("ChromaDB ✓")
        else:
            st.error("ChromaDB ✗")
    
    with col2:
        if health["ollama"]:
            st.success("Ollama ✓")
        else:
            st.error("Ollama ✗")
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    
    use_graph = st.sidebar.checkbox("Include Graph Reasoning", value=True)
    show_reasoning = st.sidebar.checkbox("Show Reasoning Path", value=True)
    
    st.sidebar.markdown("---")
    
    # Sample Questions
    st.sidebar.subheader("💡 Sample Questions")
    
    example_questions = [
        "How will the Singapore strike impact GlobalTech's ability to compete with EuroComputing?",
        "What companies depend on FlowChips?",
        "Which supply chains are at risk from the Port of Singapore backlog?",
        "How is TechFlow Inc connected to GlobalTech?",
    ]
    
    for q in example_questions:
        if st.sidebar.button(q[:50] + "...", key=q):
            st.session_state.current_question = q
    
    return use_graph, show_reasoning


def render_graph_paths(result):
    """Render knowledge graph paths in a visually appealing way."""
    if result.graph_paths:
        st.markdown("### 🔗 Knowledge Graph Paths")
        
        for i, path in enumerate(result.graph_paths, 1):
            path_str = path.to_string()
            if path_str:
                st.markdown(f"""
                <div class="graph-path-box">
                    <strong>Path {i}:</strong><br>
                    <code>{path_str}</code>
                </div>
                """, unsafe_allow_html=True)
    
    if result.graph_context:
        with st.expander("📊 Full Graph Context"):
            st.code(result.graph_context)


def render_entities(entities: list):
    """Render extracted entities as pills."""
    if entities:
        st.markdown("### 🏷️ Extracted Entities")
        cols = st.columns(min(len(entities), 4))
        for i, entity in enumerate(entities):
            with cols[i % 4]:
                st.info(entity)


def main():
    """Main application entry point."""
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    
    # Render sidebar
    use_graph, show_reasoning = render_sidebar()
    
    # Main content
    st.title("🌐 Atlas-GRAG")
    st.markdown("*Mapping Unseen Global Supply Chain Risks via Multi-Hop Graph Reasoning*")
    
    # Introduction
    with st.expander("ℹ️ About Atlas-GRAG", expanded=False):
        st.markdown("""
        **Atlas-GRAG** combines **Knowledge Graphs** with **Vector Search** to answer 
        complex supply chain questions that require multi-hop reasoning.
        
        Unlike standard RAG systems that only find similar documents, Atlas-GRAG can:
        - 🔗 **Trace relationships** across entities (companies, products, locations)
        - 🌐 **Discover hidden connections** through graph traversal
        - 🎯 **Identify cascade effects** from disruption events
        
        **Example:** "How will a Singapore port strike affect GlobalTech?"
        
        Atlas-GRAG finds: Strike → Singapore → TechFlow → FlowChips → GlobalTech
        """)
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "paths" in message and message["paths"]:
                with st.expander("🔗 Graph Reasoning Path"):
                    st.code(message["paths"])
    
    # Check for preset question from sidebar
    if st.session_state.current_question:
        prompt = st.session_state.current_question
        st.session_state.current_question = None
    else:
        prompt = st.chat_input("Ask about supply chain risks...")
    
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                retriever = get_retriever()
                chain = get_reasoning_chain()
                
                if retriever is None or chain is None:
                    st.error("System components not available. Please check connections.")
                    return
                
                # Perform hybrid retrieval
                try:
                    result = retriever.retrieve(prompt, include_graph=use_graph)
                except Exception as e:
                    st.warning(f"Graph retrieval failed, using vector-only: {e}")
                    result = retriever.retrieve(prompt, include_graph=False)
                
                # Show entities
                if result.entities and show_reasoning:
                    render_entities(result.entities)
                
                # Show graph paths
                if show_reasoning and (result.graph_paths or result.graph_context):
                    render_graph_paths(result)
                
                # Generate answer
                with st.spinner("🧠 Reasoning..."):
                    response = chain.reason(result, prompt, use_chain_of_thought=True)
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(response.answer)
                
                # Show reasoning trace
                if show_reasoning and response.reasoning:
                    with st.expander("🔍 Reasoning Trace"):
                        st.markdown(response.reasoning)
                
                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "paths": result.graph_context
                })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Neo4j, ChromaDB, and Ollama"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
