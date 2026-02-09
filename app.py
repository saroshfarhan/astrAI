"""Streamlit UI for the Multi-Agent ad-astrAI Pipeline.

This UI visualizes the complete multi-agent workflow:
- File upload & modality detection
- Real-time agent execution tracking
- Natural language reports
- Consolidated spectral/image predictions
- Dynamic Knowledge Base visualization
- Validation flags and quality checks
- Interactive Chat with Data (RAG)
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
import mlflow
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import RUN_GRAPH, configure_mlflow

# Load environment variables
load_dotenv(find_dotenv(usecwd=True))

# Configure page
st.set_page_config(
    page_title="ad-astrAI Multi-Agent",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for agent visualization
st.markdown("""
<style>
    .agent-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
        border-left: 5px solid #4e8cff;
    }
    .agent-active {
        border-left: 5px solid #00c853;
        background-color: #e8f5e9;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stAlert {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 ad-astrAI: Multi-Agent Spectro-Visual Analysis")
st.markdown("### Autonomous Multi-Modal Astronomical Analysis System")


def infer_modality(filename: str) -> str:
    """Infer modality from file extension."""
    name = filename.lower()
    if "complex" in name or "multi" in name:
        return "both"  # Heuristic for demo purposes
    if name.endswith((".fits", ".csv")):
        return "spectral"
    if name.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    return "unknown"


# Initialize Session State
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "final_state" not in st.session_state:
    st.session_state.final_state = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.header("🎮 Mission Control")
    
    # API Keys
    st.subheader("System Status")
    key_status = {
        "Google Gemini": bool(os.getenv("GOOGLE_API_KEY")),
    }
    for service, active in key_status.items():
        st.write(f"**{service}**: {'✅ Active' if active else '❌ Missing'}")
    
    st.divider()
    st.info("Upload FITS for spectral analysis or PNG/JPG for image analysis. Use filenames with 'complex' to trigger multi-modal routing.")
    
    # Reset Button
    if st.button("🔄 New Analysis"):
        st.session_state.analysis_complete = False
        st.session_state.final_state = {}
        st.session_state.chat_history = []
        st.rerun()

# File Uploader
uploaded_file = st.file_uploader(
    "Upload Observation Data",
    type=["fits", "csv", "png", "jpg", "jpeg"],
    help="Supports FITS/CSV (Spectral) and PNG/JPG (Visual)"
)

if uploaded_file:
    # Determine Modality
    modality = infer_modality(uploaded_file.name)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
    with col3:
        st.metric("Detected Modality", modality.upper(), 
                 delta="Dual Mode" if modality == "both" else "Single Mode")
    
    # Analysis Trigger
    if not st.session_state.analysis_complete:
        if st.button("🚀 Launch Analysis", type="primary", width="stretch"):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                with st.spinner("🤖 Orchestrating Agents..."):
                    configure_mlflow()
                    
                    # Run the Graph
                    with mlflow.start_run(run_name="streamlit-session") as run:
                        initial_state = {
                            "input_path": tmp_path, 
                            "modality": modality,
                        }
                        
                        # Execute Pipeline
                        final_state = RUN_GRAPH(initial_state)
                        
                        # Log Streamlit context
                        mlflow.log_param("source", "streamlit")
                        mlflow.log_param("uploaded_filename", uploaded_file.name)
                        
                        # Store in session state
                        st.session_state.final_state = final_state
                        st.session_state.analysis_complete = True
                        st.rerun()

            except Exception as exc:
                st.error(f"Analysis Failed: {exc}")
                st.exception(exc)
            
            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # Persistence: Display Results if Analysis Complete
    if st.session_state.analysis_complete:
        final_state = st.session_state.final_state
        
        st.success("Analysis Complete!")
        
        # 1. Main Report
        st.header("📝 Mission Report")
        report = final_state.get("metadata", {}).get("report", "No report generated.")
        st.markdown(report)
        
        # 2. Agent Execution Path
        with st.expander("🕵️‍♂️ Agent Trace (Execution Path)", expanded=False):
            active_agents = final_state.get("active_agents", [])
            st.write(f"**Execution Route:** {' → '.join(active_agents)}")
            st.progress(100)
        
        # 3. Validation Warnings
        flags = final_state.get("validation_flags", [])
        if flags:
            st.warning(f"⚠️ {len(flags)} Validation Checks Triggered")
            for flag in flags:
                st.write(f"- {flag}")
        
        # 4. Detailed Data Tabs
        tab1, tab2, tab3 = st.tabs(["🧪 Consolidated Predictions", "📚 Knowledge Base", "🔍 Raw Metadata"])
        
        with tab1:
            st.subheader("Validated Elements")
            results = final_state.get("validated_results", [])
            if results:
                df = pd.DataFrame(results)
                # Format dataframe
                display_cols = ["element", "probability", "confidence_flag", "source", "multi_modal", "rationale"]
                cols = [c for c in display_cols if c in df.columns]
                st.dataframe(
                    df[cols].style.background_gradient(subset=["probability"], cmap="Greens"),
                    use_container_width=True
                )
            else:
                st.info("No elements detected.")
        
        with tab2:
            st.subheader("Dynamic Session Knowledge Base")
            kb = final_state.get("knowledge_base", {})
            if kb:
                # Convert KB dict to readable format
                kb_items = []
                for elem, data in kb.items():
                    kb_items.append({
                        "Element": f"{data['name']} ({elem})",
                        "Confidence": f"{data['confidence']:.1%}",
                        "Sources": ", ".join(data['sources']),
                        "Multi-Modal": "✅" if data['multi_modal'] else "❌"
                    })
                st.table(kb_items)
            else:
                st.info("Knowledge Base is empty.")
        
        with tab3:
            st.json(final_state)
            
        # MLflow Link (Hidden for end users)
        # run_id = final_state.get("log_refs", {}).get("mlflow")
        # if run_id:
        #     st.markdown(f"**[View Trace in MLflow](http://127.0.0.1:5000/#/experiments/1/runs/{run_id})**")

        st.divider()

        # ====================================================================
        # CHAT INTERFACE
        # ====================================================================
        st.header("💬 Chat with Data")
        st.caption("Ask questions about the analysis results, spectral composition, or image features.")

        # Display Chat History
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask about the confirmed elements..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        # Prepare context from Pipeline State
                        context = f"""
                        Analysis Context:
                        - Predictions: {final_state.get('validated_results', [])}
                        - Knowledge Base: {final_state.get('knowledge_base', {})}
                        - Modality: {final_state.get('modality')}
                        - Validation Flags: {final_state.get('validation_flags', [])}
                        """
                        
                        system_prompt = f"""You are an astronomical assistant analyzing a data pipeline.
                        Use the provided context to answer the user's question.
                        Be scientific but accessible. If the answer isn't in the data, say so.
                        
                        {context}
                        """
                        
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                        response = llm.invoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=prompt)
                        ])
                        
                        answer = response.content
                        st.markdown(answer)
                        
                        # Add assistant message
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

else:
    # Placeholder for idle state
    st.info("Upload a file in the above to begin analysis.")
    
    # # Demo/Mock visual
    # st.markdown("### System Architecture")
    # st.code("""
    # Orchestrator
    #    ↓
    # [Routing] → Spectral / Image / Both
    #    ↓
    # Inference (Consolidation & KB)
    #    ↓
    # Validator (Quality Control)
    #    ↓
    # Reporter (Natural Language)
    # """, language="text")
