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
import shutil
import tempfile
from pathlib import Path

import streamlit as st
import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import RUN_GRAPH, configure_mlflow
from agent.agents.image_model import ImageModelAgent

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


def check_spectral_service() -> bool:
    """Check if Spectral Service backend is alive."""
    try:
        spectral_url = os.getenv("SPECTRAL_SERVICE_URL", "http://localhost:8001")
        response = requests.get(f"{spectral_url}/docs", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


# Initialize Session State
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "final_state" not in st.session_state:
    st.session_state.final_state = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.header("🎮 Mission Control")

    # System Status
    st.subheader("System Status")

    # Check services
    gemini_active = bool(os.getenv("GOOGLE_API_KEY"))
    spectral_active = check_spectral_service()

    service_status = {
        "Google Gemini": gemini_active,
        "Spectral Service": spectral_active,
    }

    for service, active in service_status.items():
        status_emoji = '✅' if active else '❌'
        status_text = 'Active' if active else ('Missing' if service == "Google Gemini" else 'Offline')
        st.write(f"**{service}**: {status_emoji} {status_text}")

    # Warning if Spectral Service is down
    if not spectral_active:
        st.warning("⚠️ Start Spectral Service:\n```bash\ncd \"Spectral Service\"\nuvicorn app.main:app --port 8001\n```")
    
    st.divider()
    st.info("""
**Analysis Modes:**

**🔬 Spectral Analysis** (Recommended for Unknown Exoplanets)
- Upload: Single FITS/CSV file
- Works with: UV-only OR IR-only data
- Output: Molecular composition predictions

**🎨 Graphical Analysis** (Known Planets Only)
- Upload: Both UV and IR PKL files
- Requires: Complete spectral data (UV+IR)
- Output: Comparative fingerprint visualizations
    """)
    
    # Reset Button
    if st.button("🔄 New Analysis"):
        st.session_state.analysis_complete = False
        st.session_state.final_state = {}
        st.session_state.chat_history = []
        st.rerun()

# Analysis Type Selector
st.subheader("Select Analysis Type")
analysis_type = st.radio(
    "Choose the type of analysis:",
    options=["Spectral Analysis (FITS/CSV)", "Graphical Analysis (PKL)"],
    horizontal=True,
    help="Spectral: Analyze UV/IR spectra using trained models | Image: Generate spectral barcode visualizations"
)

# Conditional File Uploaders
st.divider()
uploaded_file = None
uploaded_files = None
modality = None

if analysis_type == "Spectral Analysis (FITS/CSV)":
    st.subheader("📡 Spectral Data Upload")
    st.caption("Upload FITS or CSV file containing spectral data for molecular composition analysis")
    uploaded_file = st.file_uploader(
        "Upload Spectral Data",
        type=["fits", "csv"],
        help="FITS or CSV files with wavelength and flux data. The system will auto-detect UV vs IR domain."
    )
    if uploaded_file:
        st.info(f"**Mode**: Spectral Analysis | **File**: {uploaded_file.name}")
        modality = "spectral"

else:  # Graphical Analysis
    st.subheader("🎨 Planet Data Upload (PKL)")
    st.caption("Upload planet PKL file(s) to generate spectral fingerprint visualizations")
    st.warning("⚠️ **Requires both UV and IR data** for complete spectral fingerprints. For unknown exoplanets with incomplete data, use Spectral Analysis mode instead.")
    uploaded_files = st.file_uploader(
        "Upload Planet PKL File(s)",
        type=["pkl"],
        accept_multiple_files=True,
        help="REQUIRES both UV and IR data. Upload: (1) Combined file (jupiter_combined.pkl), OR (2) Both files (jupiter_uv.pkl + jupiter_ir.pkl). Single-domain files will only work if the pair exists in training data."
    )
    if uploaded_files:
        file_names = ", ".join([f.name for f in uploaded_files])
        st.info(f"**Mode**: Graphical Analysis (Spectral Barcodes) | **Files**: {file_names}")
        modality = "image"
        # For backward compatibility, treat as single file if only one uploaded
        uploaded_file = uploaded_files[0] if len(uploaded_files) == 1 else uploaded_files

if (modality == "spectral" and uploaded_file) or (modality == "image" and uploaded_files):
    # Handle both single file (spectral) and multi-file (image) scenarios
    files_to_display = uploaded_files if modality == "image" else [uploaded_file]

    # Display file info
    col1, col2, col3 = st.columns(3)
    with col1:
        if len(files_to_display) == 1:
            st.metric("File Name", files_to_display[0].name)
        else:
            st.metric("Files Uploaded", len(files_to_display))
    with col2:
        total_size = sum(f.size for f in files_to_display) / 1024
        st.metric("Total Size", f"{total_size:.1f} KB")
    with col3:
        mode_display = "Spectral (UV/IR)" if modality == "spectral" else "Image (Barcodes)"
        st.metric("Analysis Mode", mode_display)

    # Analysis Trigger
    if not st.session_state.analysis_complete:
        if st.button("🚀 Launch Analysis", type="primary", width="stretch"):
            tmp_paths = []
            try:
                # Save temp file(s)
                if modality == "spectral":
                    # Single file for spectral analysis
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_paths.append(tmp.name)
                    input_path = tmp_paths[0]
                    uploaded_filenames = uploaded_file.name
                else:
                    # Multiple files for Graphical Analysis - save all to same temp directory
                    temp_dir = tempfile.mkdtemp()
                    for file in uploaded_files:
                        file_path = Path(temp_dir) / file.name
                        file_path.write_bytes(file.getbuffer())
                        tmp_paths.append(str(file_path))
                    # Pass directory path or first file path
                    input_path = tmp_paths[0]
                    uploaded_filenames = ", ".join([f.name for f in uploaded_files])

                # Different execution paths for different modalities
                if modality == "image":
                    # DIRECT Graphical Analysis - Bypass pipeline, call agent directly
                    with st.spinner("🎨 Generating Spectral Visualizations..."):
                        try:
                            # Call ImageModelAgent directly
                            image_agent = ImageModelAgent()
                            agent_state = {
                                "input_path": input_path,
                                "modality": "image",
                                "errors": [],
                                "metadata": {}
                            }

                            # Generate visualizations
                            result_state = image_agent.process(agent_state)

                            # Store in session state
                            st.session_state.final_state = result_state
                            st.session_state.analysis_complete = True
                            st.rerun()

                        except Exception as e:
                            st.error(f"Visualization failed: {e}")
                            st.exception(e)

                else:
                    # SPECTRAL ANALYSIS - Use full pipeline
                    with st.spinner("🤖 Orchestrating Agents..."):
                        configure_mlflow()

                        # Run the Graph
                        with mlflow.start_run(run_name="streamlit-session") as run:
                            initial_state = {
                                "input_path": input_path,
                                "modality": modality,
                            }

                            # Execute Pipeline
                            final_state = RUN_GRAPH(initial_state)

                            # Log Streamlit context
                            mlflow.log_param("source", "streamlit")
                            mlflow.log_param("uploaded_filename", uploaded_filenames)

                            # Store in session state
                            st.session_state.final_state = final_state
                            st.session_state.analysis_complete = True
                            st.rerun()

            except Exception as exc:
                st.error(f"Analysis Failed: {exc}")
                st.exception(exc)

            finally:
                # Cleanup all temp files
                for tmp_path in tmp_paths:
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                # Cleanup temp directory if created
                if modality == "image" and tmp_paths:
                    temp_dir = Path(tmp_paths[0]).parent
                    if temp_dir.exists():
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception:
                            pass

    # Persistence: Display Results if Analysis Complete
    if st.session_state.analysis_complete:
        final_state = st.session_state.final_state
        current_modality = final_state.get("modality", "unknown")

        st.success("✅ Analysis Complete!")

        # ========================================================================
        # Graphical Analysis DISPLAY (Simplified - just visualizations + chat)
        # ========================================================================
        if current_modality == "image":
            st.header("📊 Spectral Fingerprint Visualizations")

            # Show which planets were analyzed
            metadata = final_state.get("metadata", {})
            if "planets_analyzed" in metadata:
                planets_list = metadata["planets_analyzed"]
                st.success(f"✅ Analyzed {len(planets_list)} planets: **{', '.join(planets_list).title()}**")

                # Highlight if user uploaded a new planet
                if "uploaded_planet" in metadata and metadata["uploaded_planet"] != "none":
                    uploaded = metadata["uploaded_planet"]
                    st.info(f"🆕 New planet **{uploaded.title()}** added to comparison!")

            # Display errors if any
            errors = final_state.get("errors", [])
            if errors:
                st.error("⚠️ Errors occurred during visualization:")
                for error in errors:
                    st.write(f"- {error}")

            # Display visualizations
            visualizations = final_state.get("image_visualizations", [])
            if visualizations:
                st.caption(f"Generated {len(visualizations)} spectral barcode visualizations")

                for i, viz in enumerate(visualizations):
                    st.subheader(f"{i+1}. {viz['title']}")
                    st.caption(viz.get("description", ""))

                    # Display matplotlib figure
                    st.pyplot(viz["figure"])

                    if i < len(visualizations) - 1:
                        st.divider()

                st.divider()

                # ============================================================
                # CHAT WITH VISUALIZATIONS
                # ============================================================
                st.header("💬 Chat About Visualizations")
                st.caption("Ask questions about the spectral patterns, planet similarities, or molecular fingerprints")

                # Display Chat History
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat Input
                if prompt := st.chat_input("Ask about the visualizations..."):
                    # Add user message
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing visualizations..."):
                            try:
                                # Prepare context about visualizations
                                viz_summary = "\n".join([
                                    f"- {v['title']}: {v.get('description', '')}"
                                    for v in visualizations
                                ])

                                context = f"""
                                You are analyzing spectral fingerprint visualizations of planetary atmospheres.

                                Visualizations generated:
                                {viz_summary}

                                The visualizations show:
                                - Combined UV+IR spectral barcodes for multiple planets
                                - Similarity matrix showing distances between planetary spectra
                                - Hierarchical clustering dendrogram showing planet groupings
                                - Overlaid spectral comparison

                                Answer the user's question based on typical spectral analysis patterns.
                                Be scientific but accessible.
                                """

                                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                                response = llm.invoke([
                                    SystemMessage(content=context),
                                    HumanMessage(content=prompt)
                                ])

                                answer = response.content
                                st.markdown(answer)

                                # Add assistant message
                                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                            except Exception as e:
                                st.error(f"Error generating response: {e}")

            else:
                st.warning("No visualizations were generated. Check errors above.")

            # Optional: Show raw state for debugging
            with st.expander("🔍 Debug Info", expanded=False):
                st.json(final_state)

        # ========================================================================
        # SPECTRAL ANALYSIS DISPLAY (Full pipeline output)
        # ========================================================================
        else:
            # 1. Main Report
            st.header("📝 Mission Report")
            report = final_state.get("metadata", {}).get("report", "No report generated.")
            st.markdown(report)

            # 2. Agent Execution Path
            with st.expander("🕵️‍♂️ Agent Trace (Execution Path)", expanded=False):
                active_agents = final_state.get("active_agents", [])
                st.write(f"**Execution Route:** {' → '.join(active_agents)}")
                st.progress(100)

            # 3. Errors (if any)
            errors = final_state.get("errors", [])
            if errors:
                st.error(f"🚨 {len(errors)} Error(s) Occurred")
                for error in errors:
                    st.write(f"- {error}")

            # 4. Validation Warnings
            flags = final_state.get("validation_flags", [])
            if flags:
                st.warning(f"⚠️ {len(flags)} Validation Checks Triggered")
                for flag in flags:
                    st.write(f"- {flag}")

            # 5. Detailed Data Tabs
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
            # SPECTRAL BARCODE VISUALIZATIONS (Image Agent Output)
            # ====================================================================
            visualizations = final_state.get("image_visualizations", [])
            if visualizations:
                st.header("📊 Spectral Fingerprint Analysis")
                st.caption("Multi-planet spectral barcode visualizations generated by the Image Model Agent")

                # Display each visualization
                for viz in visualizations:
                    st.subheader(viz["title"])
                    st.caption(viz["description"])

                    # Display matplotlib figure
                    st.pyplot(viz["figure"])

                    st.divider()

            st.divider()

            # ====================================================================
            # CHAT INTERFACE (for Spectral Analysis)
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
                            #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
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
