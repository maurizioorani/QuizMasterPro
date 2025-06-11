import streamlit as st
import os
import sys
import time
import re
from datetime import datetime
from dotenv import load_dotenv

# Add the src directory to sys.path to allow for absolute imports from quizmaster
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quizmaster.document_processor import DocumentProcessor
from quizmaster.quiz_generator import QuizGenerator
from quizmaster.database_manager import DatabaseManager
from quizmaster.vector_manager import VectorManager
from quizmaster.config import QuizConfig # Import QuizConfig


# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="QuizMaster Pro",
        page_icon="📚",
        layout="wide",
        menu_items={
            'about': None,
        }
    )
    
    st.title("📚 QuizMaster Pro - Document to Quiz Converter")
    st.markdown("Transform any document into an interactive quiz using local AI")

    # Initialize shared QuizConfig in session state if not already present
    if 'shared_quiz_config' not in st.session_state:
        st.session_state.shared_quiz_config = QuizConfig()
    
    # Use the shared config for all components that need it
    shared_config = st.session_state.shared_quiz_config

    # Initialize QuizGenerator
    # The "Always re-initialize" comment from original code is a bit ambiguous with session state.
    # For now, ensure it uses the shared_config. If it must be new, its internal config will be this shared one.
    st.session_state.quiz_generator = QuizGenerator(config=shared_config)

    # Initialize DocumentProcessor
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor(config=shared_config)
    else:
        # If it exists, ensure its config is the shared one and update its LLM if needed
        if st.session_state.document_processor.config is not shared_config:
            st.session_state.document_processor.config = shared_config # Point to the shared config
        # update_llm_configuration will be called when model is applied via UI button
        # No need to call it here on every rerun unless config object identity itself changed.

    # Initialize VectorManager
    if 'vector_manager' not in st.session_state:
        st.session_state.vector_manager = VectorManager(config=shared_config)
    else:
        # If it exists, ensure its config is the shared one
        if st.session_state.vector_manager.config is not shared_config:
            st.session_state.vector_manager.config = shared_config
        # update_llm_configuration will be called when model is applied via UI button

    if 'processed_content' not in st.session_state:
        st.session_state.processed_content = None
    
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False

    # Sidebar for settings and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Add navigation links
        st.page_link("pages/02_Saved_Quizzes.py", label="💾 Browse Saved Quizzes", icon="📚")
        st.divider()
        
        # Unified Model Selection for Both Embedding and Quiz Generation
        st.subheader("🤖 AI Model Selection")
        st.caption("This model will be used for both document processing (embedding) and quiz generation. Local models are automatically pulled if needed.")

        # Enhanced model descriptions with clearer status indicators
        MODEL_DESCRIPTIONS = {
            # Ollama Models (from LLMManager's predefined list)
            "llama3.3:8b": "⭐ Meta's Llama 3.3 (8B) - Strong general-purpose capabilities",
            "mistral:7b": "⭐ Mistral (7B) - Efficient, good for JSON generation (4.1GB)",
            "qwen2.5:7b": "⭐ Qwen 2.5 (7B) - Strong structured output, multilingual (4.4GB)",
            "deepseek-coder:6.7b": "⭐ DeepSeek Coder (6.7B) - Excellent for code-related tasks and structured output",
            # Add any other Ollama models from LLMManager's predefined list here if they differ

            # OpenAI Models (from QuizConfig)
            "gpt-4o-mini": "⚡ OpenAI: GPT-4o Mini - Fast, cost-effective, optimized for structured output",
            "gpt-4": "⚡ OpenAI: GPT-4 - High-quality, powerful model", # Description for gpt-4
            "gpt-3.5-turbo": "⚡ OpenAI: GPT-3.5 Turbo - Balanced performance and cost" # Description for gpt-3.5-turbo
        }
        
        # Get current model and local availability
        current_model = st.session_state.quiz_generator.llm_manager.get_current_model()
        all_local_models = st.session_state.quiz_generator.llm_manager.get_local_models() # Fetches all from Ollama
        all_available_models = st.session_state.quiz_generator.llm_manager.get_available_models() # Combined list
        openai_model_names = st.session_state.quiz_generator.llm_manager.get_openai_models()

        ollama_models_for_display = [m for m in all_available_models if m not in openai_model_names]
        openai_models_for_display = [m for m in all_available_models if m in openai_model_names]

        # Create model display options with status indicators and headers
        model_options = []
        
        # Ollama Header
        model_options.append({"name": "##OLLAMA_MODELS_HEADER##", "display": "--- Ollama Local Models ---", "description": "Locally hosted models via Ollama.", "is_header": True})
        for model in sorted(ollama_models_for_display):
            status = "📥 Auto-pull needed"
            if model in all_local_models: # Check against the dynamically fetched list
                status = "✅ Local"
            model_options.append({
                "name": model,
                "display": f"{model} - {status}",
                "description": MODEL_DESCRIPTIONS.get(model, "No description"),
                "is_header": False
            })

        # OpenAI Header
        model_options.append({"name": "##OPENAI_MODELS_HEADER##", "display": "--- OpenAI API Models ---", "description": "Models accessed via OpenAI API (requires API key).", "is_header": True})
        for model in sorted(openai_models_for_display):
            status = "⚡ API"
            if not os.environ.get("OPENAI_API_KEY"):
                status += " (API key missing)"
            model_options.append({
                "name": model,
                "display": f"{model} - {status}",
                "description": MODEL_DESCRIPTIONS.get(model, "No description"),
                "is_header": False
            })
        
        # Create selectbox with custom formatting
        # The format_func will just use the 'display' key. Headers are normal items.
        # We need to ensure the initial index is for a real model, not a header.
        try:
            initial_model_index = next(i for i, opt in enumerate(model_options) if opt["name"] == current_model and not opt.get("is_header"))
        except StopIteration:
            initial_model_index = 0 # Default to first item if current_model not found or is a header
            # Try to find the first non-header item if current_model wasn't found
            first_real_model_index = next((i for i, opt in enumerate(model_options) if not opt.get("is_header")), 0)
            initial_model_index = first_real_model_index


        selected_option_obj = st.selectbox(
            "Select AI Model",
            model_options,
            format_func=lambda x: x["display"], # Headers will be displayed as is
            index=initial_model_index,
            key="unified_model_select",
            help="This model will be used for both document processing and quiz generation"
        )

        # Handle if a header was somehow selected, though selectbox usually prevents this with distinct objects.
        # For safety, if a header is selected, try to use the actual current_model or default.
        if selected_option_obj.get("is_header"):
            st.warning("Please select an actual model, not a category header.")
            # Attempt to fall back to the globally set current_model if a header is chosen
            # This part might need more robust handling depending on Streamlit's selectbox behavior with identical display strings
            quiz_model = current_model
            selected_description = MODEL_DESCRIPTIONS.get(quiz_model, "No description")
        else:
            quiz_model = selected_option_obj["name"]
            selected_description = selected_option_obj["description"]
        
        # Show model description
        st.caption(selected_description)
        
        # Show info about unified model usage
        st.info("💡 This model handles both document processing (embedding/concept extraction) and quiz generation automatically.")
        
        # Check if current model is an OpenAI or Ollama model
        is_openai_model = quiz_model in openai_model_names
        
        # Use different layouts based on model type
        if is_openai_model:
            # For OpenAI models, just show the Apply button
            if st.button("Apply AI Model", key="apply_ai_model_openai", use_container_width=True):
                if not os.environ.get("OPENAI_API_KEY"):
                    st.error("OpenAI API key not found in .env file")
                else:
                    with st.spinner(f"Setting up AI model {quiz_model}..."):
                        try:
                            # Set model for all components to ensure consistency
                            st.session_state.quiz_generator.llm_manager.set_model(quiz_model) # This updates the shared config
                            
                            # Get the newly updated config
                            # updated_shared_config is not strictly needed here as components should use their self.config
                            # which points to the st.session_state.shared_quiz_config that llm_manager just updated.

                            # Update DocumentProcessor's LLM
                            if hasattr(st.session_state.document_processor, 'update_llm_configuration'):
                                st.session_state.document_processor.update_llm_configuration()
                            else:
                                st.warning("Dev: DocumentProcessor does not have update_llm_configuration method.")
                            
                            # Update VectorManager's LLM
                            if hasattr(st.session_state.vector_manager, 'update_llm_configuration'):
                                st.session_state.vector_manager.update_llm_configuration()
                            else:
                                st.warning("Dev: VectorManager does not have update_llm_configuration method.")
                            
                            st.success(f"✅ AI model set to {quiz_model} for all tasks")
                            st.info(f"🔄 {quiz_model} will handle both document processing and quiz generation")
                        except Exception as e:
                            st.error(f"Error setting AI model: {str(e)}")
        else:
            # For Ollama models, show Apply and Test buttons side by side
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Apply AI Model", key="apply_ai_model_ollama", use_container_width=True):
                    with st.spinner(f"Setting up AI model {quiz_model}..."):
                        try:
                            # For local models, check if available
                            if quiz_model not in all_local_models: # Corrected variable name
                                with st.status(f"📥 Downloading {quiz_model}...", expanded=True) as status:
                                    status.write("Starting model download...")
                                    success = st.session_state.quiz_generator.llm_manager.pull_model(quiz_model)
                                    if success:
                                        status.update(label="✅ Download completed!", state="complete", expanded=False)
                                    else:
                                        status.update(label="❌ Download failed", state="error")
                                        st.error(f"Failed to download {quiz_model}")
                                        st.stop()
                            
                            # Set model for all components to ensure consistency
                            st.session_state.quiz_generator.llm_manager.set_model(quiz_model) # This updates the shared config

                            # updated_shared_config is not strictly needed here.

                            # Update DocumentProcessor's LLM
                            if hasattr(st.session_state.document_processor, 'update_llm_configuration'):
                                st.session_state.document_processor.update_llm_configuration()
                            else:
                                st.warning("Dev: DocumentProcessor does not have update_llm_configuration method.")
                            
                            # Update VectorManager's LLM
                            if hasattr(st.session_state.vector_manager, 'update_llm_configuration'):
                                st.session_state.vector_manager.update_llm_configuration()
                            else:
                                st.warning("Dev: VectorManager does not have update_llm_configuration method.")
                            
                            st.success(f"✅ AI model set to {quiz_model} for all tasks")
                            st.info(f"🔄 {quiz_model} will handle both document processing and quiz generation")
                        except Exception as e:
                            st.error(f"Error setting quiz model: {str(e)}")
            
            with col2:
                # Test Connection for Quiz Model
                if st.button("🔗 Test Ollama", key="test_quiz_connection", use_container_width=True):
                    with st.spinner("Testing connection..."):
                        if st.session_state.quiz_generator.llm_manager.test_ollama_connection():
                            st.success("✅ Connection successful!")
                        else:
                            st.error("❌ Connection failed. Make sure Ollama is running")

        # Removed duplicate model selection code - using separate models now
        
        # Quiz Settings
        st.subheader("📋 Quiz Settings")
        num_questions = st.slider("Number of Questions", 1, 20, 5)
        difficulty = st.selectbox(
            "Difficulty Level", 
            ["Easy", "Medium", "Hard"],
            help="Easy: Basic facts, Medium: Understanding, Hard: Analysis"
        )
        
        question_types = st.multiselect(
            "Question Types",
            ["Multiple Choice", "Open-Ended", "True/False", "Fill-in-the-Blank"],
            default=["Multiple Choice"],
            help="Select one or more question types"
        )
        
        # Document Info
        if st.session_state.processed_content:
            st.subheader("📄 Document Info")
            metadata = st.session_state.processed_content['metadata']
            st.write(f"**File:** {st.session_state.processed_content['filename']}")
            st.write(f"**Format:** {st.session_state.processed_content['format'].upper()}")
            st.write(f"**Words:** {metadata['total_words']:,}")
            st.write(f"**Paragraphs:** {metadata['total_paragraphs']}")
            
            if metadata['chapters']:
                st.write(f"**Chapters:** {len(metadata['chapters'])}")
            if metadata['sections']:
                st.write(f"**Sections:** {len(metadata['sections'])}")

    # Main content area - Section 1: Document Upload & Processing
    st.header("📤 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=['pdf', 'docx', 'txt', 'html'],
        help="Supported formats: PDF, DOCX, TXT, HTML"
    )

    if uploaded_file:
        st.success(f"Document '{uploaded_file.name}' uploaded successfully!")
        
        # Placeholders for dynamic UI elements
        stop_button_placeholder = st.empty()
        progress_bar_placeholder = st.empty()

        if st.button("🔄 Process Document", type="primary", disabled=st.session_state.get('is_processing', False)):
            st.session_state.is_processing = True
            st.session_state.stop_processing = False
            # Store file content in session state to avoid re-reading if rerun happens for stop button
            if 'uploaded_file_content' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
                st.session_state.uploaded_file_content = uploaded_file.read()
                st.session_state.uploaded_file_name = uploaded_file.name
            st.rerun()

        if st.session_state.get('is_processing', False):
            # Display stop button
            with stop_button_placeholder.container():
                if st.button("⏹️ Stop Processing", key="stop_processing_button"):
                    st.session_state.stop_processing = True
                    st.warning("Stop signal sent. Processing will halt after the current chunk...")
            
            # Display and initialize progress bar
            current_progress_bar = progress_bar_placeholder.progress(0)

            def update_progress_callback(chunk_index, total_chunks):
                if total_chunks > 0:
                    progress_value = (chunk_index) / total_chunks # chunk_index is 1-based from processor
                    current_progress_bar.progress(progress_value)

            try:
                with st.spinner("Processing document... (this may take a while for large documents)"):
                    # Ensure uploaded_file_content is available
                    if 'uploaded_file_content' not in st.session_state or \
                       st.session_state.uploaded_file_name != uploaded_file.name:
                        # This case should ideally be prevented by the logic in "Process Document" button
                        st.error("File content not found in session. Please re-upload.")
                        st.session_state.is_processing = False
                        st.rerun()

                    # Add DEBUG logs here
                    st.write(f"DEBUG: Just before processing, shared_config.current_model is: {st.session_state.shared_quiz_config.current_model}")
                    if st.session_state.document_processor.contextgem_llm:
                        st.write(f"DEBUG: DP's ContextGem LLM model: {st.session_state.document_processor.contextgem_llm.model}")
                    else:
                        st.write("DEBUG: DP's ContextGem LLM is None.")

                    processed_content = st.session_state.document_processor.process(
                        st.session_state.uploaded_file_content, # Use stored content
                        "direct_input",
                        custom_filename=st.session_state.uploaded_file_name,
                        stop_signal_check=lambda: st.session_state.get('stop_processing', False),
                        chunk_processed_callback=update_progress_callback
                    )
                    
                    if st.session_state.stop_processing:
                        st.warning("Processing was stopped by the user.")
                        st.session_state.processed_content = None
                    else:
                        st.session_state.processed_content = processed_content
                        current_progress_bar.progress(1.0) # Mark as 100% if not stopped
                        if processed_content:
                            try:
                                doc_id = st.session_state.vector_manager.store_document(processed_content)
                                st.success(f"✅ Document processed and stored with embeddings! ID: {doc_id[:8]}...")
                            except Exception as e:
                                st.error(f"❌ Error storing document with embeddings: {str(e)}")
                                st.success("✅ Document processed successfully (without embeddings)!")
                        else:
                             st.info("No content was processed to store (possibly stopped early).")
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.info("💡 Make sure the document is not corrupted and is in a supported format.")
            finally:
                st.session_state.is_processing = False
                stop_button_placeholder.empty()
                progress_bar_placeholder.empty() # Clear progress bar after processing
                # Only rerun if explicitly needed, e.g., after a stop, to refresh UI state
                # If processing completed normally, new UI elements might appear, so rerun is often good.
                st.rerun()

    # Document Management Section
    st.header("🗃️ Document Management")
    
    # List stored documents from pgvector
    try:
        stored_docs = st.session_state.vector_manager.list_documents()
        if stored_docs:
            st.subheader("Stored Documents")
            
            # Create a dictionary to track selected documents
            if 'selected_docs' not in st.session_state:
                st.session_state.selected_docs = {}
            
            # Display documents with checkboxes and topic information
            for doc in stored_docs:
                doc_key = f"doc_{doc['id']}"
                
                # Display document with extracted info
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    is_selected = st.checkbox(
                        f"📄 {doc['filename']}",
                        value=st.session_state.selected_docs.get(doc_key, False),
                        key=doc_key
                    )
                    st.session_state.selected_docs[doc_key] = is_selected
                    
                    # Show document details
                    with st.expander(f"📋 Details for {doc['filename']}", expanded=False):
                        st.write(f"**Document ID:** {doc['id'][:12]}...")
                        st.write(f"**Format:** {doc['format'].upper()}")
                        st.write(f"**Created:** {doc['created_at'][:16] if doc['created_at'] else 'Unknown'}")
                        
                        # Show extraction summary
                        if 'extracted_summary' in doc:
                            summary = doc['extracted_summary']
                            st.write(f"**Main Topic:** {summary.get('main_topic', 'Not processed yet')}")
                            st.write(f"**Document Type:** {summary.get('document_type', 'Unknown')}")
                            st.write(f"**Concepts Extracted:** {summary.get('concept_count', 0)}")
                            st.write(f"**Status:** {summary.get('extraction_status', 'unknown').title()}")
                        else:
                            st.write("**Status:** No extraction data available")
                
                with col2:
                    # Show status indicator with improved logic to avoid duplicate Processing messages
                    if 'extracted_summary' in doc:
                        status = doc['extracted_summary'].get('extraction_status', 'unknown')
                        concept_count = doc['extracted_summary'].get('concept_count', 0)
                        main_topic = doc['extracted_summary'].get('main_topic', 'Unknown')
                        
                        # Simplified and more accurate status determination
                        if concept_count > 0 and main_topic not in ['Processing...', 'Not processed yet', 'Unknown', '', 'Processing failed', 'No content available']:
                            st.success("✅ Ready")
                        elif status == 'failed' or main_topic in ['Processing failed', 'No content available']:
                            st.error("❌ Failed")
                        elif status == 'pending' and main_topic in ['Processing...', 'Not processed yet']:
                            st.warning("⏳ Processing")
                        else:
                            # Default to Ready if we have any concepts, otherwise needs processing
                            if concept_count > 0:
                                st.success("✅ Ready")
                            else:
                                st.info("🔄 Needs processing")
                    else:
                        st.info("🔄 Needs processing")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Load Selected Documents", type="secondary", use_container_width=True):
                    selected_ids = []
                    for doc in stored_docs:
                        doc_key = f"doc_{doc['id']}"
                        if st.session_state.selected_docs.get(doc_key, False):
                            selected_ids.append(doc['id'])
                    
                    if not selected_ids:
                        st.warning("Please select at least one document")
                    else:
                        try:
                            all_langchain_documents = []
                            combined_content = ""
                            combined_metadata = {'chapters': [], 'sections': [], 'total_words': 0, 'total_paragraphs': 0}
                            
                            combined_extracted_concepts = {}
                            
                            for doc_id in selected_ids:
                                doc_data = st.session_state.vector_manager.get_document(doc_id)
                                if doc_data:
                                    # Reconstruct processed_content for display purposes
                                    combined_content += doc_data['content'] + "\n\n"
                                    
                                    # Update combined metadata (basic aggregation)
                                    combined_metadata['total_words'] += doc_data['metadata']['total_words']
                                    combined_metadata['total_paragraphs'] += doc_data['metadata']['total_paragraphs']
                                    
                                    # Properly combine extracted concepts from database (ContextGem format)
                                    if 'extracted_concepts' in doc_data and doc_data['extracted_concepts']:
                                        stored_concepts = doc_data['extracted_concepts']
                                        
                                        # Handle both dict (ContextGem) and list (document processor) formats
                                        if isinstance(stored_concepts, dict):
                                            # ContextGem format: merge concept categories
                                            for concept_name, concept_data in stored_concepts.items():
                                                if concept_name not in combined_extracted_concepts:
                                                    combined_extracted_concepts[concept_name] = {"items": []}
                                                
                                                # Add items, avoiding duplicates
                                                existing_values = {item.get("value", "") for item in combined_extracted_concepts[concept_name]["items"]}
                                                for item in concept_data.get("items", []):
                                                    if item.get("value") and item["value"] not in existing_values:
                                                        combined_extracted_concepts[concept_name]["items"].append(item)
                                        
                                        elif isinstance(stored_concepts, list):
                                            # Document processor format: convert to ContextGem format
                                            for concept in stored_concepts:
                                                if isinstance(concept, dict):
                                                    concept_name = concept.get('concept_name', 'Extracted Concepts')
                                                    concept_content = concept.get('content', '')
                                                    
                                                    if concept_content and concept_content.strip():
                                                        if concept_name not in combined_extracted_concepts:
                                                            combined_extracted_concepts[concept_name] = {"items": []}
                                                        
                                                        # Check for duplicates
                                                        existing_values = {item.get("value", "") for item in combined_extracted_concepts[concept_name]["items"]}
                                                        if concept_content not in existing_values:
                                                            combined_extracted_concepts[concept_name]["items"].append({
                                                                "value": concept_content,
                                                                "references": [],
                                                                "justification": concept.get("metadata", {}).get("extraction_method", "Loaded from database")
                                                            })
                                    
                            if combined_content.strip():
                                st.session_state.processed_content = {
                                    'content': combined_content.strip(),
                                    'segments': st.session_state.document_processor.intelligent_segmentation(combined_content.strip()),
                                    'metadata': combined_metadata,
                                    'filename': f"{len(selected_ids)} selected documents",
                                    'format': 'combined',
                                    'extracted_concepts': combined_extracted_concepts  # Properly formatted ContextGem concepts
                                }
                                st.success(f"Loaded {len(selected_ids)} documents")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error loading documents: {str(e)}")
            with col2:
                if st.button("Delete Selected Documents", type="secondary", use_container_width=True):
                    selected_ids = []
                    for doc in stored_docs:
                        doc_key = f"doc_{doc['id']}"
                        if st.session_state.selected_docs.get(doc_key, False):
                            selected_ids.append(doc['id'])
                    
                    if not selected_ids:
                        st.warning("Please select at least one document")
                    else:
                        try:
                            for doc_id in selected_ids:
                                if st.session_state.vector_manager.delete_document(doc_id):
                                    # Remove from selected state
                                    doc_key = f"doc_{doc_id}"
                                    if doc_key in st.session_state.selected_docs:
                                        del st.session_state.selected_docs[doc_key]
                            
                            # Clear processed_content if the selected documents were loaded
                            if st.session_state.processed_content:
                                # Check if the current processed document is one of the deleted ones
                                if "filename" in st.session_state.processed_content:
                                    if st.session_state.processed_content["filename"] == f"{len(selected_ids)} selected documents":
                                        st.session_state.processed_content = None
                                        if 'selected_concepts' in st.session_state:
                                            st.session_state.selected_concepts = []
                            
                            st.success(f"Deleted {len(selected_ids)} documents")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting document: {str(e)}")
            with col3:
                if st.button("Clear Selection", type="secondary", use_container_width=True):
                    for key in list(st.session_state.selected_docs.keys()):
                        st.session_state.selected_docs[key] = False
                    st.rerun()
        else:
            st.info("No documents stored yet.")
    except Exception as e:
        st.error(f"Error accessing document storage: {str(e)}")

    st.divider()

    # Display document summary and preview if content is processed
    if st.session_state.processed_content:
        with st.expander("📊 Document Summary", expanded=True):
            summary = st.session_state.document_processor.get_content_summary(st.session_state.processed_content)
            st.text(summary)
        
        # Display extracted concepts for selection - check both ContextGem and document processor concepts
        has_contextgem_concepts = 'extracted_concepts' in st.session_state.processed_content and st.session_state.processed_content['extracted_concepts']
        has_processor_concepts = 'concepts' in st.session_state.processed_content and st.session_state.processed_content['concepts']
        
        if has_contextgem_concepts or has_processor_concepts:
            with st.expander("🎯 Select Topics for Quiz", expanded=True):
                st.markdown("**Select the concepts you want to focus on for quiz generation:**")
                
                # Initialize selected concepts in session state
                if 'selected_concepts' not in st.session_state:
                    st.session_state.selected_concepts = []
                
                # Combine concepts from both sources
                concept_groups = {}
                
                # Add ContextGem concepts if available (including those loaded from database)
                if has_contextgem_concepts:
                    extracted_concepts = st.session_state.processed_content['extracted_concepts']
                    if isinstance(extracted_concepts, dict):
                        for concept_name, concept_data in extracted_concepts.items():
                            if isinstance(concept_data, dict) and 'items' in concept_data:
                                valid_concepts = [item.get('value', '').strip() for item in concept_data['items'] if item.get('value', '').strip()]
                                if valid_concepts:  # Only add if we have valid concepts
                                    concept_groups[concept_name] = valid_concepts
                    elif isinstance(extracted_concepts, list):
                        # Handle case where extracted_concepts is a list of concept objects
                        for i, concept in enumerate(extracted_concepts):
                            if isinstance(concept, dict):
                                concept_name = concept.get('concept_name', f'Concept {i+1}')
                                concept_content = concept.get('content', '').strip()
                                if concept_content:
                                    if concept_name not in concept_groups:
                                        concept_groups[concept_name] = []
                                    concept_groups[concept_name].append(concept_content)
                
                # Add document processor concepts if available
                if has_processor_concepts:
                    processor_concepts = st.session_state.processed_content['concepts']
                    for concept in processor_concepts:
                        concept_type = concept.get('concept_name', 'Extracted Concepts')
                        concept_content = concept.get('content', '')
                        
                        if concept_content and concept_content.strip():
                            if concept_type not in concept_groups:
                                concept_groups[concept_type] = []
                            concept_groups[concept_type].append(concept_content)
                
                # Display concept selection in columns
                if concept_groups:
                    # Select All / Deselect All buttons
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("Select All", key="select_all_concepts"):
                            all_concepts = []
                            for concepts_list in concept_groups.values():
                                all_concepts.extend(concepts_list)
                            st.session_state.selected_concepts = list(set(all_concepts))
                            st.rerun()
                    with col2:
                        if st.button("Deselect All", key="deselect_all_concepts"):
                            st.session_state.selected_concepts = []
                            st.rerun()
                    
                    st.markdown("---")
                    
                    # Display concepts by category
                    for concept_type, concepts_list in concept_groups.items():
                        if concepts_list:
                            st.markdown(f"**{concept_type}:**")
                            
                            # Create checkboxes for each concept - single column for better readability
                            for i, concept in enumerate(concepts_list):
                                if concept.strip():  # Only show non-empty concepts
                                    is_selected = concept in st.session_state.selected_concepts
                                    # Show full concept text without truncation
                                    if st.checkbox(concept, value=is_selected, key=f"concept_{concept_type}_{i}_{concept[:20]}"):
                                        if concept not in st.session_state.selected_concepts:
                                            st.session_state.selected_concepts.append(concept)
                                    else:
                                        if concept in st.session_state.selected_concepts:
                                            st.session_state.selected_concepts.remove(concept)
                            
                            st.markdown("")  # Add spacing
                    
                    # Show selected concepts summary
                    if st.session_state.selected_concepts:
                        st.success(f"✅ Selected {len(st.session_state.selected_concepts)} concepts for quiz focus")
                        # Replace nested expander with a container to avoid the nesting error
                        concepts_container = st.container()
                        with concepts_container:
                            st.markdown("**📝 Selected Concepts:**")
                            for concept in st.session_state.selected_concepts:
                                st.write(f"• {concept}")
                    else:
                        st.info("💡 No concepts selected - quiz will cover all topics")
                else:
                    st.info("No specific concepts extracted from this document")
                    
                    # Add manual concept extraction button
                    if st.button("🔄 Extract Topics Manually", type="secondary"):
                        try:
                            with st.spinner("Extracting topics from document..."):
                                # Get content from processed_content
                                content = st.session_state.processed_content['content']
                                filename = st.session_state.processed_content['filename']
                                
                                # Try to extract concepts using document processor
                                concepts = st.session_state.document_processor.extract_concepts(
                                    content, filename, st.session_state.processed_content['format']
                                )
                                
                                if concepts:
                                    # Add concepts to processed_content
                                    st.session_state.processed_content['concepts'] = concepts
                                    
                                    # IMPORTANT: Save concepts to database for persistence
                                    try:
                                        # Find the document ID by content hash
                                        import hashlib
                                        doc_id = hashlib.sha256(content.encode()).hexdigest()
                                        
                                        # Convert document processor concepts to ContextGem format for storage
                                        formatted_concepts = {}
                                        for concept in concepts:
                                            concept_name = concept.get('concept_name', 'Extracted Concepts')
                                            concept_content = concept.get('content', '')
                                            
                                            if concept_content.strip():
                                                if concept_name not in formatted_concepts:
                                                    formatted_concepts[concept_name] = {"items": []}
                                                
                                                formatted_concepts[concept_name]["items"].append({
                                                    "value": concept_content,
                                                    "references": [],
                                                    "justification": "Extracted by document processor"
                                                })
                                        
                                        # Update database with new concepts
                                        if formatted_concepts:
                                            import json
                                            with st.session_state.vector_manager.db._get_connection() as conn:
                                                with conn.cursor() as cursor:
                                                    cursor.execute('''
                                                        UPDATE documents_enhanced 
                                                        SET extracted_concepts = %s, updated_at = CURRENT_TIMESTAMP
                                                        WHERE id = %s
                                                    ''', (json.dumps(formatted_concepts), doc_id))
                                                conn.commit()
                                            
                                            st.info("💾 Concepts saved to database for persistence")
                                        
                                    except Exception as save_error:
                                        st.warning(f"Concepts extracted but couldn't save to database: {str(save_error)}")
                                    
                                    st.success(f"✅ Successfully extracted {len(concepts)} concepts!")
                                    st.rerun()
                                else:
                                    st.warning("No concepts could be extracted. Try checking your OpenAI API key or model configuration.")
                        except Exception as e:
                            st.error(f"Error extracting concepts: {str(e)}")
                            st.info("💡 Make sure your OpenAI API key is configured and the model is available.")
        else:
            st.info("💡 Upload a document to see extracted topics for quiz focus")
            
        with st.expander("👀 Content Preview"):
            content_preview_text = st.session_state.processed_content['content']
            
            # Check if content looks like binary PDF data
            is_binary_pdf = False
            if content_preview_text.startswith("%PDF-") and re.search(r'stream|endobj|xref|startxref', content_preview_text[:1000]):
                is_binary_pdf = True
            
            if is_binary_pdf:
                st.warning("⚠️ This PDF appears to contain primarily binary data or images rather than extractable text.")
                st.info("The system will attempt to extract concepts based on any available text content. If you need better results, try uploading a text-based PDF.")
                
                # Try to find any actual text in the document
                text_parts = re.findall(r'[A-Za-z]{3,}[\s.,;:!?][A-Za-z\s.,;:!?]{20,}', content_preview_text)
                if text_parts:
                    st.subheader("Extracted Text Fragments:")
                    extracted_text = " ".join(text_parts[:5])  # Show first 5 text fragments
                    st.text_area("Extracted text samples:", extracted_text[:500], height=100)
                else:
                    st.error("No meaningful text could be extracted from this PDF. Consider trying a different document.")
            else:
                # Normal text preview
                display_preview = content_preview_text[:500] + "..." if len(content_preview_text) > 500 else content_preview_text
                st.text_area("First 500 characters:", display_preview, height=150)

    st.divider()

    # Section 2: Quiz Generation
    if st.session_state.processed_content:
        st.header("✨ Ready to Generate Your Quiz?")
        
        # Get quiz settings from sidebar (ensure these are defined in the sidebar section)
        # These variables (num_questions, difficulty, question_types, focus_section)
        # are assumed to be available from the sidebar's scope.
        # If not, they need to be accessed via st.session_state if stored there by sidebar,
        # or passed around if sidebar logic is refactored.
        # For this diff, we assume they are available as before.

        if not question_types: # question_types is from sidebar st.multiselect
            st.warning("⚠️ Please select at least one question type from the sidebar settings.")
        else:
            if st.button("🚀 Generate Quiz", type="primary", disabled=not question_types):
                try:
                    with st.spinner("Generating quiz questions..."):
                        gen_progress_bar = st.progress(0)
                        
                        gen_progress_bar.progress(10)
                        
                        # Ensure the model selected in the sidebar (quiz_model) is applied everywhere
                        selected_model_from_sidebar = quiz_model # quiz_model is from the sidebar's selected_option_obj
                        
                        # Update the shared config via LLMManager
                        st.session_state.quiz_generator.llm_manager.set_model(selected_model_from_sidebar)
                        
                        # Tell DocumentProcessor and VectorManager to update their internal ContextGem LLMs
                        # based on the now-updated shared config.
                        if hasattr(st.session_state.document_processor, 'update_llm_configuration'):
                            st.session_state.document_processor.update_llm_configuration()
                        
                        if hasattr(st.session_state.vector_manager, 'update_llm_configuration'):
                            st.session_state.vector_manager.update_llm_configuration()

                        # For quiz generation itself, LLMManager (via QuizGenerator) will use the model from shared_config.
                        # For connection checks, use the model that LLMManager reports it's using.
                        current_model_for_quiz_gen = st.session_state.quiz_generator.llm_manager.get_current_model()
                        st.info(f"Using model for quiz generation: {current_model_for_quiz_gen}") # Changed to st.info
                        
                        # Check connection based on model type
                        is_openai_model = current_model_for_quiz_gen in st.session_state.quiz_generator.llm_manager.get_openai_models()
                        
                        if is_openai_model:
                            # For OpenAI models, check API key
                            if not os.environ.get("OPENAI_API_KEY"):
                                st.error("❌ OpenAI API key not found. Please add it to your .env file.")
                                st.stop()
                        else:
                            # For Ollama models, test connection
                            if not st.session_state.quiz_generator.test_ollama_connection():
                                st.error("❌ Cannot connect to Ollama. Please make sure it's running.")
                                st.stop()
                        
                        gen_progress_bar.progress(30)
                        
                        # Prepare focus parameters using selected concepts
                        focus_topics = []
                        
                        # Add selected concepts if any
                        if hasattr(st.session_state, 'selected_concepts') and st.session_state.selected_concepts:
                            focus_topics.extend(st.session_state.selected_concepts)
                            st.info(f"🎯 Focusing quiz on {len(st.session_state.selected_concepts)} selected concepts")
                        
                        # focus_topics is already a List[str] of selected concept contents
                        # combined_focus = "; ".join(focus_topics) if focus_topics else None # Old way
                        
                        # The generate_quiz method will be updated to accept List[str] for focus
                        quiz_data = st.session_state.quiz_generator.generate_quiz( # Corrected method name
                            st.session_state.processed_content,
                            question_types, # from sidebar
                            num_questions,  # from sidebar
                            difficulty,     # from sidebar
                            focus_topics # Pass the list of selected concept strings
                        )
                        gen_progress_bar.progress(90); time.sleep(0.1)
                        gen_progress_bar.progress(100)
                        
                        quiz_data['config_difficulty'] = difficulty # Store selected difficulty
                        st.session_state.quiz_data = quiz_data
                        
                        st.session_state.current_question = 0
                        st.session_state.user_answers = {}
                        st.session_state.show_results = False
                        
                    # Check if questions were generated
                    if len(quiz_data['questions']) > 0:
                        st.success(f"✅ Generated {len(quiz_data['questions'])} questions!")
                        
                        # Automatically save the quiz with default metadata
                        try:
                            db = DatabaseManager()
                            default_title = f"{uploaded_file.name if uploaded_file else 'Generated'} Quiz - {time.strftime('%Y-%m-%d %H:%M')}"
                            default_description = f"A {difficulty} difficulty quiz with {num_questions} {', '.join(question_types)} questions"
                            default_tags = [difficulty.lower()] + [qt.lower().replace('-', '') for qt in question_types]
                            
                            # Always store extracted topics and concepts in the quiz_data
                            quiz_data['extracted_topics'] = {
                                'selected_concepts': st.session_state.get('selected_concepts', []),
                                'concept_count': len(st.session_state.get('selected_concepts', [])),
                                'extraction_method': 'user_selected'
                            }
                            
                            # Also store document metadata for reference
                            if st.session_state.processed_content:
                                quiz_data['document_info'] = {
                                    'filename': st.session_state.processed_content.get('filename', 'Unknown'),
                                    'format': st.session_state.processed_content.get('format', 'Unknown'),
                                    'total_words': st.session_state.processed_content.get('metadata', {}).get('total_words', 0),
                                    'total_paragraphs': st.session_state.processed_content.get('metadata', {}).get('total_paragraphs', 0)
                                }
                                
                                # Store all available concepts (not just selected ones)
                                all_concepts = []
                                if 'extracted_concepts' in st.session_state.processed_content:
                                    extracted_concepts = st.session_state.processed_content['extracted_concepts']
                                    if isinstance(extracted_concepts, dict):
                                        for concept_name, concept_data in extracted_concepts.items():
                                            if isinstance(concept_data, dict) and 'items' in concept_data:
                                                for item in concept_data['items']:
                                                    if isinstance(item, dict) and item.get('value'):
                                                        all_concepts.append({
                                                            'category': concept_name,
                                                            'content': item['value'],
                                                            'source': 'contextgem'
                                                        })
                                    elif isinstance(extracted_concepts, list):
                                        for concept in extracted_concepts:
                                            if isinstance(concept, dict) and concept.get('content'):
                                                all_concepts.append({
                                                    'category': concept.get('concept_name', 'General'),
                                                    'content': concept['content'],
                                                    'source': 'contextgem'
                                                })
                                
                                if 'concepts' in st.session_state.processed_content:
                                    for concept in st.session_state.processed_content['concepts']:
                                        if isinstance(concept, dict) and concept.get('content'):
                                            all_concepts.append({
                                                'category': concept.get('concept_name', 'General'),
                                                'content': concept['content'],
                                                'source': 'document_processor'
                                            })
                                
                                quiz_data['extracted_topics']['all_available_concepts'] = all_concepts
                            
                            quiz_id = db.save_quiz(
                                title=default_title,
                                quiz_data=quiz_data,
                                description=default_description,
                                tags=default_tags
                            )
                            
                            if quiz_id:
                                st.success(f"✅ Quiz automatically saved with ID: {quiz_id}")
                        except Exception as e:
                            st.error(f"Error automatically saving quiz: {str(e)}")
                        
                        # Option to save custom version
                        with st.expander("💾 Save Custom Version", expanded=False):
                            quiz_title = st.text_input("Custom Title", value=default_title)
                            quiz_description = st.text_area("Custom Description", value=default_description)
                            quiz_tags = st.text_input("Custom Tags (comma-separated)", value=', '.join(default_tags))
                            
                            if st.button("Save Custom Version", type="primary"):
                                try:
                                    tags_list = [tag.strip() for tag in quiz_tags.split(',')] if quiz_tags else default_tags
                                    
                                    # Make sure selected concepts are stored in the quiz_data for custom save as well
                                    if hasattr(st.session_state, 'selected_concepts') and st.session_state.selected_concepts:
                                        if 'selected_concepts' not in quiz_data:
                                            quiz_data['selected_concepts'] = st.session_state.selected_concepts
                                    
                                    custom_quiz_id = db.save_quiz(
                                        title=quiz_title,
                                        quiz_data=quiz_data,
                                        description=quiz_description,
                                        tags=tags_list
                                    )
                                    
                                    if custom_quiz_id:
                                        st.success(f"✅ Custom version saved with ID: {custom_quiz_id}")
                                    else:
                                        st.error("Failed to save custom version")
                                except Exception as e:
                                    st.error(f"Error saving custom version: {str(e)}")
                        
                        # Create a more prominent button for the quiz page
                        st.markdown("### Ready to take the quiz?")
                        
                        # Use columns to center the button and make it larger
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.page_link(
                                "pages/01_Interactive_Quiz.py",
                                label="🎓 Go to Interactive Quiz",
                                icon="➡️",
                                use_container_width=True
                            )
                            
                            # Add custom CSS to make the button much bigger and more prominent
                            st.markdown("""
                            <style>
                            .stPageLink {
                                font-size: 2.5rem !important;
                                font-weight: bold !important;
                                padding: 2rem 3rem !important;
                                text-align: center !important;
                                margin: 2rem 0 !important;
                                background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
                                color: white !important;
                                border-radius: 15px !important;
                                box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
                                text-decoration: none !important;
                                display: block !important;
                                transform: scale(1.1) !important;
                                transition: all 0.3s ease !important;
                            }
                            .stPageLink:hover {
                                transform: scale(1.15) !important;
                                box-shadow: 0 12px 24px rgba(0,0,0,0.3) !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No questions were generated. Please try a different model or content.")
                        
                    # Show generation info
                    with st.expander("ℹ️ Generation Info", expanded=True):
                        st.write(f"**Model Used:** {quiz_data['model_used']}")
                        st.write(f"**Content Length:** {quiz_data['content_length']} characters")
                        st.write(f"**Questions Generated:** {len(quiz_data['questions'])}")
                        st.write(f"**Selected Difficulty:** {difficulty}")
                        
                        # Show fallback info if used
                        if quiz_data.get('fallback_used', False):
                            st.warning(f"⚠️ Fallback to OpenAI was used because {quiz_data.get('original_model', 'the selected model')} failed to generate questions.")
                            st.info("To use local models successfully, make sure Ollama is running and the model is properly installed.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating quiz: {str(e)}")
                    st.info("💡 Try reducing the number of questions or check your Ollama connection.")
    else:
        st.info("☝️ Please upload and process a document above to enable quiz generation.")

if __name__ == "__main__":
    main()
