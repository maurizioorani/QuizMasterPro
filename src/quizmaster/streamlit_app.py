import streamlit as st
from document_processor import DocumentProcessor
from quiz_generator import QuizGenerator
import os
import time

def main():
    st.set_page_config(
        page_title="QuizMaster Pro", 
        page_icon="📚", 
        layout="wide"
    )
    
    st.title("📚 QuizMaster Pro - Document to Quiz Converter")
    st.markdown("Transform any document into an interactive quiz using local AI")

    # Initialize processors
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'quiz_generator' not in st.session_state:
        st.session_state.quiz_generator = QuizGenerator()
    
    if 'processed_content' not in st.session_state:
        st.session_state.processed_content = None
    
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None


    # Sidebar for settings and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        st.subheader("🤖 AI Model")

        MODEL_DESCRIPTIONS = {
            "llama3.3:8b": "Meta's Llama 3.3: Latest generation, 8B parameters, strong all-rounder.",
            "llama3.3:70b": "Meta's Llama 3.3: Latest generation, 70B parameters, for highly complex tasks.",
            "gemma3:9b": "Google's Gemma 3: New generation model, 9B parameters, good for coding & reasoning.",
            "phi4:latest": "Microsoft's Phi-4: Efficient and capable smaller language model.",
            "deepseek-r1:latest": "DeepSeek-R1: Strong model from DeepSeek AI, particularly for coding.",
            "granite3.3:latest": "IBM's Granite 3.3: Enterprise-focused model for various tasks.",
            "llama3.1:8b": "Meta's Llama 3.1: Previous generation, 8B parameters, balanced performance.",
            "mistral:7b": "⭐ RECOMMENDED: Mistral-7B - Excellent JSON generation, efficient (4.1GB), reliable structured output.",
            "qwen2.5:7b": "⭐ RECOMMENDED: Qwen 2.5 - Strong JSON/structured output, multilingual, efficient (4.4GB).",
            "deepseek-coder:6.7b": "🔧 JSON SPECIALIST: DeepSeek Coder - Code-focused, excellent structured formats (3.8GB).",
            "codellama:7b": "🔧 JSON SPECIALIST: CodeLlama - Meta's coding model, strong with structured data (3.8GB).",
            "gemma2:9b": "⭐ RECOMMENDED: Gemma2-9B - Google's instruction-following model, good JSON reliability (5.4GB)."
            # Add more descriptions as new models are verified and added
        }

        available_models = st.session_state.quiz_generator.get_available_models()
        current_model = st.session_state.quiz_generator.get_current_model()
        
        # Get locally available models
        local_models = st.session_state.quiz_generator.get_local_models()
        
        # Create display options with status indicators and add default option
        DEFAULT_PROMPT = "Please select a model first"
        model_options = [DEFAULT_PROMPT]
        for model in available_models:
            if model in local_models:
                model_options.append(f"✅ {model} (Local)")
            else:
                model_options.append(f"📥 {model} (Pull needed)")
        
        # Always default to the "Please select a model first" option
        initial_index = 0
        
        selected_display = st.selectbox(
            "Select Ollama Model",
            model_options,
            index=initial_index,
            help="✅ = Available locally, 📥 = Will be downloaded automatically"
        )
        
        # Track if a valid model is selected
        is_model_selected = selected_display != DEFAULT_PROMPT
        
        # Extract model name only if not default prompt
        if selected_display != DEFAULT_PROMPT:
            selected_model = selected_display.split(" ")[1]
            
            # Display model description
            model_desc = MODEL_DESCRIPTIONS.get(selected_model, "No description available for this model.")
            st.caption(model_desc)
            
            if selected_model != current_model:
                with st.spinner(f"Setting up model {selected_model}..."):
                    try:
                        # Check if model needs to be pulled
                        if not st.session_state.quiz_generator.is_model_available(selected_model):
                            st.info(f"📥 Model {selected_model} not found locally. Downloading...")
                        
                        # Create progress placeholder
                        progress_placeholder = st.empty()
                        # Initial text like "Pulling model..." will be set by pull_model itself.
                        
                        # Pull the model, passing the placeholder
                        success = st.session_state.quiz_generator.pull_model(selected_model, progress_placeholder=progress_placeholder)
                        
                        if success:
                            # Success message on placeholder is handled by pull_model
                            st.session_state.quiz_generator.set_model(selected_model) # Use set_model to initialize LLM/Embeddings
                            st.session_state.document_processor.set_model(selected_model) # Set model for document processing too
                            st.success(f"Model changed to {selected_model}") # General success for model change
                        else:
                            # Error message on placeholder is handled by pull_model
                            # Display a general error message for the Streamlit app UI
                            st.error(f"Failed to set up model {selected_model}. Please check logs or Ollama status.")
                        
                    except Exception as e:
                        st.error(f"Error setting model: {str(e)}")
        
        # Test Connection (disabled when no model selected)
        if st.button("🔗 Test Ollama Connection", disabled=not is_model_selected):
            with st.spinner("Testing connection..."):
                if st.session_state.quiz_generator.test_ollama_connection():
                    st.success("✅ Ollama connection successful!")
                else:
                    st.error("❌ Ollama connection failed. Make sure Ollama is running on localhost:11434")
        
        st.divider()
        
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
        
        focus_section = st.text_input(
            "Focus Section (optional)",
            help="Enter keywords to focus on specific sections"
        )
        
        st.divider()
        
        # Concept Extraction Settings
        st.subheader("🧠 Concept Extraction")
        use_concept_extraction = st.checkbox(
            "Enable Concept Extraction",
            value=True,
            help="Extract key concepts to improve quiz quality. Uses robust fallback method without JSON requirements."
        )
        
        if use_concept_extraction:
            st.caption("📚 Will extract Key Definitions, Important Facts, and Main Ideas using your selected model")
            # Update document processor settings
            if hasattr(st.session_state, 'document_processor'):
                st.session_state.document_processor.fallback_extraction_enabled = True
        else:
            st.caption("⚡ Faster processing without concept extraction")
            # Update document processor settings
            if hasattr(st.session_state, 'document_processor'):
                st.session_state.document_processor.fallback_extraction_enabled = False
        
        st.divider()
        
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
        
        if st.button("🔄 Process Document", type="primary"):
            try:
                with st.spinner("Processing document..."):
                    progress_bar = st.progress(0)
                    # Simulating steps for demo
                    progress_bar.progress(25); time.sleep(0.1)
                    processed_content = st.session_state.document_processor.process(
                        uploaded_file.read(),
                        uploaded_file.name
                    )
                    progress_bar.progress(75); time.sleep(0.1)
                    st.session_state.processed_content = processed_content
                    progress_bar.progress(100)
                    
                    # Store the processed document as JSON file
                    try:
                        doc_id = st.session_state.document_processor.store_document(processed_content)
                        st.success(f"✅ Document processed and stored successfully! ID: {doc_id[:8]}...")
                    except Exception as e:
                        st.error(f"❌ Error storing document: {str(e)}")
                        st.success("✅ Document processed successfully!")
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.info("💡 Make sure the document is not corrupted and is in a supported format.")

    # Document Management Section
    st.header("🗃️ Document Management")
    
    # List stored documents
    try:
        stored_docs = st.session_state.document_processor.list_documents()
        if stored_docs:
            st.subheader("Stored Documents")
            
            # Create a dictionary to track selected documents
            if 'selected_docs' not in st.session_state:
                st.session_state.selected_docs = {}
            
            # Display documents with checkboxes
            for doc in stored_docs:
                doc_key = f"doc_{doc['id']}"
                is_selected = st.checkbox(
                    f"{doc['filename']} (ID: {doc['id']})",
                    value=st.session_state.selected_docs.get(doc_key, False),
                    key=doc_key
                )
                st.session_state.selected_docs[doc_key] = is_selected
            
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
                            
                            for doc_id in selected_ids:
                                doc_data = st.session_state.document_processor.get_document(doc_id)
                                if doc_data:
                                    # Reconstruct processed_content for display purposes
                                    combined_content += doc_data['content'] + "\n\n"
                                    
                                    # Update combined metadata (basic aggregation)
                                    combined_metadata['total_words'] += doc_data['metadata']['total_words']
                                    combined_metadata['total_paragraphs'] += doc_data['metadata']['total_paragraphs']
                                    
                                    # Add extracted concepts if available
                                    if 'extracted_concepts' in doc_data:
                                        if 'all_extracted_concepts' not in combined_metadata:
                                            combined_metadata['all_extracted_concepts'] = []
                                        combined_metadata['all_extracted_concepts'].extend(doc_data['extracted_concepts'])
                                    
                            if combined_content.strip():
                                st.session_state.processed_content = {
                                    'content': combined_content.strip(),
                                    'segments': st.session_state.document_processor.intelligent_segmentation(combined_content.strip()),
                                    'metadata': combined_metadata,
                                    'filename': f"{len(selected_ids)} selected documents",
                                    'format': 'combined',
                                    'extracted_concepts': combined_metadata.get('all_extracted_concepts', [])
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
                                if st.session_state.document_processor.delete_document(doc_id):
                                    # Remove from selected state
                                    doc_key = f"doc_{doc_id}"
                                    if doc_key in st.session_state.selected_docs:
                                        del st.session_state.selected_docs[doc_key]
                            
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
            
        with st.expander("👀 Content Preview"):
            content_preview_text = st.session_state.processed_content['content']
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
            if st.button("🚀 Generate Quiz", type="primary", disabled=not is_model_selected or not question_types):
                try:
                    with st.spinner("Generating quiz questions..."):
                        gen_progress_bar = st.progress(0)
                        
                        gen_progress_bar.progress(10)
                        if not st.session_state.quiz_generator.test_ollama_connection():
                            st.error("❌ Cannot connect to Ollama. Please make sure it's running.")
                            st.stop()
                        
                        gen_progress_bar.progress(30)
                        quiz_data = st.session_state.quiz_generator.generate(
                            st.session_state.processed_content,
                            question_types, # from sidebar
                            num_questions,  # from sidebar
                            difficulty,     # from sidebar
                            focus_section if focus_section else None # from sidebar
                        )
                        gen_progress_bar.progress(90); time.sleep(0.1)
                        gen_progress_bar.progress(100)
                        
                        quiz_data['config_difficulty'] = difficulty # Store selected difficulty
                        st.session_state.quiz_data = quiz_data
                        
                        st.session_state.current_question = 0
                        st.session_state.user_answers = {}
                        st.session_state.show_results = False
                        
                    st.success(f"✅ Generated {len(quiz_data['questions'])} questions!")
                    st.page_link("pages/01_Interactive_Quiz.py", label="🎓 Go to Interactive Quiz", icon="➡️")
                    
                    with st.expander("ℹ️ Generation Info"):
                        st.write(f"**Model Used:** {quiz_data['model_used']}")
                        st.write(f"**Content Length:** {quiz_data['content_length']} characters")
                        st.write(f"**Questions Generated:** {len(quiz_data['questions'])}")
                        st.write(f"**Selected Difficulty:** {difficulty}")
                        
                except Exception as e:
                    st.error(f"❌ Error generating quiz: {str(e)}")
                    st.info("💡 Try reducing the number of questions or check your Ollama connection.")
    else:
        st.info("☝️ Please upload and process a document above to enable quiz generation.")

if __name__ == "__main__":
    main()
