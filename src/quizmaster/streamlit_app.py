import streamlit as st
from document_processor import DocumentProcessor
from quiz_generator import QuizGenerator
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    
    # Always re-initialize QuizGenerator to pick up code changes
    st.session_state.quiz_generator = QuizGenerator()
    
    if 'processed_content' not in st.session_state:
        st.session_state.processed_content = None
    
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None


    # Sidebar for settings and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Separate Model Selection for Document Processing (Embedding)
        st.subheader("🤖 Embedding Model (Document Processing)")
        st.caption("For extracting concepts and embedding documents. OpenAI models recommended.")

        MODEL_DESCRIPTIONS = {
            "gpt-4.1-nano": "⚡ OPENAI: GPT-4.1 Nano - Lightweight but powerful model optimized for efficiency",
            "gpt-4o-mini": "⚡ OPENAI: GPT-4O Mini - Specialized for structured output and JSON generation",
            # Include local models as fallback
            "llama3.3:8b": "Meta's Llama 3.3: 8B parameters (fallback)",
            "mistral:7b": "Mistral-7B - Efficient but not recommended for embedding"
        }

        # Get current embedding model
        current_embedding_model = st.session_state.document_processor.get_current_model()
        
        # Enhanced embedding model selection with status indicators
        embedding_model_options = []
        
        # Only add OpenAI models for embedding
        for model in ["gpt-4.1-nano", "gpt-4o-mini"]:
            status = "⚡ API"
            if not os.environ.get("OPENAI_API_KEY"):
                status += " (key missing)"
            embedding_model_options.append({
                "name": model,
                "display": f"{model} - {status}",
                "description": MODEL_DESCRIPTIONS.get(model, "No description")
            })
        
        # Create selectbox with custom formatting
        selected_option = st.selectbox(
            "Select Embedding Model",
            embedding_model_options,
            format_func=lambda x: x["display"],
            index=next((i for i, m in enumerate(embedding_model_options) if m["name"] == current_embedding_model), 0),
            key="embedding_model_select"
        )
        embedding_model = selected_option["name"]
        
        # Show model description
        st.caption(selected_option["description"])
        
        if st.button("Apply Embedding Model", key="apply_embedding_model"):
            if embedding_model.startswith('gpt-') and not os.environ.get("OPENAI_API_KEY"):
                st.error("OpenAI API key not found in .env file")
            else:
                st.session_state.document_processor.set_model(embedding_model)
                st.success(f"Embedding model set to {embedding_model}")
        
        st.divider()
        
        # Separate Model Selection for Quiz Generation
        st.subheader("🤖 Quiz Generation Model")
        st.caption("For generating quiz questions. Local models recommended.")

        # Enhanced model descriptions with clearer status indicators
        QUIZ_MODEL_DESCRIPTIONS = {
            "llama3.3:8b": "⭐ RECOMMENDED: Meta's Llama 3.3, 8B parameters",
            "mistral:7b": "⭐ RECOMMENDED: Efficient JSON generation (4.1GB)",
            "qwen2.5:7b": "⭐ RECOMMENDED: Strong JSON/structured output, multilingual",
            "gpt-4.1-nano": "⚡ OPENAI: GPT-4.1 Nano - Requires API key",
            "gpt-4o-mini": "⚡ OPENAI: GPT-4O Mini - Requires API key"
        }
        
        # Get current model and local availability
        current_quiz_model = st.session_state.quiz_generator.get_current_model()
        local_models = st.session_state.quiz_generator.get_local_models()
        
        # Create model display options with status indicators
        quiz_model_options = []
        for model in st.session_state.quiz_generator.get_available_models():
            if model.startswith('gpt-'):
                # OpenAI models
                status = "⚡ API"
                if not os.environ.get("OPENAI_API_KEY"):
                    status += " (API key missing)"
            else:
                # Ollama models
                if model in local_models:
                    status = "✅ Local"
                else:
                    status = "📥 To be pulled"
                
            quiz_model_options.append({
                "name": model,
                "display": f"{model} - {status}",
                "description": QUIZ_MODEL_DESCRIPTIONS.get(model, "No description")
            })
        
        # Create selectbox with custom formatting
        selected_option = st.selectbox(
            "Select Quiz Model",
            quiz_model_options,
            format_func=lambda x: x["display"],
            index=next((i for i, m in enumerate(quiz_model_options) if m["name"] == current_quiz_model), 0),
            key="quiz_model_select"
        )
        quiz_model = selected_option["name"]
        
        # Show model description
        st.caption(selected_option["description"])
        
        # Check if current model is an OpenAI or Ollama model
        is_openai_model = quiz_model in st.session_state.quiz_generator.openai_models
        
        # Use different layouts based on model type
        if is_openai_model:
            # For OpenAI models, just show the Apply button
            if st.button("Apply Quiz Model", key="apply_quiz_model", use_container_width=True):
                if not os.environ.get("OPENAI_API_KEY"):
                    st.error("OpenAI API key not found in .env file")
                else:
                    with st.spinner(f"Setting up quiz model {quiz_model}..."):
                        try:
                            st.session_state.quiz_generator.set_model(quiz_model)
                            st.success(f"Quiz model set to {quiz_model}")
                        except Exception as e:
                            st.error(f"Error setting quiz model: {str(e)}")
        else:
            # For Ollama models, show Apply and Test buttons side by side
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Apply Quiz Model", key="apply_quiz_model", use_container_width=True):
                    with st.spinner(f"Setting up quiz model {quiz_model}..."):
                        try:
                            # For local models, check if available
                            if quiz_model not in local_models:
                                with st.status(f"📥 Downloading {quiz_model}...", expanded=True) as status:
                                    status.write("Starting model download...")
                                    success = st.session_state.quiz_generator.pull_model(quiz_model)
                                    if success:
                                        status.update(label="✅ Download completed!", state="complete", expanded=False)
                                    else:
                                        status.update(label="❌ Download failed", state="error")
                                        st.error(f"Failed to download {quiz_model}")
                                        st.stop()
                            
                            st.session_state.quiz_generator.set_model(quiz_model)
                            st.success(f"Quiz model set to {quiz_model}")
                        except Exception as e:
                            st.error(f"Error setting quiz model: {str(e)}")
            
            with col2:
                # Test Connection for Quiz Model
                if st.button("🔗 Test Ollama", key="test_quiz_connection", use_container_width=True):
                    with st.spinner("Testing connection..."):
                        if st.session_state.quiz_generator.test_ollama_connection():
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
        
        focus_section = st.text_input(
            "Focus Section (optional)",
            help="Enter keywords to focus on specific sections"
        )
        
        # Removed unused "Enable Concept Extraction" checkbox
        # Concept extraction is always enabled in the DocumentProcessor
        
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
            if st.button("🚀 Generate Quiz", type="primary", disabled=not question_types):
                try:
                    with st.spinner("Generating quiz questions..."):
                        gen_progress_bar = st.progress(0)
                        
                        gen_progress_bar.progress(10)
                        
                        # Check connection based on model type
                        current_model = st.session_state.quiz_generator.get_current_model()
                        is_openai_model = current_model in st.session_state.quiz_generator.openai_models
                        
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
