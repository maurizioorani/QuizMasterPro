# 📚 QuizMaster Pro

Are you struggling to memorize concepts for an exam or do you simply want to improve in a professional topic? Then simply upload your document to QuizMaster Pro and let the AI create a custom quiz in order to test and monitor your learning. QuizMaster Pro uses OpenAI models for document processing and offers both OpenAI and local Ollama models for quiz generation from PDF, DOCX, TXT, and HTML documents.

> **📢 Important Update**: After migrating to ContextGem for improved document processing, we found that local models struggled with JSON creation during the embedding process. To ensure more robust document processing, embedding is now only possible with OpenAI models, while quiz generation can still be performed successfully with local models (for example, Mistral 7b).

![QuizMaster Pro Screenshot 1](screenshots/quiz1.png) ![QuizMaster Pro Screenshot 2](screenshots/quiz2.png)

## 🌟 Features

### Document Processing
- **Multi-format Support**: PDF, DOCX, TXT, HTML files
- **Intelligent Segmentation**: Automatically identifies chapters, sections, and content structure
- **Advanced Concept Extraction**: Robust AI-powered extraction of Key Definitions, Important Facts, and Main Ideas
- **OpenAI-Powered Embedding**: Uses OpenAI models for reliable document processing and embedding
- **ContextGem Integration**: Enhanced document understanding with automatic fallback to basic segmentation
- **Metadata Extraction**: Extracts document structure and statistics
- **Content Preview**: Review processed content before quiz generation

### 🗃️ Document Management
- **Persistent Storage**: Documents are automatically stored using ChromaDB for future use.
- **Document Listing**: View all previously processed and stored documents.
- **Load Previous Documents**: Easily load any stored document to regenerate quizzes or review content.
- **Individual Deletion**: Delete documents from storage when no longer needed.
- **Metadata Preservation**: All original document metadata is retained with the stored content.

### AI-Powered Quiz Generation
- **Flexible Model Selection**: Choose between OpenAI models (requires API key) or local Ollama models for quiz generation
- **Enhanced Question Relevance**: Questions are generated from specific, relevant text segments, avoiding irrelevant introductory or summary content.
- **Conceptual Questioning**: Prompts are refined to encourage questions about concepts, principles, and implications, rather than direct textual recall.
- **Natural Question Phrasing**: Questions avoid phrases like "According to the text segment" for a more natural quiz experience.
- **Automatic Model Management**: Smart model detection and automatic downloading
- **Multiple Question Types**: 
  - Multiple Choice Questions (MCQ) with plausible distractors
  - Open-Ended Questions
  - True/False Statements
  - Fill-in-the-Blank Questions
- **Difficulty Levels**: Easy, Medium, Hard
- **Contextual Focus**: Generate questions from specific sections
- **Detailed Explanations**: Each answer includes comprehensive explanations with text references

### Interactive Quiz Experience
- **Modern UI**: Clean, responsive Streamlit interface
- **Question Navigation**: Move between questions with progress tracking
- **Immediate Feedback**: Instant answer validation and explanations
- **Results Summary**: Comprehensive quiz completion statistics
- **Session Management**: Maintains quiz state across interactions

### 📊 Quiz Reporting & Insights
- **Detailed Performance Report**: Get a comprehensive summary of your quiz performance.
- **Topic-Specific Insights**: Identifies key areas and concepts where you need improvement.
- **Personalized Study Recommendations**: Provides actionable strategies to enhance your learning.
- **Database Storage**: All quiz results, user answers, and generated insights are saved to a local database (`quiz_reports.db`) for future review and progress tracking.

### Smart Model Management
- **Automatic Model Detection**: Shows which models are locally available vs. need downloading
- **One-Click Model Setup**: Simply select any model - downloading happens automatically
- **Visual Status Indicators**: Clear icons show model availability status (Local, To be pulled, API)
- **Background Downloads**: Models download seamlessly with progress feedback
- **Zero Manual Setup**: No need to manually run `ollama pull` commands
- **Separate Model Selection**: Different models for document processing (OpenAI) and quiz generation (OpenAI or Ollama)
- **Contextual UI**: Interface adapts based on selected model type

### Performance & Reliability
- **Flexible Model Selection**: Choose from OpenAI or Ollama models
- **Connection Testing**: Verify model connectivity with one click
- **Progress Indicators**: Real-time feedback during processing
- **Error Handling**: Graceful handling of processing failures
- **Automatic Fallback**: Basic segmentation when ContextGem is unavailable

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- **OpenAI API Key**: Required for document processing
- [Ollama](https://ollama.ai/): Optional, only needed if using local models for quiz generation
- **ChromaDB**: Automatically installed with dependencies, but ensures persistent document storage

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/maurizioorani/QuizMasterPro.git
   cd QuizMasterPro
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up OpenAI API Key**
   Create a `.env` file in the project root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Optional: Install and Setup Ollama** (only if you plan to use local models for quiz generation)
   ```bash
   # Install Ollama (visit https://ollama.ai/ for platform-specific instructions)
   
   # Start Ollama Server
   ollama serve
   
   # Models will be downloaded automatically when selected in the app
   ```

5. **Launch QuizMaster Pro**
   ```bash
   streamlit run src/quizmaster/streamlit_app.py
   ```

6. **Open in Browser**
   Navigate to `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Configure Settings
- **Embedding Model Selection**: Choose from available OpenAI models in the sidebar for document processing
  - ⚡ OpenAI API key required in .env file
  - ⭐ **Recommended**: Use `gpt-4.1-nano` or `gpt-4o-mini` for best embedding results
  
- **Quiz Generation Model Selection**: Choose from available OpenAI or Ollama models
  - ✅ Ollama models marked as "Local" are already downloaded
  - 📥 Ollama models marked as "To be pulled" will be automatically downloaded
  - ⚡ OpenAI models require API key
  - ⭐ **Recommended**: Use `mistral:7b`, `qwen2.5:7b`, or `gemma2:9b` for best quiz generation
  
- **Automatic Model Management**: Simply select any model - QuizMaster will handle the rest
- **Connection Testing**: Use "Test Ollama" button to verify local model connections
- Configure quiz settings:
  - Number of questions (1-20)
  - Difficulty level (Easy/Medium/Hard)
  - Question types to include
  - Optional focus section keywords

### Step 2: Upload & Process Document
- Click "Upload a document" and select your file.
- Supported formats: PDF, DOCX, TXT, HTML.
- Click "Process Document" to analyze the content.
- The document will be automatically stored in the local ChromaDB.
- Review the document summary and content preview.

### Step 3: Manage Stored Documents (New!)
- In the "Document Management" section, you can:
  - See a list of all previously processed and stored documents.
  - Select a document from the dropdown to load it back into the application.
  - Click "Load Selected Document" to prepare it for quiz generation or review.
  - Click "Delete Selected Document" to remove it from storage.

### Step 4: Generate Quiz
- Ensure a document is processed or loaded.
- Click "Generate Quiz" to create questions.
- Monitor the progress bar during generation.
- Review generation info (model used, content length).

### Step 4: Take the Quiz
- Navigate through questions using Previous/Next buttons
- Answer each question according to its type
- Click "Show Answer" to see explanations
- View final results at the end

## 🔧 Configuration

### Supported Models

#### Embedding Models (Document Processing)
- `gpt-4.1-nano` - Lightweight but powerful model optimized for efficiency
- `gpt-4o-mini` - Specialized for structured output and JSON generation

#### Quiz Generation Models

**⭐ Recommended for Best Performance:**
- `mistral:7b` - Excellent JSON generation, efficient (4.1GB), reliable structured output
- `qwen2.5:7b` - Strong JSON/structured output, multilingual, efficient (4.4GB)
- `gemma2:9b` - Google's instruction-following model, good JSON reliability (5.4GB)

**🔧 JSON Specialists (Great for Quiz Generation):**
- `deepseek-coder:6.7b` - Code-focused, excellent structured formats (3.8GB)
- `codellama:7b` - Meta's coding model, strong with structured data (3.8GB)

**Other Capable Models:**
- `llama3.3:8b` - Meta's latest generation, 8B parameters, strong all-rounder
- `llama3.3:70b` - Meta's latest generation, 70B parameters (large size, high resource requirements)
- `llama3.1:8b` - Previous generation, may have JSON formatting issues
- `gemma3:9b` - Google's newer model, good for coding & reasoning
- `phi4:latest` - Microsoft's efficient model
- `deepseek-r1:latest` - DeepSeek's advanced reasoning model
- `granite3.3:latest` - IBM's enterprise-focused model

### Question Types
- **Multiple Choice**: 4 options with plausible distractors
- **Open-Ended**: Requires detailed written responses
- **True/False**: Binary choice questions
- **Fill-in-the-Blank**: Complete sentences with missing terms

### Difficulty Levels
- **Easy**: Basic facts and straightforward concepts
- **Medium**: Understanding and application of ideas
- **Hard**: Analysis, synthesis, and critical thinking

## 🏗️ Architecture

### Core Components
- **`streamlit_app.py`**: Main application interface and document management.
- **`document_processor.py`**: Handles document parsing, intelligent segmentation (now more granular and filtered), and metadata extraction.
- **`quiz_generator.py`**: LLM-powered question generation with refined prompts for conceptual and natural questions.
- **`insights_generator.py`**: Generates personalized quiz performance reports and study insights using LLMs.
- **`database_manager.py`**: Manages the SQLite database for storing quiz sessions, answers, and insights reports.
- **`chroma_manager.py`**: Manages persistent storage and retrieval of documents using ChromaDB.
- **`requirements.txt`**: Python dependencies.

### Data Flow
1. Document upload and validation.
2. Content extraction and intelligent segmentation (now sentence-level with irrelevant content filtering).
3. Metadata extraction and structured content creation.
4. **Document Storage**: Processed documents are automatically stored in ChromaDB.
5. **Document Retrieval**: Stored documents can be loaded from ChromaDB for reuse.
6. LLM prompt engineering and question generation (using specific segments and refined prompts).
7. Interactive quiz presentation and scoring.
8. **Quiz Reporting & Storage**: After quiz completion, user answers are analyzed, insights are generated by an LLM, and all quiz data (session, answers, report) is saved to the local SQLite database.

### Hybrid Model Design
- Document processing requires OpenAI API access for reliable embedding
- Quiz generation can use either OpenAI models or local Ollama models
- Document storage uses local ChromaDB for persistent access
- Quiz reports and user data stored in local SQLite database

## 📋 Requirements Compliance

✅ **Document Processing**: Multi-format support with enhanced intelligent segmentation and irrelevant content filtering.  
✅ **Document Management**: Persistent storage, listing, loading, and deletion of documents via ChromaDB.  
✅ **Quiz Generation**: All question types with difficulty levels, now generating more conceptual and naturally phrased questions.  
✅ **Interactive Experience**: Full quiz interface with navigation.  
✅ **Quiz Reporting & Insights**: Detailed performance reports, topic-specific insights, and personalized study recommendations, all stored in a local database.  
✅ **Local LLM Integration**: Complete Ollama integration with model selection for both quiz generation and insights.  
✅ **Performance**: Optimized for responsiveness with progress indicators.  
✅ **Reliability**: Comprehensive error handling and fallback mechanisms.  
✅ **Usability**: Intuitive interface with helpful guidance.  
✅ **Flexibility**: Hybrid approach with option to use local models for quiz generation.  
✅ **Architecture**: Clean, modular codebase structure.  

## 🛠️ Troubleshooting

### Common Issues

**Ollama Connection Failed**
- Ensure Ollama is installed and running (`ollama serve`)
- Check if the service is accessible at `http://localhost:11434`
- Verify at least one model is downloaded (`ollama list`)

**Document Processing Errors**
- Ensure file is not corrupted or password-protected
- Check file format is supported (PDF, DOCX, TXT, HTML)
- Try with a smaller document first

**Quiz Generation Issues**
- Reduce the number of questions requested
- Try a different Ollama model
- Ensure document content is substantial enough
- Check Ollama model has sufficient context window

**Performance Issues**
- Use smaller models like `llama3.2:3b` for faster generation
- Reduce document size or use focus sections
- Ensure adequate system resources for chosen model

### System Requirements
- **Minimum**: 8GB RAM, 4GB available storage
- **Recommended**: 16GB RAM, 10GB available storage
- **For large models**: 32GB+ RAM recommended

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Ollama documentation
3. Create an issue on GitHub

---

**QuizMaster Pro** - Empowering learning through intelligent quiz generation 🎓
