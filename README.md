# 📚 QuizMaster Pro

Transform any document into intelligent, personalized quizzes using AI. QuizMaster Pro offers flexible document processing and quiz generation using either OpenAI or local Ollama models.

![QuizMasterPro Demo](https://github.com/maurizioorani/QuizMasterPro/raw/main/screenshots/QuizMasterPro.gif)

## 🎯 Core Concept

Upload documents → AI extracts key concepts → Select topics to focus on → Generate targeted quizzes → Get personalized learning insights

## ✨ Key Features

### 📄 Smart Document Processing
- **Multi-format support**: PDF, DOCX, TXT, HTML with robust error handling
- **Advanced concept extraction**: Dual-source extraction using ContextGem + direct LLM analysis
- **Interactive topic selection**: Browse and select from automatically extracted concepts and topics
- **Persistent storage**: Document management with semantic search capabilities
- **Model synchronization**: Consistent AI model usage across document processing and quiz generation

### 🧠 Intelligent Quiz Generation
- **Unified model management**: Seamless switching between OpenAI and local Ollama models
- **Enhanced model support**: deepseek-r1, mistral:7b, qwen2.5:7b, gemma2:9b, and more
- **Topic-focused generation**: Create quizzes from selected concepts rather than entire documents
- **Multiple question types**: Multiple choice, open-ended, true/false, fill-in-the-blank
- **Three difficulty levels**: Easy, medium, hard with adaptive content based on selected topics
- **Smart question diversity**: Advanced algorithms to prevent repetitive questions

### 📊 Advanced Analytics
- **Performance insights**: Comprehensive analysis using enhanced InsightsGenerator
- **Learning patterns**: Cognitive load assessment, retention analysis, knowledge gap identification
- **Personalized recommendations**: Study strategies, improvement plans, actionable next steps
- **Progress tracking**: Historical performance and milestone progression

### 🎮 Interactive Experience
- **Modern interface**: Clean Streamlit UI with real-time progress tracking
- **Smart navigation**: Question flow with immediate feedback and explanations
- **Session management**: Resume quizzes and track completion status

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- **AI Model Access**: Either OpenAI API Key OR [Ollama](https://ollama.ai/) for local models
- Recommended: Both for maximum flexibility

### Installation
```bash
# 1. Clone repository
git clone https://github.com/maurizioorani/QuizMasterPro.git
cd QuizMasterPro

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
streamlit run src/quizmaster/streamlit_app.py
```

### Model Setup (Choose One or Both)

**Option A: OpenAI Models**
```bash
# Configure OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

**Option B: Local Ollama Models**
```bash
# Install Ollama (visit https://ollama.ai/)
ollama serve

# Models auto-download when selected in app
```

## 🎯 How to Use

### 1. **Process Documents & Extract Topics**
- Upload PDF, DOCX, TXT, or HTML files
- **AI automatically extracts key concepts** using dual-source analysis:
  - ContextGem for advanced document understanding  
  - Direct LLM analysis for topic identification
- **Review extracted concepts** organized by categories (Key Definitions, Main Ideas, Important Facts)
- Documents stored persistently with concept metadata for reuse

### 2. **Interactive Topic Selection**
- **Browse automatically extracted concepts** in an organized interface
- **Select specific topics** you want to focus on for learning
- **Combine multiple concepts** from different categories
- **Use "Select All" or individual checkboxes** for fine-grained control
- Preview selected concepts before quiz generation

### 3. **Generate Targeted Quiz**
- **Configure quiz settings**: question types, difficulty, number of questions
- **Select your AI model**: Choose between OpenAI or local Ollama models  
- **Enable model synchronization** for consistent concept extraction and quiz generation
- AI creates questions **focused exclusively on your selected topics**
- Get immediate feedback and detailed explanations

### 4. **Analyze Performance & Track Progress**
- Receive detailed insights and analytics on your selected topics
- Get personalized study recommendations based on performance
- Track learning progress over time with concept-specific metrics
- Review which topics need more attention

## 🔧 Model Configuration

### Document Processing & Quiz Generation
**🏆 Recommended Local Models (Ollama):**
- **`deepseek-r1`** - Advanced reasoning and structured output
- **`mistral:7b`** - Excellent JSON generation (4.1GB)
- **`qwen2.5:7b`** - Strong structured output (4.4GB)
- **`gemma2:9b`** - Reliable instruction following (5.4GB)

**☁️ OpenAI Models:**
- **`gpt-4o-mini`** - Fast and cost-effective
- **`gpt-4o`** - High-quality structured output
- **`gpt-3.5-turbo`** - Balanced performance

### Model Selection Tips
- **For document processing**: deepseek-r1, mistral:7b, or gpt-4o-mini work well
- **For quiz generation**: All models supported with consistent quality
- **Model synchronization**: Use the same model for both processing and generation for best results

## 🏗️ Architecture

### Core Components
```
├── streamlit_app.py        # Main interface & topic selection
├── document_processor.py   # ContextGem integration & parsing
├── vector_manager.py       # Enhanced ChromaDB operations
├── quiz_generator.py       # AI-powered question generation
├── insights_generator.py   # Advanced analytics & recommendations
└── database_manager.py     # Quiz sessions & performance tracking
```

### Document Processing with ContextGem
- **ContextGem Integration**: We leverage ContextGem for advanced document understanding and concept extraction. ContextGem is ideal for this application because it excels at breaking down complex documents into structured, meaningful concepts and aspects, which is crucial for generating targeted and relevant quiz questions. Its ability to identify key definitions, main ideas, and important facts directly supports our goal of transforming documents into personalized learning experiences.
- **Smart Retrieval**: Semantic search based on extracted concepts allows for efficient retrieval of relevant document sections for quiz generation.
- **Rich Metadata**: Document structure, creation timestamps, and extraction summaries are stored as metadata for comprehensive document management.
- **Topic Mapping**: Organized concept categories facilitate targeted quiz generation based on specific topics selected by the user.

### Intelligent Analytics Pipeline
- **Performance analysis**: Multi-dimensional scoring and pattern recognition
- **Learning style detection**: Adaptive recommendations based on quiz behavior
- **Cognitive load assessment**: Personalized study strategies and pacing
- **Progress benchmarking**: Milestone tracking and improvement measurement

## 🛠️ Troubleshooting

### Common Solutions
| Issue | Solution |
|-------|----------|
| Ollama connection failed | Run `ollama serve` and ensure port 11434 is available |
| OpenAI API errors | Verify OpenAI API key in `.env` file (if using OpenAI models) |
| Document processing errors | Check selected model is available and running |
| Poor quiz quality | Use recommended models (deepseek-r1, mistral:7b, qwen2.5:7b) |
| Slow performance | Reduce document size or use focus sections |

### System Requirements
- **Minimum**: 8GB RAM, 4GB storage
- **Recommended**: 16GB RAM, 10GB storage
- **Large models**: 32GB+ RAM

## 🔄 Recent Enhancements

### ContextGem Integration
- **Advanced concept extraction**: Automatic identification of key topics, definitions, and facts
- **Topic selection interface**: Choose specific concepts for targeted quiz generation
- **Enhanced document understanding**: Better context awareness for question generation

### Enhanced Analytics
- **Comprehensive insights**: Multi-layered performance analysis with cognitive load assessment
- **Personalized recommendations**: Learning style detection and adaptive study strategies
- **Progress tracking**: Historical performance analysis and milestone progression

### Vector Storage Improvements
- **Smart document management**: Persistent storage with semantic search capabilities
- **Concept-based organization**: Documents indexed by extracted topics and aspects
- **Enhanced retrieval**: Better context understanding for quiz generation

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Submit pull request

---

**QuizMaster Pro** - Intelligent learning through AI-powered quiz generation 🎓
