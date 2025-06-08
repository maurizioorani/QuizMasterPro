import json
import random
import re
import subprocess
import requests
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from litellm import completion
import tiktoken
from functools import lru_cache
from dataclasses import dataclass
from jsonschema import validate, ValidationError
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuizConfig:
    """Configuration for quiz generation."""
    ollama_base_url: str = "http://localhost:11434"
    max_retries: int = 3
    retry_delay: float = 1.0
    pull_timeout: int = 1800  # 30 minutes
    default_temperature: float = 0.7
    cache_size: int = 100
    
class QuizGenerator:
    """Enhanced quiz generator with improved robustness and error handling."""
    
    # JSON schemas for validation
    MCQ_SCHEMA = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "letter": {"type": "string", "pattern": "^[A-D]$"},
                        "text": {"type": "string"}
                    },
                    "required": ["letter", "text"]
                },
                "minItems": 4,
                "maxItems": 4
            },
            "correct_answer": {"type": "string", "pattern": "^[A-D]$"},
            "explanation": {"type": "string"}
        },
        "required": ["question", "options", "correct_answer", "explanation"]
    }
    
    OPEN_ENDED_SCHEMA = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "sample_answer": {"type": "string"},
            "key_points": {
                "type": "array",
                "items": {"type": "string"}
            },
            "explanation": {"type": "string"}
        },
        "required": ["question", "sample_answer", "key_points", "explanation"]
    }
    
    TRUE_FALSE_SCHEMA = {
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "answer": {"type": "boolean"},
            "explanation": {"type": "string"}
        },
        "required": ["statement", "answer", "explanation"]
    }
    
    FILL_BLANK_SCHEMA = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "explanation": {"type": "string"}
        },
        "required": ["question", "answer", "explanation"]
    }
    
    def __init__(self, config: Optional[QuizConfig] = None):
        self.config = config or QuizConfig()
        self.question_types = ["Multiple Choice", "Open-Ended", "True/False", "Fill-in-the-Blank"]
        self.difficulty_levels = ["Easy", "Medium", "Hard"]
        self.available_models = [
            # JSON-optimized models (recommended first)
            "mistral:7b",         # ⭐ RECOMMENDED: Excellent JSON generation, efficient (4.1GB)
            "qwen2.5:7b",         # ⭐ RECOMMENDED: Strong JSON/structured output, multilingual (4.4GB)
            "gemma2:9b",          # ⭐ RECOMMENDED: Google's instruction-following, good JSON (5.4GB)
            "deepseek-coder:6.7b", # 🔧 JSON SPECIALIST: Code-focused, excellent structured formats (3.8GB)
            "codellama:7b",       # 🔧 JSON SPECIALIST: Meta's coding model, strong structured data (3.8GB)
            # Other capable models
            "llama3.3:8b",        # Meta's Llama 3.3, 8 billion parameters
            "llama3.3:70b",       # Meta's Llama 3.3, 70 billion parameters (large size)
            "gemma3:9b",          # Google's Gemma 3, 9 billion parameters
            "phi4:latest",        # Microsoft's Phi-4 model
            "deepseek-r1:latest", # DeepSeek-R1 model
            "granite3.3:latest",  # IBM's Granite 3.3 model
            "llama3.1:8b"         # Meta's Llama 3.1 (may have JSON formatting issues)
        ]
        self.current_model = "llama3.3:8b"  # Default to a new Llama3.3 model
        self._question_cache = {}
        
    def set_model(self, model_name: str, auto_pull: bool = True) -> bool:
        """Set the Ollama model to use for generation with optional auto-pull."""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Choose from: {self.available_models}")
            
        # Check if model is available locally
        if not self.is_model_available(model_name):
            if auto_pull:
                logger.info(f"Model {model_name} not found locally. Attempting to pull...")
                success = self.pull_model(model_name)
                if not success:
                    logger.error(f"Failed to pull model {model_name}")
                    return False
            else:
                logger.warning(f"Model {model_name} not available locally and auto_pull is disabled")
                return False
                
        self.current_model = model_name
        logger.info(f"Successfully set model to {model_name}")
        return True
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally in Ollama with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(
                    f"{self.config.ollama_base_url}/api/tags",
                    timeout=10
                )
                if response.status_code == 200:
                    models = response.json()
                    available_models = [model['name'] for model in models.get('models', [])]
                    return model_name in available_models
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed checking model availability: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    
        return False
    
    def pull_model(self, model_name: str, progress_placeholder: Optional[Any] = None) -> bool:
        """Pull a model using Ollama with progress tracking and Streamlit placeholder updates."""
        try:
            logger.info(f"Pulling model {model_name}... This may take a few minutes.")
            if progress_placeholder:
                progress_placeholder.text(f"🔄 Initializing pull for {model_name}...")

            # Start the pull process
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )
            
            # Monitor progress and stream output
            if progress_placeholder:
                output_lines = []
                max_lines_to_display = 10 # Display last N lines to avoid clutter
                for line in iter(process.stdout.readline, ''):
                    cleaned_line = line.strip()
                    if cleaned_line: # Avoid empty lines
                        logger.info(f"Ollama pull output: {cleaned_line}")
                        output_lines.append(cleaned_line)
                        # Display the last few lines of output
                        display_text = "\n".join(output_lines[-max_lines_to_display:])
                        progress_placeholder.text(f"🔄 Pulling {model_name}:\n{display_text}")
                    if process.poll() is not None: # Check if process ended
                        break
                process.stdout.close()

            return_code = process.wait(timeout=self.config.pull_timeout) # Wait for process to complete

            if return_code == 0:
                logger.info(f"Successfully pulled model {model_name}")
                if progress_placeholder:
                    progress_placeholder.success(f"✅ Successfully downloaded {model_name}")
                return True
            else:
                # stderr was already captured if combined, otherwise read it here
                # stderr_output = process.stderr.read() if process.stderr else "Unknown error"
                logger.error(f"Error pulling model {model_name} (return code: {return_code})")
                if progress_placeholder:
                    progress_placeholder.error(f"❌ Failed to download {model_name}. Please check logs or Ollama status.")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while pulling model {model_name}")
            if progress_placeholder:
                progress_placeholder.error(f"⌛ Timeout pulling {model_name}.")
            if process: # Ensure process is defined
                process.terminate()
            return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            if progress_placeholder:
                progress_placeholder.error(f"❌ Error pulling {model_name}: {str(e)}")
            return False
    
    def _ensure_embedding_model(self, model_name: str):
        """Ensure the embedding model is available, pull if necessary"""
        try:
            # Check if model is available
            response = requests.get(f"{self.config.ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                if model_name in available_models:
                    logger.info(f"✅ Embedding model {model_name} is already available")
                    return
            
            # Model not available, try to pull it
            logger.info(f"🔄 Pulling embedding model {model_name}...")
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully pulled embedding model {model_name}")
            else:
                logger.error(f"❌ Failed to pull embedding model {model_name}: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking/pulling embedding model {model_name}: {e}")

    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama server with retries."""
        for attempt in range(self.config.max_retries):
            try:
                response = completion(
                    model=f"ollama/{self.current_model}",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base=self.config.ollama_base_url,
                    max_tokens=10,
                    temperature=0.1
                )
                return True
            except Exception as e:
                logger.warning(f"Ollama connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    
        logger.error("Failed to connect to Ollama after all retries")
        return False
    
    def _make_llm_request(self, prompt: str, max_tokens: int = 500, 
                         temperature: Optional[float] = None) -> Optional[str]:
        """Make LLM request with retry logic using litellm."""        
        temperature = temperature or self.config.default_temperature

        for attempt in range(self.config.max_retries):
            try:
                response = completion(
                    model=f"ollama/{self.current_model}",
                    messages=[{"role": "user", "content": prompt}],
                    api_base=self.config.ollama_base_url,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"LLM request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    
        return None
    
    def _parse_json_response(self, response_text: str, schema: Dict) -> Optional[Dict]:
        """Parse and validate JSON response from LLM with improved robustness."""
        if not response_text:
            return None
            
        # Try multiple extraction methods
        json_data = None
        
        # Method 1: Direct JSON parsing
        try:
            json_data = json.loads(response_text)
        except:
            pass
            
        # Method 2: Extract JSON using balanced brace matching
        if not json_data:
            # Find the first opening brace
            start_index = response_text.find('{')
            if start_index != -1:
                brace_count = 0
                end_index = -1
                # Scan through the string to find the matching closing brace
                for i in range(start_index, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_index = i
                            break
                if end_index != -1:
                    json_str = response_text[start_index:end_index+1]
                    try:
                        json_data = json.loads(json_str)
                    except:
                        pass
                    
        # Method 3: Try to find JSON array if object not found
        if not json_data:
            # Find the first opening bracket
            start_index = response_text.find('[')
            if start_index != -1:
                bracket_count = 0
                end_index = -1
                # Scan through the string to find the matching closing bracket
                for i in range(start_index, len(response_text)):
                    if response_text[i] == '[':
                        bracket_count += 1
                    elif response_text[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_index = i
                            break
                if end_index != -1:
                    json_str = response_text[start_index:end_index+1]
                    try:
                        json_data = json.loads(json_str)
                    except:
                        pass
                    
        # Validate against schema
        if json_data:
            try:
                validate(instance=json_data, schema=schema)
                return json_data
            except ValidationError as e:
                logger.warning(f"JSON validation failed: {str(e)}")
                
        return None
    
    @lru_cache(maxsize=100)
    def _get_cache_key(self, content_hash: str, question_type: str, 
                      difficulty: str, question_num: int, focus_section: str = "") -> str:
        """Generate cache key for questions with focus section support."""
        # Add a random component to ensure unique cache keys for each question
        import uuid
        return f"{content_hash}_{focus_section}_{question_type}_{difficulty}_{question_num}_{uuid.uuid4().hex[:8]}"
    
    def generate(self, processed_content: Dict, question_types: List[str], 
                num_questions: int, difficulty: str, focus_section: Optional[str] = None,
                use_cache: bool = False) -> Dict:
        """Generate quiz questions using ContextGem extracted data."""
        
        # Get content and extracted concepts from ContextGem processing
        content = processed_content.get('content', '')
        extracted_concepts = processed_content.get('extracted_concepts', [])
        
        if not content:
            logger.error("No content found in processed_content.")
            return {
                'questions': [],
                'metadata': processed_content.get('metadata', {}),
                'model_used': self.current_model,
                'content_length': 0,
                'generation_stats': {
                    'requested': num_questions,
                    'generated': 0,
                    'failed': num_questions,
                    'success_rate': 0
                }
            }

        # Prepare context data for question generation
        # Use both the full content and ContextGem extracted concepts
        context_data = []
        
        # Add full document content
        context_data.append({
            'content': self._truncate_content(content, 2000),
            'type': 'full_document',
            'source': 'document'
        })
        
        # Add ContextGem extracted concepts
        for concept in extracted_concepts:
            context_data.append({
                'content': concept['content'],
                'type': concept['concept_name'],
                'source': concept.get('source_sentence', '')
            })

        questions = []
        generated_count = 0
        failed_count = 0
        
        # Create a pool of question types to draw from
        question_pool = []
        for question_type, count in self._distribute_questions(question_types, num_questions).items():
            for _ in range(count):
                question_pool.append(question_type)
        
        random.shuffle(question_pool)

        for i, question_type in enumerate(question_pool):
            try:
                # Select relevant context for this question
                selected_context = self._select_relevant_context(context_data, question_type, focus_section)
                
                # Generate question using selected context
                question = self._generate_question_with_context(
                    selected_context, question_type, difficulty
                )

                if question:
                    questions.append(question)
                    generated_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to generate {question_type} question {i+1}.")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error generating {question_type} question: {str(e)}")
        
        return {
            'questions': questions,
            'metadata': {
                **processed_content.get('metadata', {}),
                'generation_stats': {
                    'requested': num_questions,
                    'generated': generated_count,
                    'failed': failed_count,
                    'success_rate': generated_count / num_questions if num_questions > 0 else 0
                }
            },
            'model_used': self.current_model,
            'content_length': len(content)
        }
    
    def _select_relevant_context(self, context_data: List[Dict], question_type: str, focus_section: Optional[str] = None) -> str:
        """Select relevant context for question generation using simple similarity."""
        if not context_data:
            return ""
        
        # If focus_section is specified, try to find relevant content
        if focus_section:
            relevant_contexts = []
            for ctx in context_data:
                if focus_section.lower() in ctx['content'].lower():
                    relevant_contexts.append(ctx['content'])
            
            if relevant_contexts:
                return "\n\n".join(relevant_contexts[:3])  # Limit to top 3 matches
        
        # For different question types, prefer different types of context
        if question_type == "Multiple Choice":
            # Prefer Key Definitions and Important Facts
            concepts = [ctx for ctx in context_data if ctx['type'] in ['Key Definition', 'Important Fact']]
            if concepts:
                return "\n\n".join([ctx['content'] for ctx in concepts[:2]])
        
        elif question_type == "True/False":
            # Prefer Important Facts
            facts = [ctx for ctx in context_data if ctx['type'] == 'Important Fact']
            if facts:
                return facts[0]['content']
        
        elif question_type == "Fill-in-the-Blank":
            # Prefer Key Definitions
            definitions = [ctx for ctx in context_data if ctx['type'] == 'Key Definition']
            if definitions:
                return definitions[0]['content']
        
        # Fallback: use a mix of available context
        selected = random.sample(context_data, min(3, len(context_data)))
        return "\n\n".join([ctx['content'] for ctx in selected])
    
    def _generate_question_with_context(self, context: str, question_type: str, difficulty: str) -> Optional[Dict]:
        """Generate a single question based on context."""
        if not context.strip():
            return None
        
        # Create prompt based on question type
        prompt = f"""You are an expert quiz question generator. Create a {difficulty} difficulty {question_type} question based on the following context.

Context:
{context}

Instructions:
- The question must be directly answerable from the context.
- Do NOT use any external knowledge.
- Ensure the question tests understanding of a concept, not just recall of a specific phrase.
- Do NOT reference the source directly (no "according to the text" phrases).
"""

        if question_type == "Multiple Choice":
            prompt += """
- Provide 4 options (A, B, C, D) with only one correct answer.
- Make incorrect options plausible but clearly wrong.

Format your response as valid JSON:
{
  "question": "Your question here",
  "options": [
    {"letter": "A", "text": "Option A text"},
    {"letter": "B", "text": "Option B text"},
    {"letter": "C", "text": "Option C text"},
    {"letter": "D", "text": "Option D text"}
  ],
  "correct_answer": "A",
  "explanation": "Detailed explanation of why the answer is correct"
}"""
        
        elif question_type == "Open-Ended":
            prompt += """
Format your response as valid JSON:
{
  "question": "Your question here",
  "sample_answer": "A comprehensive sample answer",
  "key_points": ["Key point 1", "Key point 2", "Key point 3"],
  "explanation": "Explanation of what makes a good answer"
}"""
        
        elif question_type == "True/False":
            prompt += """
Format your response as valid JSON:
{
  "statement": "A clear true or false statement",
  "answer": true or false,
  "explanation": "Detailed explanation of why the statement is true or false"
}"""
        
        elif question_type == "Fill-in-the-Blank":
            prompt += """
- Replace a key term/phrase with "______"
Format your response as valid JSON:
{
  "question": "Your question with ______ for the blank",
  "answer": "The correct word/phrase for the blank",
  "explanation": "Explanation of why this is the correct answer"
}"""

        # Make LLM request
        response = self._make_llm_request(prompt, max_tokens=800)
        if not response:
            return None
        
        # Parse and validate JSON
        schema = self.get_schema_for_question_type(question_type)
        parsed_question = self._parse_json_response(response, schema)
        
        if not parsed_question:
            return None
        
        # Convert to internal format
        if question_type == "Multiple Choice":
            return {
                'type': 'Multiple Choice',
                'text': parsed_question['question'],
                'options': parsed_question['options'],
                'answer': parsed_question['correct_answer'],
                'explanation': parsed_question['explanation']
            }
        elif question_type == "Open-Ended":
            return {
                'type': 'Open-Ended',
                'text': parsed_question['question'],
                'answer': parsed_question['sample_answer'],
                'key_points': parsed_question.get('key_points', []),
                'explanation': parsed_question['explanation']
            }
        elif question_type == "True/False":
            return {
                'type': 'True/False',
                'text': parsed_question['statement'],
                'answer': 'True' if parsed_question['answer'] else 'False',
                'explanation': parsed_question['explanation']
            }
        elif question_type == "Fill-in-the-Blank":
            return {
                'type': 'Fill-in-the-Blank',
                'text': parsed_question['question'],
                'answer': parsed_question['answer'],
                'explanation': parsed_question['explanation']
            }
        
        return None
    
    def get_schema_for_question_type(self, question_type: str) -> Dict:
        """Returns the JSON schema for a given question type."""
        schemas = {
            "Multiple Choice": self.MCQ_SCHEMA,
            "Open-Ended": self.OPEN_ENDED_SCHEMA,
            "True/False": self.TRUE_FALSE_SCHEMA,
            "Fill-in-the-Blank": self.FILL_BLANK_SCHEMA
        }
        return schemas.get(question_type, {})

    def _distribute_questions(self, question_types: List[str], total_questions: int) -> Dict[str, int]:
        """Distribute total questions across selected question types."""
        distribution = {}
        questions_per_type = max(1, total_questions // len(question_types))
        remaining_questions = total_questions % len(question_types)
        
        for i, question_type in enumerate(question_types):
            distribution[question_type] = questions_per_type
            if i < remaining_questions:
                distribution[question_type] += 1
                
        return distribution
    
    def _truncate_content(self, content: str, max_tokens: int = 3000) -> str:
        """Truncate content to fit within token limit."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            tokens = encoding.encode(content)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                return encoding.decode(truncated_tokens)
        except Exception as e:
            logger.warning(f"Token encoding failed, using character truncation: {str(e)}")
            # Fallback to character-based truncation
            estimated_chars = max_tokens * 4
            if len(content) > estimated_chars:
                return content[:estimated_chars] + "..."
                
        return content
    
    # Removed _generate_question, _generate_mcq, _generate_open_ended, _generate_true_false, _generate_fill_blank
    # and all fallback methods as LangChain RAG will handle this.
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        return self.available_models.copy()
    
    def get_current_model(self) -> str:
        """Get currently selected model."""
        return self.current_model
    
    def get_local_models(self) -> List[str]:
        """Get list of locally available models."""
        try:
            response = requests.get(f"{self.config.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            return []
        except Exception as e:
            logger.error(f"Error getting local models: {str(e)}")
            return []
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about quiz generation."""
        return {
            'cache_size': len(self._question_cache),
            'current_model': self.current_model,
            'config': {
                'max_retries': self.config.max_retries,
                'retry_delay': self.config.retry_delay,
                'default_temperature': self.config.default_temperature
            }
        }
