import json
import os
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
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adding a comment to force Streamlit reload
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
        self.openai_models = [
            "gpt-4.1-nano",       # ⚡ OPENAI: GPT-4.1 Nano - Lightweight but powerful
            "gpt-4o-mini",        # ⚡ OPENAI: GPT-4O Mini - Optimized for structured output
        ]
        self.ollama_models = [
            # JSON-optimized models (recommended first)
            "mistral:7b",         # ⭐ RECOMMENDED: Excellent JSON generation, efficient (4.1GB)
            "qwen2.5:7b",         # ⭐ RECOMMENDED: Strong JSON/structured output, multilingual (4.4GB)
            "gemma2:9b",          # ⭐ RECOMMENDED: Google's instruction-following, good JSON (5.4GB)
            "deepseek-coder:6.7b", # 🔧 JSON SPECIALIST: Code-focused, excellent structured formats (3.8GB) 
            "codellama:7b",       # 🔧 JSON SPECIALIST: Meta's coding model, strong structured data (3.8GB)
            "mistral-nemo:latest", # Nvidia's optimized version of Mistral for structured generation
            # Other capable models  
            "llama3.3:8b",        # Meta's Llama 3.3, 8 billion parameters
            "llama3.3:70b",       # Meta's Llama 3.3, 70 billion parameters (large size)
            "gemma3:9b",          # Google's Gemma 3, 9 billion parameters
            "phi4:latest",        # Microsoft's Phi-4 model
            "deepseek-r1:latest", # DeepSeek-R1 model
            "granite3.3:latest",  # IBM's Granite 3.3 model
            "llama3.1:8b"         # Meta's Llama 3.1 (may have JSON formatting issues)
        ]
        self.available_models = self.openai_models + self.ollama_models
        self.current_model = "gpt-4.1-nano"  # Default to OpenAI model
        self._question_cache = {}
        
    def set_model(self, model_name: str, auto_pull: bool = True) -> bool:
        """Set the model to use for generation."""
        # Log the requested model
        logger.info(f"Setting model to: {model_name}")
        
        # Normalize model name by removing 'ollama/' prefix if present
        model_name = model_name.replace('ollama/', '')
        
        if model_name not in self.available_models:
            logger.error(f"Model {model_name} not available. Available models: {self.available_models}")
            raise ValueError(f"Model {model_name} not available. Choose from: {self.available_models}")
            
        # Skip local checks for OpenAI models
        if model_name in self.openai_models:
            # Store previous model for logging
            prev_model = self.current_model
            self.current_model = model_name
            logger.info(f"Model changed from {prev_model} to OpenAI model {model_name}")
            return True
            
        # For Ollama models, check local availability
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
        
        # Store previous model for logging
        prev_model = self.current_model
        self.current_model = model_name
        logger.info(f"Model changed from {prev_model} to Ollama model {model_name}")
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
                    
        logger.error("Failed to connect to Ollama after all retries")
        return False
    
    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama server with retries."""
        # If current model is an OpenAI model, test OpenAI connection instead
        if self.current_model in self.openai_models:
            try:
                # Just check for OPENAI_API_KEY presence
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.error("OpenAI API key not found in environment variables")
                    return False
                logger.info("OpenAI API key found, assuming connection is available")
                return True
            except Exception as e:
                logger.error(f"Error checking OpenAI connection: {str(e)}")
                return False
        
        # For Ollama models
        for attempt in range(self.config.max_retries):
            try:
                # Use proper format for Ollama models
                response = completion(
                    model=f"ollama/{self.current_model}",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base=self.config.ollama_base_url,
                    max_tokens=10,
                    temperature=0.1
                )
                logger.info(f"Successfully connected to Ollama with model {self.current_model}")
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
        
        # Log which model is being used
        logger.info(f"Making LLM request with model: {self.current_model}")
        
        for attempt in range(self.config.max_retries):
            try:
                if self.current_model in self.openai_models:
                    # Use OpenAI API for OpenAI models
                    logger.info(f"Using OpenAI API with model {self.current_model}")
                    response = completion(
                        model=self.current_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                else:
                    # Use Ollama for local models - prefix with "ollama/"
                    logger.info(f"Using Ollama API with model {self.current_model}")
                    response = completion(
                        model=f"ollama/{self.current_model}",
                        messages=[{"role": "user", "content": prompt}],
                        api_base=self.config.ollama_base_url,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False  # Ensure streaming is disabled for consistent behavior
                    )
                
                # Log success
                logger.info(f"LLM request successful with model {self.current_model}")
                return response.choices[0].message.content
                
            except RuntimeError as e:
                if "cannot schedule new futures after interpreter shutdown" in str(e):
                    logger.error("Interpreter shutdown detected - restart required")
                    return None
                logger.error(f"RuntimeError in LLM request: {str(e)}")
                raise
                
            except Exception as e:
                logger.warning(f"LLM request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {self.config.retry_delay * (attempt + 1)} seconds...")
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed for model {self.current_model}")
                    
        return None
    
    def _parse_json_response(self, response_text: str, schema: Dict) -> Optional[Dict]:
        """Parse and validate JSON response from LLM with improved robustness."""
        if not response_text:
            logger.warning("Empty response from LLM")
            return None
            
        # Log the response for debugging
        logger.info(f"Parsing JSON from response (first 100 chars): {response_text[:100]}...")
            
        # Try multiple extraction methods
        json_data = None
        
        # Method 1: Direct JSON parsing
        try:
            json_data = json.loads(response_text)
            logger.info("Successfully parsed JSON directly")
        except Exception as e:
            logger.info(f"Direct JSON parsing failed: {str(e)}")
            
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
                        logger.info("Successfully parsed JSON using brace matching")
                    except Exception as e:
                        logger.info(f"Brace matching JSON parsing failed: {str(e)}")
                    
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
                        logger.info("Successfully parsed JSON array")
                    except Exception as e:
                        logger.info(f"Array JSON parsing failed: {str(e)}")
        
        # Method 4: Try to fix common JSON errors
        if not json_data:
            # Try to fix common JSON errors
            fixed_text = response_text
            # Replace single quotes with double quotes
            fixed_text = re.sub(r"'([^']*)':", r'"\1":', fixed_text)
            # Fix trailing commas
            fixed_text = re.sub(r',\s*}', '}', fixed_text)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            
            try:
                # Try to extract JSON with regex
                json_match = re.search(r'({.*})', fixed_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    json_data = json.loads(json_str)
                    logger.info("Successfully parsed JSON after fixing common errors")
            except Exception as e:
                logger.info(f"Fixed JSON parsing failed: {str(e)}")
                    
        # Validate against schema
        if json_data:
            try:
                validate(instance=json_data, schema=schema)
                logger.info("JSON validation successful")
                return json_data
            except ValidationError as e:
                logger.warning(f"JSON validation failed: {str(e)}")
                # Store the error for debugging
                self._parse_json_response_error = str(e)
                
                # Try to fix common validation issues
                if isinstance(json_data, dict):
                    # For Multiple Choice questions, ensure options have correct format
                    if 'options' in json_data and isinstance(json_data['options'], list):
                        for i, opt in enumerate(json_data['options']):
                            if isinstance(opt, str):
                                # Convert string options to proper format
                                letter = chr(65 + i)  # A, B, C, D
                                json_data['options'][i] = {"letter": letter, "text": opt}
                        
                        # Try validation again after fixes
                        try:
                            validate(instance=json_data, schema=schema)
                            logger.info("JSON validation successful after fixes")
                            return json_data
                        except ValidationError:
                            pass
                
        logger.warning("All JSON parsing methods failed")
        return None
    
    @lru_cache(maxsize=100)
    def _get_cache_key(self, content_hash: str, question_type: str, 
                      difficulty: str, question_num: int, focus_section: str = "") -> str:
        """Generate cache key for questions with focus section support."""
        # Add a random component to ensure unique cache keys for each question
        import uuid
        return f"{content_hash}_{focus_section}_{question_type}_{difficulty}_{question_num}_{uuid.uuid4().hex[:8]}"
    
    def _is_question_unique(self, new_question: Dict, existing_questions: List[Dict]) -> bool:
        """Check if a new question is unique compared to existing questions."""
        # If we have very few questions, be more lenient
        if len(existing_questions) < 2:
            return True
            
        new_text = new_question['text'].lower()
        
        # Check for similar core concepts
        new_core = re.sub(r'\b(?:which|what|how|is|are|does|do|can|could|would|should)\b', '', new_text)
        new_core = re.sub(r'\W+', ' ', new_core).strip()[:50]
        
        for existing in existing_questions:
            existing_text = existing['text'].lower()
            existing_core = re.sub(r'\b(?:which|what|how|is|are|does|do|can|could|would|should)\b', '', existing_text)
            existing_core = re.sub(r'\W+', ' ', existing_core).strip()[:50]
            
            # Check if core concepts are too similar - more lenient now
            if new_core == existing_core and len(new_core) > 15:  # Only reject if substantial overlap
                logger.info(f"Rejecting question due to core concept similarity: {new_core}")
                return False
                
            # Check for high text similarity - more lenient threshold
            if len(new_text) > 30 and len(existing_text) > 30:
                similarity = SequenceMatcher(None, new_text, existing_text).ratio()
                if similarity > 0.85:  # Increased from 0.7 to 0.85
                    logger.info(f"Rejecting question due to high similarity: {similarity}")
                    return False
                    
        return True
        
    def generate(self, processed_content: Dict, question_types: List[str], 
                num_questions: int, difficulty: str, focus_section: Optional[str] = None,
                use_cache: bool = False) -> Dict:
        """Generate quiz questions using ContextGem extracted data."""
        
        # Get content and extracted concepts from ContextGem processing
        content = processed_content.get('content', '')
        extracted_concepts = processed_content.get('extracted_concepts', [])
        
        logger.info(f"Starting quiz generation for {num_questions} questions of type {question_types} and difficulty {difficulty}.")
        logger.info(f"Processed content available: {bool(content)}")
        logger.info(f"Extracted concepts count: {len(extracted_concepts)}")

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

        # Prepare context data for question generation with usage tracking
        context_data = []
        
        # Add full document content
        context_data.append({
            'content': self._truncate_content(content, 2000),
            'type': 'full_document',
            'source': 'document',
            'used': False
        })
        
        # Add ContextGem extracted concepts
        for concept in extracted_concepts:
            context_data.append({
                'content': concept['content'],
                'type': concept['concept_name'],
                'source': concept.get('source_sentence', ''),
                'used': False
            })

        questions = []
        generated_count = 0
        failed_count = 0
        used_context_indices = set()
        
        # Create a pool of question types to draw from
        question_pool = []
        for question_type, count in self._distribute_questions(question_types, num_questions).items():
            for _ in range(count):
                question_pool.append(question_type)
        
        random.shuffle(question_pool)

        logger.info(f"Initial question pool distribution: {self._distribute_questions(question_types, num_questions)}")
        logger.info(f"Question pool size: {len(question_pool)}")

        # Generate questions with retries for failed ones
        attempts = 0
        max_attempts = num_questions * 6  # Allow more attempts for diversity (doubled)

        logger.info(f"Starting question generation loop. Target: {num_questions}, Max attempts: {max_attempts}")
        
        while len(questions) < num_questions and attempts < max_attempts:
            question_type = question_pool[attempts % len(question_pool)]  # Cycle through question types
            logger.info(f"Attempt {attempts + 1}/{max_attempts}: Generating {question_type} question.")
            
            try:
                # Select relevant context (avoiding used contexts)
                available_indices = [i for i, ctx in enumerate(context_data) if not ctx['used']]
                if not available_indices:
                    available_indices = list(range(len(context_data)))  # Fallback to all contexts
                    
                selected_idx = random.choice(available_indices)
                selected_context = context_data[selected_idx]['content']
                context_data[selected_idx]['used'] = True
                used_context_indices.add(selected_idx)
                logger.info(f"Selected context from index {selected_idx}. Context snippet: {selected_context[:100]}...")
                
                # Generate question using selected context
                question = self._generate_question_with_context(
                    selected_context, question_type, difficulty
                )

                if question:
                    logger.info(f"Successfully generated a question (type: {question.get('type', 'N/A')}). Checking uniqueness.")
                    # Check for uniqueness before adding
                    if self._is_question_unique(question, questions):
                        questions.append(question)
                        generated_count += 1
                        logger.info(f"Question added. Total generated: {generated_count}/{num_questions}")
                    else:
                        # Release context for reuse
                        context_data[selected_idx]['used'] = False
                        used_context_indices.discard(selected_idx)
                        failed_count += 1
                        logger.warning(f"Duplicate question skipped: {question['text'][:50]}...")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to generate {question_type} question (attempt {attempts + 1})")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error generating question (attempt {attempts + 1}): {str(e)}", exc_info=True)
                
            attempts += 1
            
        logger.info(f"Question generation loop finished. Generated: {len(questions)}, Failed: {failed_count}, Attempts: {attempts}")

        # Fallback for remaining questions
        if len(questions) < num_questions:
            remaining = num_questions - len(questions)
            logger.warning(f"Generating {remaining} additional questions with any available context")
            
            for i in range(remaining):
                try:
                    # Find any unused context
                    available_indices = [i for i, ctx in enumerate(context_data) if not ctx['used']]
                    if not available_indices:
                        available_indices = list(range(len(context_data)))  # Fallback to all contexts
                        
                    selected_idx = random.choice(available_indices)
                    selected_context = context_data[selected_idx]['content']
                    context_data[selected_idx]['used'] = True
                    
                    # Generate with random question type
                    q_type = random.choice(question_types)
                    question = self._generate_question_with_context(
                        selected_context, q_type, difficulty
                    )
                    
                    if question and self._is_question_unique(question, questions):
                        questions.append(question)
                        generated_count += 1
                    else:
                        failed_count += 1
                        context_data[selected_idx]['used'] = False
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error generating additional question: {str(e)}", exc_info=True)
        
        logger.info(f"Final question count: {len(questions)}")
        
        # If no questions were generated with Ollama model, try fallback to OpenAI
        if len(questions) == 0 and self.current_model not in self.openai_models:
            logger.warning(f"Failed to generate any questions with {self.current_model}. Attempting fallback to OpenAI.")
            
            # Save current model to restore later
            original_model = self.current_model
            
            # Try with OpenAI model if available
            if os.environ.get("OPENAI_API_KEY") and len(self.openai_models) > 0:
                fallback_model = self.openai_models[0]  # Use first available OpenAI model
                logger.info(f"Using fallback model: {fallback_model}")
                
                try:
                    # Temporarily switch to OpenAI model
                    self.current_model = fallback_model
                    
                    # Generate questions with OpenAI
                    fallback_result = self.generate(
                        processed_content,
                        question_types,
                        num_questions,
                        difficulty,
                        focus_section
                    )
                    
                    # Restore original model
                    self.current_model = original_model
                    
                    # Return fallback result with note
                    fallback_result['fallback_used'] = True
                    fallback_result['original_model'] = original_model
                    logger.info(f"Successfully generated {len(fallback_result['questions'])} questions with fallback model.")
                    return fallback_result
                    
                except Exception as e:
                    # Restore original model if fallback fails
                    self.current_model = original_model
                    logger.error(f"Fallback to OpenAI also failed: {str(e)}")
            else:
                logger.warning("Cannot use OpenAI fallback: API key missing or no OpenAI models available")
        
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
            'content_length': len(content),
            'fallback_used': False
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
        
        # Enhanced prompt for diversity
        prompt = f"""You are an expert quiz question generator. Create a {difficulty} difficulty {question_type} question based on the following context.

Important Instructions:
- Focus on distinct concepts not covered in previous questions
- Vary question phrasing and structure
- Avoid repetitive question patterns
- Ensure the question tests deep understanding, not just recall

Context:
{context}
"""
        if question_type == "Multiple Choice":
            prompt += """
- Provide 4 plausible options with only one correct answer
- Make incorrect options realistic but clearly wrong

Format response as valid JSON:
{
  "question": "Question text",
  "options": [
    {{"letter": "A", "text": "Option A"}},
    {{"letter": "B", "text": "Option B"}},
    {{"letter": "C", "text": "Option C"}},
    {{"letter": "D", "text": "Option D"}}
  ],
  "correct_answer": "A",
  "explanation": "Detailed explanation"
}"""
        elif question_type == "Open-Ended":
             prompt += """
Format response as valid JSON:
{
  "question": "Question text",
  "sample_answer": "A detailed sample answer",
  "key_points": ["Point 1", "Point 2"],
  "explanation": "Detailed explanation"
}"""
        elif question_type == "True/False":
             prompt += """
Format response as valid JSON:
{
  "statement": "Statement text",
  "answer": true, // or false
  "explanation": "Detailed explanation"
}"""
        elif question_type == "Fill-in-the-Blank":
             prompt += """
Format response as valid JSON:
{
  "question": "Question text with [BLANK] placeholder",
  "answer": "The correct word or phrase for the blank",
  "explanation": "Detailed explanation"
}"""
        
        logger.info(f"Prompt for LLM (snippet): {prompt[:500]}...")
        # Make LLM request
        response = self._make_llm_request(prompt, max_tokens=800)
        logger.info(f"Raw LLM response: {response}")
        if not response:
            logger.warning("LLM request returned None.")
            return None
        
        # Parse and validate JSON
        schema = self.get_schema_for_question_type(question_type)
        logger.info(f"Parsing and validating JSON response against schema for {question_type}.")
        parsed_question = self._parse_json_response(response, schema)
        
        if not parsed_question:
            logger.warning("JSON parsing or validation failed.")
            # Log validation error if available
            if hasattr(self, '_parse_json_response_error') and self._parse_json_response_error:
                 logger.error(f"JSON validation error: {self._parse_json_response_error}")
            return None
        
        logger.info("JSON parsed and validated successfully.")
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

    def _distribute_questions(self, question_types: List[str], total_questions: int) -> Dict[str, int]:
        """Distribute total questions across selected question types."""
        distribution = {}
        if not question_types:
            return distribution
            
        # Calculate base questions per type and remainder
        base_count = total_questions // len(question_types)
        remainder = total_questions % len(question_types)
        
        # Distribute base counts
        for q_type in question_types:
            distribution[q_type] = base_count
            
        # Distribute remainder questions
        for i in range(remainder):
            distribution[question_types[i]] += 1
            
        # Ensure we have exactly the requested total
        actual_total = sum(distribution.values())
        if actual_total < total_questions:
            # Add remaining questions to first type
            distribution[question_types[0]] += (total_questions - actual_total)
            
        return distribution

    # Other methods remain unchanged...
    
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
            estimated_chars = max_tokens * 4
            if len(content) > estimated_chars:
                return content[:estimated_chars] + "..."
        return content

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.available_models.copy()
    
    def get_current_model(self) -> str:
        """Get the currently selected model name."""
        return self.current_model
    
    def get_model_type(self) -> str:
        """Get the type of current model (openai or ollama)."""
        return "openai" if self.current_model in self.openai_models else "ollama"
    
    def get_local_models(self) -> List[str]:
        """Get list of locally available Ollama models."""
        try:
            response = requests.get(f"{self.config.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
            return []
        except Exception as e:
            logger.error(f"Error getting local models: {str(e)}")
            return []

    # Additional unchanged methods...
    
    def get_schema_for_question_type(self, question_type: str) -> Dict:
        """Returns the JSON schema for a given question type."""
        schemas = {
            "Multiple Choice": self.MCQ_SCHEMA,
            "Open-Ended": self.OPEN_ENDED_SCHEMA,
            "True/False": self.TRUE_FALSE_SCHEMA,
            "Fill-in-the-Blank": self.FILL_BLANK_SCHEMA
        }
        return schemas.get(question_type, {})
