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
            "llama3.3:8b",        # Meta's Llama 3.3, 8 billion parameters
            "llama3.3:70b",       # Meta's Llama 3.3, 70 billion parameters
            "gemma3:9b",          # Google's Gemma 3, 9 billion parameters
            "phi4:latest",        # Microsoft's Phi-4 model
            "deepseek-r1:latest", # DeepSeek-R1 model
            "granite3.3:latest",  # IBM's Granite 3.3 model
            "llama3.1:8b",        # Meta's Llama 3.1, 8 billion parameters
            "mistral:7b",         # Mistral AI's Mistral 7B, efficient
            "qwen2.5:7b"          # Alibaba's Qwen 2.5, 7 billion parameters
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
                    progress_placeholder.error(f"❌ Failed to download {model_name}. Check logs.")
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
        """Make LLM request with retry logic."""
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
        """Generate quiz questions with improved robustness and section focus."""
        
        # Get all segments from the processed content
        all_segments = processed_content.get('segments', [])
        
        # Filter segments based on focus_section if provided
        if focus_section:
            relevant_segments = []
            for seg in all_segments:
                # Check section headings first for better matching
                if 'heading' in seg and focus_section.lower() in seg['heading'].lower():
                    relevant_segments.append(seg)
                # Then check content if no heading matches
                elif focus_section.lower() in seg['text'].lower():
                    relevant_segments.append(seg)
            
            if not relevant_segments:
                logger.warning(f"Focus section '{focus_section}' not found. Using full document content for generation.")
                segments_to_use = all_segments
            else:
                segments_to_use = relevant_segments
        else:
            segments_to_use = all_segments
            
        if not segments_to_use:
            logger.error("No segments available for quiz generation.")
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

        # Distribute questions across available segments and types
        questions = []
        generated_count = 0
        failed_count = 0
        
        # Create a pool of (segment, question_type) pairs to draw from
        question_pool = []
        for question_type, count in self._distribute_questions(question_types, num_questions).items():
            for _ in range(count):
                # Randomly select a segment for each question
                if segments_to_use:
                    selected_segment = random.choice(segments_to_use)
                    question_pool.append((selected_segment, question_type))
                else:
                    logger.warning("No segments available to generate questions from.")
                    break
        
        random.shuffle(question_pool) # Shuffle to mix question types and segments

        for i, (segment, question_type) in enumerate(question_pool):
            segment_text = segment['text']
            
            # Truncate segment content if needed (e.g., for very long paragraphs)
            # The max_tokens here should be smaller than the overall document truncation
            # to ensure the LLM focuses on the specific segment.
            truncated_segment_text = self._truncate_content(segment_text, max_tokens=1000) 
            
            try:
                question = self._generate_question(truncated_segment_text, question_type, difficulty, i+1)
                if question:
                    questions.append(question)
                    generated_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to generate {question_type} question {i+1} from segment.")
            except Exception as e:
                failed_count += 1
                logger.error(f"Error generating {question_type} question from segment: {str(e)}")
        
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
            'content_length': len(processed_content['content']) # Total content length
        }
    
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
    
    def _generate_question(self, content: str, question_type: str, 
                          difficulty: str, question_num: int) -> Optional[Dict]:
        """Generate a single question with proper error handling."""
        
        generators = {
            "Multiple Choice": (self._generate_mcq, self._create_enhanced_fallback_mcq),
            "Open-Ended": (self._generate_open_ended, self._create_enhanced_fallback_open_ended),
            "True/False": (self._generate_true_false, self._create_enhanced_fallback_true_false),
            "Fill-in-the-Blank": (self._generate_fill_blank, self._create_enhanced_fallback_fill_blank)
        }
        
        if question_type not in generators:
            logger.error(f"Unknown question type: {question_type}")
            return None
            
        generator, fallback = generators[question_type]
        
        # Try primary generator
        question = generator(content, difficulty, question_num)
        
        # Use fallback if primary fails
        if not question:
            logger.info(f"Using fallback for {question_type} question")
            question = fallback(content, difficulty)
            
        return question
    
    def _generate_mcq(self, content: str, difficulty: str, question_num: int) -> Optional[Dict]:
        """Generate multiple choice question with validation."""
        
        difficulty_instructions = {
            "Easy": "Create questions about basic concepts or definitions. Focus on 'what' and 'who'.",
            "Medium": "Create questions that require understanding, comparison, or application of concepts. Focus on 'how' and 'why'.",
            "Hard": "Create questions that require analysis, evaluation, synthesis, or problem-solving based on the concepts. Focus on 'analyze', 'evaluate', 'predict'."
        }
        
        prompt = f"""Based *only* on the following text segment, create a multiple choice question.
Do NOT use any external knowledge or general information about the document (e.g., author, summary, publication date, specific step numbers, or exact phrases like "as described in the text segment").
Focus on the underlying concepts, principles, or implications.

Text Segment: {content}

Instructions:
- {difficulty_instructions[difficulty]}
- Generate 1 question with 4 answer options (A, B, C, D)
- Only one option should be correct.
- Make incorrect options plausible but clearly wrong, and conceptually related to the provided text segment.
- Provide a detailed explanation for the correct answer, explaining the concept and referencing the *provided text segment* generally (e.g., "as discussed in the segment").
- Ensure the question tests understanding of a concept, not just recall of a specific phrase.
- The question itself MUST NOT contain phrases like "According to the text segment", "As described in the text", "In the context", "From the provided text", or similar direct references to the source. Assume the context is already established.

You MUST format your response as valid JSON exactly like this:
{{
    "question": "Your question here?",
    "options": [
        {{"letter": "A", "text": "Option A"}},
        {{"letter": "B", "text": "Option B"}},
        {{"letter": "C", "text": "Option C"}},
        {{"letter": "D", "text": "Option D"}}
    ],
    "correct_answer": "A",
    "explanation": "Detailed explanation with text reference"
}}"""

        response_text = self._make_llm_request(prompt, max_tokens=500)
        if not response_text:
            return None
            
        question_data = self._parse_json_response(response_text, self.MCQ_SCHEMA)
        if not question_data:
            return None
            
        return {
            'type': 'Multiple Choice',
            'text': question_data['question'],
            'options': question_data['options'],
            'answer': question_data['correct_answer'],
            'explanation': question_data['explanation']
        }
    
    def _generate_open_ended(self, content: str, difficulty: str, question_num: int) -> Optional[Dict]:
        """Generate open-ended question with validation."""
        
        difficulty_instructions = {
            "Easy": "Ask for basic definitions or explanations of concepts. Focus on 'what' and 'who'.",
            "Medium": "Ask for comparisons, examples, or applications of concepts. Focus on 'how' and 'why'.",
            "Hard": "Ask for analysis, evaluation, or synthesis of complex ideas. Focus on 'analyze', 'evaluate', 'predict'."
        }
        
        prompt = f"""Based *only* on the following text segment, create an open-ended question.
Do NOT use any external knowledge or general information about the document (e.g., author, summary, publication date, specific step numbers, or exact phrases like "as described in the text segment").
Focus on the underlying concepts, principles, or implications.

Text Segment: {content}

Instructions:
- {difficulty_instructions[difficulty]}
- Create a question that requires a thoughtful, detailed response, directly answerable from the provided text segment.
- Provide a comprehensive sample answer, strictly based on the provided text segment.
- Include general references to the *provided text segment* (e.g., "as discussed in the segment").
- The question itself MUST NOT contain phrases like "According to the text segment", "As described in the text", "In the context", "From the provided text", or similar direct references to the source. Assume the context is already established.

You MUST format your response as valid JSON exactly like this:
{{
    "question": "Your open-ended question?",
    "sample_answer": "Comprehensive sample answer",
    "key_points": ["Key point 1", "Key point 2", "Key point 3"],
    "explanation": "Explanation of what makes a good answer"
}}"""

        response_text = self._make_llm_request(prompt, max_tokens=400)
        if not response_text:
            return None
            
        question_data = self._parse_json_response(response_text, self.OPEN_ENDED_SCHEMA)
        if not question_data:
            return None
            
        return {
            'type': 'Open-Ended',
            'text': question_data['question'],
            'answer': question_data['sample_answer'],
            'key_points': question_data.get('key_points', []),
            'explanation': question_data['explanation']
        }
    
    def _generate_true_false(self, content: str, difficulty: str, question_num: int) -> Optional[Dict]:
        """Generate true/false question with validation."""
        
        prompt = f"""Based *only* on the following text segment, create a true/false statement.
Do NOT use any external knowledge or general information about the document (e.g., author, summary, publication date, specific step numbers, or exact phrases like "as described in the text segment").
Focus on the underlying concepts, principles, or implications.

Text Segment: {content}

Instructions:
- Create a statement that can be clearly marked as true or false, directly from the provided text segment.
- For {difficulty.lower()} difficulty, make the statement {'straightforward' if difficulty == 'Easy' else 'require careful analysis' if difficulty == 'Hard' else 'moderately challenging'}.
- Provide explanation for why the statement is true or false, referencing the *provided text segment* generally (e.g., "as discussed in the segment").
- The question itself MUST NOT contain phrases like "According to the text segment", "As described in the text", "In the context", "From the provided text", or similar direct references to the source. Assume the context is already established.

You MUST format your response as valid JSON exactly like this:
{{
    "statement": "Your true/false statement",
    "answer": true,
    "explanation": "Explanation with text reference"
}}

Note: The answer field must be a boolean (true or false), not a string."""

        response_text = self._make_llm_request(prompt, max_tokens=300)
        if not response_text:
            return None
            
        question_data = self._parse_json_response(response_text, self.TRUE_FALSE_SCHEMA)
        if not question_data:
            return None
            
        return {
            'type': 'True/False',
            'text': question_data['statement'],
            'answer': 'True' if question_data['answer'] else 'False',
            'explanation': question_data['explanation']
        }
    
    def _generate_fill_blank(self, content: str, difficulty: str, question_num: int) -> Optional[Dict]:
        """Generate fill-in-the-blank question with validation."""
        
        prompt = f"""Based *only* on the following text segment, create a fill-in-the-blank question.
Do NOT use any external knowledge or general information about the document (e.g., author, summary, publication date, specific step numbers, or exact phrases like "as described in the text segment").
Focus on the underlying concepts, principles, or implications.

Text Segment: {content}

Instructions:
- Select an important term or concept to blank out from the *provided text segment*.
- The missing word should be directly answerable from the *provided text segment*.
- Replace the key word/phrase with "______".
- Provide the correct answer and explanation, referencing the *provided text segment* generally (e.g., "as discussed in the segment").
- The question itself MUST NOT contain phrases like "According to the text segment", "As described in the text", "In the context", "From the provided text", or similar direct references to the source. Assume the context is already established.
- For {difficulty.lower()} difficulty

You MUST format your response as valid JSON exactly like this:
{{
    "question": "Statement with ______ blank",
    "answer": "correct word/phrase",
    "explanation": "Explanation with text reference"
}}"""

        response_text = self._make_llm_request(prompt, max_tokens=300)
        if not response_text:
            return None
            
        question_data = self._parse_json_response(response_text, self.FILL_BLANK_SCHEMA)
        if not question_data:
            return None
            
        return {
            'type': 'Fill-in-the-Blank',
            'text': question_data['question'],
            'answer': question_data['answer'],
            'explanation': question_data['explanation']
        }
    
    # Enhanced fallback methods that extract content
    def _create_enhanced_fallback_mcq(self, content: str, difficulty: str) -> Dict:
        """Create content-aware fallback MCQ."""
        # Extract key terms from content
        sentences = [s.strip() for s in content.split('.') if 20 < len(s.strip()) < 200]
        if not sentences:
            return self._create_generic_fallback_mcq()
            
        selected_sentence = random.choice(sentences[:10])  # Use early sentences
        
        # Extract potential key terms (longer words, likely to be important)
        words = selected_sentence.split()
        key_terms = [w for w in words if len(w) > 5 and w[0].isupper()]
        
        if not key_terms:
            key_terms = [w for w in words if len(w) > 6]
            
        if not key_terms:
            return self._create_generic_fallback_mcq()
            
        key_term = random.choice(key_terms)
        
        # Create distractors from other sentences
        other_terms = []
        for sent in sentences[:20]:
            for word in sent.split():
                if len(word) > 5 and word != key_term and word not in other_terms:
                    other_terms.append(word)
                    
        distractors = random.sample(other_terms, min(3, len(other_terms)))
        while len(distractors) < 3:
            distractors.append(f"Option {len(distractors) + 1}")
            
        return {
            'type': 'Multiple Choice',
            'text': f"In the context of the document, what term is associated with: '{selected_sentence[:100]}...'?",
            'options': [
                {'letter': 'A', 'text': key_term},
                {'letter': 'B', 'text': distractors[0]},
                {'letter': 'C', 'text': distractors[1]},
                {'letter': 'D', 'text': distractors[2]}
            ],
            'answer': 'A',
            'explanation': f"The correct answer is '{key_term}' as mentioned in the provided context."
        }
    
    def _create_enhanced_fallback_open_ended(self, content: str, difficulty: str) -> Dict:
        """Create content-aware fallback open-ended question."""
        # Extract main topics from content
        sentences = content.split('.')[:5]  # Use first 5 sentences
        topics = []
        
        for sent in sentences:
            if len(sent) > 30:
                topics.append(sent.strip())
                
        question_templates = {
            "Easy": "Explain the main concept discussed in the beginning of this document.",
            "Medium": "Compare and contrast the different aspects mentioned in this document.",
            "Hard": "Analyze the implications of the concepts presented in this document."
        }
        
        return {
            'type': 'Open-Ended',
            'text': question_templates.get(difficulty, question_templates["Medium"]),
            'answer': f"The document discusses: {'. '.join(topics[:2])}...",
            'key_points': topics[:3] if topics else ['Main concepts', 'Supporting details', 'Conclusions'],
            'explanation': 'A comprehensive answer should address the key topics introduced in the document.'
        }
    
    def _create_enhanced_fallback_true_false(self, content: str, difficulty: str) -> Dict:
        """Create content-aware fallback true/false question."""
        sentences = [s.strip() for s in content.split('.') if 20 < len(s.strip()) < 150]
        
        if sentences:
            # Take a sentence and potentially negate it
            selected = random.choice(sentences[:10])
            is_true = random.choice([True, False])
            
            if not is_true:
                # Simple negation
                if "is" in selected:
                    statement = selected.replace("is", "is not", 1)
                elif "are" in selected:
                    statement = selected.replace("are", "are not", 1)
                else:
                    statement = f"It is false that {selected.lower()}"
            else:
                statement = selected
                
            return {
                'type': 'True/False',
                'text': statement,
                'answer': 'True' if is_true else 'False',
                'explanation': f"This statement is {'true' if is_true else 'false'} based on the document content."
            }
            
        return self._create_generic_fallback_true_false()
    
    def _create_enhanced_fallback_fill_blank(self, content: str, difficulty: str) -> Dict:
        """Create content-aware fallback fill-in-the-blank question."""
        sentences = [s.strip() for s in content.split('.') if 30 < len(s.strip()) < 150]
        
        if sentences:
            selected = random.choice(sentences[:10])
            words = selected.split()
            
            # Find a good word to blank out
            target_words = [w for w in words if 4 < len(w) < 15 and w.isalpha()]
            
            if target_words:
                target = random.choice(target_words)
                question = selected.replace(target, "______", 1)
                
                return {
                    'type': 'Fill-in-the-Blank',
                    'text': question,
                    'answer': target,
                    'explanation': f"The blank should be filled with '{target}' based on the original text."
                }
                
        return self._create_generic_fallback_fill_blank()
    
    # Generic fallbacks as last resort
    def _create_generic_fallback_mcq(self) -> Dict:
        """Create generic fallback MCQ."""
        return {
            'type': 'Multiple Choice',
            'text': 'Which of the following best describes a key concept from this document?',
            'options': [
                {'letter': 'A', 'text': 'A primary concept discussed'},
                {'letter': 'B', 'text': 'A secondary concept mentioned'},
                {'letter': 'C', 'text': 'An unrelated concept'},
                {'letter': 'D', 'text': 'None of the above'}
            ],
            'answer': 'A',
            'explanation': 'Please review the document to identify the main concepts.'
        }
    
    def _create_generic_fallback_open_ended(self) -> Dict:
        """Create generic fallback open-ended question."""
        return {
            'type': 'Open-Ended',
            'text': 'Summarize the main points discussed in this document.',
            'answer': 'A comprehensive summary should include the key concepts and ideas presented in the text.',
            'key_points': ['Main concept identification', 'Supporting details', 'Overall themes'],
            'explanation': 'A good answer should demonstrate understanding of the core content.'
        }
    
    def _create_generic_fallback_true_false(self) -> Dict:
        """Create generic fallback true/false question."""
        return {
            'type': 'True/False',
            'text': 'This document contains factual information.',
            'answer': 'True',
            'explanation': 'The statement is true as the document presents information and concepts.'
        }
    
    def _create_generic_fallback_fill_blank(self) -> Dict:
        """Create generic fallback fill-in-the-blank question."""
        return {
            'type': 'Fill-in-the-Blank',
            'text': 'The document discusses various ______ and concepts.',
            'answer': 'topics',
            'explanation': 'The document covers multiple topics and concepts as evidenced by the content.'
        }
    
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
