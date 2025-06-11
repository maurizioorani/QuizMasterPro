import random
import logging
from typing import Dict, List, Optional

from .helpers import is_question_unique, distribute_questions, parse_json_response
from .context_manager import ContextManager
from .quiz_result import QuizResult
from .config import QuizConfig
from .schemas import MCQ_SCHEMA, OPEN_ENDED_SCHEMA, TRUE_FALSE_SCHEMA, FILL_BLANK_SCHEMA
from .prompts import get_question_prompt # Import the function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .llm_manager import LLMManager
from .question_formatter import QuestionFormatter

class QuizGenerator:
    def __init__(self, config: Optional[QuizConfig] = None):
        self.config = config or QuizConfig()
        self.llm_manager = LLMManager(self.config)
        self.question_formatter = QuestionFormatter()
        self.current_model = self.config.current_model
        self.openai_models = self.config.openai_models
        self.MCQ_SCHEMA = MCQ_SCHEMA
        self.OPEN_ENDED_SCHEMA = OPEN_ENDED_SCHEMA
        self.TRUE_FALSE_SCHEMA = TRUE_FALSE_SCHEMA
        self.FILL_BLANK_SCHEMA = FILL_BLANK_SCHEMA

    def generate_quiz(self, processed_content: Dict, question_types: List[str],
                      num_questions: int, difficulty: str, focus_topics: Optional[List[str]] = None) -> Dict:
        """Generate quiz questions using ContextGem extracted data."""

        original_content = processed_content.get('content', '') # Renamed to avoid conflict
        extracted_concepts = processed_content.get('extracted_concepts', [])

        logger.info(f"Starting quiz generation for {num_questions} questions of type {question_types} and difficulty {difficulty}. Focus topics: {focus_topics}")
        logger.info(f"Processed content available: {bool(original_content)}")
        logger.info(f"Extracted concepts count: {len(extracted_concepts)}")

        if not original_content:
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

        context_data = []
        
        context_data.append({
            'content': self.llm_manager._truncate_content(original_content, 2000),
            'type': 'full_document',
            'source': 'document',
            'used': False
        })
        
        for concept in extracted_concepts:
            try:
                if isinstance(concept, dict):
                    concept_content = concept.get('content', '') # Renamed to avoid conflict
                    concept_name = concept.get('concept_name', 'General')
                    source = concept.get('source_sentence', '')
                elif isinstance(concept, str):
                    concept_content = concept
                    concept_name = 'Extracted Concept'
                    source = ''
                else:
                    logger.warning(f"Skipping unknown concept format: {type(concept)}")
                    continue
                
                if concept_content and concept_content.strip():
                    context_data.append({
                        'content': concept_content,
                        'type': concept_name,
                        'source': source,
                        'used': False
                    })
            except Exception as e:
                logger.warning(f"Error processing concept: {str(e)}")
                continue

        questions = []
        generated_count = 0
        failed_count = 0
        # used_context_indices = set() # This variable is unused
        
        question_pool = []
        for question_type, count in distribute_questions(question_types, num_questions).items():
            for _ in range(count):
                question_pool.append(question_type)
        
        random.shuffle(question_pool)

        logger.info(f"Initial question pool distribution: {distribute_questions(question_types, num_questions)}")
        logger.info(f"Question pool size: {len(question_pool)}")

        attempts = 0
        max_attempts = num_questions * 6

        logger.info(f"Starting question generation loop. Target: {num_questions}, Max attempts: {max_attempts}")
        
        while len(questions) < num_questions and attempts < max_attempts:
            question_type = question_pool[attempts % len(question_pool)]
            logger.info(f"Attempt {attempts + 1}/{max_attempts}: Generating {question_type} question.")
            
            try:
                context_manager = ContextManager(context_data)
                selected_context = context_manager.select_relevant_context(question_type, focus_topics)
                
                question = self._generate_question_with_context(
                    selected_context, question_type, difficulty
                )

                if question:
                    logger.info(f"Successfully generated a question (type: {question.get('type', 'N/A')}). Checking uniqueness.")
                    if is_question_unique(question, questions):
                        questions.append(question)
                        generated_count += 1
                        logger.info(f"Question added. Total generated: {generated_count}/{num_questions}")
                    else:
                        failed_count += 1
                        logger.warning(f"Duplicate question skipped: {question.get('text', 'N/A')[:50]}...")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to generate {question_type} question (attempt {attempts + 1})")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error generating question (attempt {attempts + 1}): {str(e)}", exc_info=True)
                
            attempts += 1
            
        logger.info(f"Question generation loop finished. Generated: {len(questions)}, Failed: {failed_count}, Attempts: {attempts}")

        if len(questions) < num_questions:
            remaining = num_questions - len(questions)
            logger.warning(f"Generating {remaining} additional questions with any available context")
            
            for _ in range(remaining): # Use _ if i is not used
                try:
                    context_manager = ContextManager(context_data)
                    # Pass focus_section=None explicitly if not using it here
                    selected_context = context_manager.select_relevant_context(random.choice(question_types), focus_topics=focus_topics if len(questions) < num_questions // 2 else None) # Apply focus more to initial questions
                    
                    q_type = random.choice(question_types)
                    question = self._generate_question_with_context(
                        selected_context, q_type, difficulty
                    )
                    
                    if question and is_question_unique(question, questions):
                        questions.append(question)
                        generated_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error generating additional question: {str(e)}", exc_info=True)
        
        logger.info(f"Final question count: {len(questions)}")
        
        return QuizResult(
            questions=questions,
            metadata={
                **processed_content.get('metadata', {}),
                'generation_stats': {
                    'requested': num_questions,
                    'generated': generated_count,
                    'failed': failed_count,
                    'success_rate': generated_count / num_questions if num_questions > 0 else 0
                }
            },
            model_used=self.current_model,
            content_length=len(original_content), # Use the original content length
            fallback_used=False
        ).to_dict()

    def _generate_question_with_context(self, context: str, question_type: str, difficulty: str) -> Optional[Dict]:
        """Generate a single question based on context."""
        if not context or not context.strip(): # Added check for None or empty context
            logger.warning("Context is empty or None, cannot generate question.")
            return None

        prompt = get_question_prompt(question_type, difficulty, context) # Call the imported function
        
        logger.info(f"Prompt for LLM (snippet): {prompt[:500]}...")
        response = self.llm_manager.make_llm_request(prompt, self.current_model, self.openai_models, max_tokens=800)
        logger.info(f"Raw LLM response: {response}")

        if not response:
            logger.warning("LLM request returned None.")
            return None
        
        schema = self.get_schema_for_question_type(question_type)
        parsed_question = parse_json_response(response, schema)

        if not parsed_question:
            logger.warning("JSON parsing or validation failed.")
            return None

        return self.question_formatter.format_question(question_type, parsed_question)

    def get_schema_for_question_type(self, question_type: str) -> Dict:
        """Returns the JSON schema for a given question type."""
        schemas = {
            "Multiple Choice": self.MCQ_SCHEMA,
            "Open-Ended": self.OPEN_ENDED_SCHEMA,
            "True/False": self.TRUE_FALSE_SCHEMA,
            "Fill-in-the-Blank": self.FILL_BLANK_SCHEMA
        }
        return schemas.get(question_type, {})
