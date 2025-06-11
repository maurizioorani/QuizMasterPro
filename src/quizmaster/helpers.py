import json
import re
import logging
from typing import Dict, List, Optional, Any
from jsonschema import validate, ValidationError
from difflib import SequenceMatcher
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_json_response(response_text: str, schema: Dict) -> Optional[Dict]:
    """Parse and validate JSON response from LLM with improved robustness."""
    if not response_text:
        logger.warning("Empty response from LLM")
        return None
        
    logger.info(f"Parsing JSON from response (first 100 chars): {response_text[:100]}...")
        
    json_data = None
    
    try:
        json_data = json.loads(response_text)
        logger.info("Successfully parsed JSON directly")
    except Exception as e:
        logger.info(f"Direct JSON parsing failed: {str(e)}")
        
    if not json_data:
        start_index = response_text.find('{')
        if start_index != -1:
            brace_count = 0
            end_index = -1
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
                
    if not json_data:
        start_index = response_text.find('[')
        if start_index != -1:
            bracket_count = 0
            end_index = -1
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
    
    if not json_data:
        fixed_text = response_text
        fixed_text = re.sub(r"'([^']*)':", r'"\1":', fixed_text)
        fixed_text = re.sub(r',\s*}', '}', fixed_text)
        fixed_text = re.sub(r',\s*]', ']', fixed_text)
        
        try:
            json_match = re.search(r'({.*})', fixed_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                json_data = json.loads(json_str)
                logger.info("Successfully parsed JSON after fixing common errors")
        except Exception as e:
            logger.info(f"Fixed JSON parsing failed: {str(e)}")
                
    if json_data:
        try:
            validate(instance=json_data, schema=schema)
            logger.info("JSON validation successful")
            return json_data
        except ValidationError as e:
            logger.warning(f"JSON validation failed: {str(e)}")
            
            if isinstance(json_data, dict):
                if 'options' in json_data and isinstance(json_data['options'], list):
                    for i, opt in enumerate(json_data['options']):
                        if isinstance(opt, str):
                            letter = chr(65 + i)
                            json_data['options'][i] = {"letter": letter, "text": opt}
                    
                    try:
                        validate(instance=json_data, schema=schema)
                        logger.info("JSON validation successful after fixes")
                        return json_data
                    except ValidationError:
                        pass
            
    logger.warning("All JSON parsing methods failed")
    return None

def is_question_unique(new_question: Dict, existing_questions: List[Dict], similarity_threshold: float = 0.85) -> bool:
    """
    Check if a new question is unique compared to existing questions using text similarity.
    """
    if not existing_questions:
        return True

    new_text = new_question.get('text', '').lower()
    if not new_text:
        return False

    for existing in existing_questions:
        existing_text = existing.get('text', '').lower()
        if not existing_text:
            continue

        similarity = SequenceMatcher(None, new_text, existing_text).ratio()
        if similarity > similarity_threshold:
            logger.warning(f"Duplicate question detected with similarity {similarity:.2f}. New: '{new_text[:50]}...', Existing: '{existing_text[:50]}...'")
            return False

    return True

def truncate_content(content: str, max_tokens: int = 3000) -> str:
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

def distribute_questions(question_types: List[str], total_questions: int) -> Dict[str, int]:
    """Distribute total questions across selected question types."""
    distribution = {}
    if not question_types:
        return distribution
        
    base_count = total_questions // len(question_types)
    remainder = total_questions % len(question_types)
    
    for q_type in question_types:
        distribution[q_type] = base_count
        
    for i in range(remainder):
        distribution[question_types[i]] += 1
        
    actual_total = sum(distribution.values())
    if actual_total < total_questions:
        distribution[question_types[0]] += (total_questions - actual_total)
        
    return distribution
