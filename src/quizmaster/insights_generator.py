import json
import os
import logging
from litellm import completion
import time
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_insights(questions, model_name: str):
    """
    Generate personalized insights based on quiz performance.
    
    Args:
        questions: List of question dictionaries with user answers and correctness
        model_name: The name of the model to use for insights generation
    
    Returns:
        String containing personalized insights report
    """
    # Log original model name
    logger.info(f"Original model name received: {model_name}")
    
    # Normalize model name by removing any existing 'ollama/' prefix
    original_model_name = model_name
    model_name = model_name.replace('ollama/', '')
    
    if original_model_name != model_name:
        logger.info(f"Normalized model name: {model_name}")
    
    # Calculate basic statistics for the prompt
    total_questions = len(questions)
    answered_questions = sum(1 for q in questions if q.get('user_answer') is not None)
    correct_questions = sum(1 for q in questions if q.get('is_correct') is True)
    
    # Prepare the prompt with more detailed instructions for structured output
    prompt = f"""
Analyze these quiz results and provide a detailed, well-structured learning insights report with the following sections:

1. PERFORMANCE SUMMARY
   - Overall score: {correct_questions}/{total_questions} questions answered correctly
   - Strengths and weaknesses overview

2. KEY IMPROVEMENT AREAS
   - Identify 2-3 specific topics or concepts the student needs to improve
   - For each area, explain why it needs improvement based on the quiz results

3. STUDY RECOMMENDATIONS
   - For each improvement area, provide specific study strategies
   - Include recommended resources or practice exercises

4. STRENGTHS & ENCOURAGEMENT
   - Highlight what the student did well
   - Provide motivational feedback

Format each section with clear headings and bullet points for readability.

Quiz Results:
"""
    
    # Add each question and the user's performance
    for i, q in enumerate(questions):
        prompt += f"\nQuestion {i+1}: {q['text']}\n"
        prompt += f"Type: {q['type']}\n"
        prompt += f"User Answer: {q.get('user_answer', 'No answer provided')}\n"
        prompt += f"Correct Answer: {q['correct_answer']}\n"
        
        if 'is_correct' in q:
            prompt += f"Result: {'Correct' if q['is_correct'] else 'Incorrect'}\n"
        else:
            prompt += "Result: Not graded (open-ended)\n"
    
    prompt += "\nProvide a comprehensive, well-structured report with clear section headings and detailed analysis in each section."
    
    try:
        # Determine provider based on model name
        if model_name.startswith("gpt-"):
            # Use OpenAI
            logger.info(f"Using OpenAI model: {model_name}")
            response = completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert educational analyst who provides detailed, well-structured learning insights with clear section headings and comprehensive analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500  # Increased token limit for more detailed response
            )
        else:
            # Use Ollama
            logger.info(f"Using Ollama model: {model_name}")
            response = completion(
                model=f"ollama/{model_name}",
                messages=[
                    {"role": "system", "content": "You are an expert educational analyst who provides detailed, well-structured learning insights with clear section headings and comprehensive analysis."},
                    {"role": "user", "content": prompt}
                ],
                api_base="http://localhost:11434",
                temperature=0.7,
                max_tokens=1500,  # Increased token limit for more detailed response
                stream=False
            )
        
        # Process the response to ensure proper formatting
        insight_text = response.choices[0].message.content.strip()
        
        # Ensure the response has proper section headings
        if "PERFORMANCE SUMMARY" not in insight_text and "Performance Summary" not in insight_text:
            # Add default section headings if they're missing
            formatted_insight = format_insights_with_sections(insight_text, correct_questions, total_questions)
            return formatted_insight
        
        return insight_text
    
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        # Fallback to structured analysis if API fails
        return generate_fallback_insights(questions, correct_questions, total_questions)

def format_insights_with_sections(text, correct_questions, total_questions):
    """Format insights with proper section headings if they're missing."""
    # Split the text into paragraphs
    paragraphs = text.split('\n\n')
    
    # Create a structured report
    formatted_text = f"""# PERFORMANCE SUMMARY

Overall Score: {correct_questions}/{total_questions} questions answered correctly.

{paragraphs[0] if paragraphs else 'The student shows areas for improvement based on the quiz results.'}

# KEY IMPROVEMENT AREAS

{paragraphs[1] if len(paragraphs) > 1 else '- Review the concepts covered in the incorrect questions.\n- Focus on understanding the underlying principles rather than memorizing facts.'}

# STUDY RECOMMENDATIONS

{paragraphs[2] if len(paragraphs) > 2 else '- Review the material related to the topics you struggled with.\n- Practice with additional examples to reinforce your understanding.\n- Consider creating flashcards for key concepts.'}

# STRENGTHS & ENCOURAGEMENT

{paragraphs[3] if len(paragraphs) > 3 else 'You\'ve shown good effort in taking this quiz. With focused study on the areas identified above, you can improve your understanding and performance. Keep up the good work!'}
"""
    return formatted_text

def generate_fallback_insights(questions, correct_questions, total_questions):
    """Generate a structured fallback insight report when API calls fail."""
    # Identify incorrect questions
    incorrect_questions = [q for q in questions if q.get('is_correct') is False]
    
    # Extract potential topics
    topics = set()
    for q in incorrect_questions:
        # Extract potential topics from questions
        if "topic" in q:
            topics.add(q["topic"])
        elif "category" in q:
            topics.add(q["category"])
        else:
            # Extract potential topic from question text
            words = q['text'].split()
            if len(words) > 5:
                potential_topic = ' '.join(words[:3]) + "..."
                topics.add(potential_topic)
    
    topics_str = ", ".join(topics) if topics else "the areas covered in the incorrect questions"
    
    # Create a structured report
    report = f"""# PERFORMANCE SUMMARY

Overall Score: {correct_questions}/{total_questions} questions answered correctly.

The quiz results indicate areas where additional study would be beneficial.

# KEY IMPROVEMENT AREAS

- Focus on improving understanding of: {topics_str}
- Pay attention to the specific concepts covered in the incorrect questions
- Review the fundamental principles related to these topics

# STUDY RECOMMENDATIONS

- Review the material related to {topics_str}
- Practice with additional examples to reinforce your understanding
- Consider creating flashcards for key concepts
- Seek additional resources that explain these topics from different perspectives

# STRENGTHS & ENCOURAGEMENT

You've shown good effort in taking this quiz. With focused study on the areas identified above, you can improve your understanding and performance. Remember that learning is a process, and each attempt helps build your knowledge.
"""
    return report
