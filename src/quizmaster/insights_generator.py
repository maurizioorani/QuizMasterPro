import json
import os
from litellm import completion
import time

def generate_insights(questions, model_name: str):
    """
    Generate personalized insights based on quiz performance using Ollama.
    
    Args:
        questions: List of question dictionaries with user answers and correctness
        model_name: The name of the Ollama model to use for insights generation
    
    Returns:
        String containing personalized insights report
    """
    # Prepare the prompt
    prompt = (
        "You are an expert tutor analyzing a student's quiz results. "
        "Based on the following questions and answers, provide:\n"
        "1. A summary of the student's overall performance\n"
        "2. Identification of 2-3 key topics the student needs to improve\n"
        "3. Specific study recommendations for each identified topic\n"
        "4. Encouraging feedback highlighting strengths\n\n"
        "Quiz Results:\n"
    )
    
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
    
    prompt += "\nPlease provide a comprehensive learning insights report:"
    
    try:
        # Call the Ollama API via LiteLLM
        response = completion(
            model=f"ollama/{model_name}",
            messages=[
                {"role": "system", "content": "You are a helpful and encouraging tutor."},
                {"role": "user", "content": prompt}
            ],
            api_base="http://localhost:11434",  # Local Ollama server
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        # Fallback to simple analysis if Ollama fails
        incorrect_questions = [q for q in questions if q.get('is_correct') is False]
        
        if incorrect_questions:
            topics = set()
            for q in incorrect_questions:
                # Extract potential topics from questions
                if "topic" in q:
                    topics.add(q["topic"])
                elif "category" in q:
                    topics.add(q["category"])
            
            if topics:
                topics_str = ", ".join(topics)
                return (
                    f"Based on your quiz results, you should focus on improving these topics: {topics_str}.\n"
                    "Review the material related to these areas and try the quiz again."
                )
        
        return (
            "Review your answers and focus on areas where you were unsure. "
            "Consider re-reading the study materials related to the quiz questions."
        )
