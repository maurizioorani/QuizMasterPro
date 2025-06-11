def get_question_prompt(question_type: str, difficulty: str, context: str) -> str:
    """Constructs the prompt for generating a question."""
    base_prompt = f"""
You are an expert quiz question generator. Create a {difficulty} difficulty {question_type} question based on the following context.
Ensure the question tests deep understanding, not just recall.

Context:
{context}
"""
    
    if question_type == "Multiple Choice":
        return base_prompt + """
- Provide 4 plausible options with only one correct answer.
- Make incorrect options realistic but clearly wrong.

Format response as valid JSON:
{
  "question": "Question text",
  "options": [
    {"letter": "A", "text": "Option A"},
    {"letter": "B", "text": "Option B"},
    {"letter": "C", "text": "Option C"},
    {"letter": "D", "text": "Option D"}
  ],
  "correct_answer": "A",
  "explanation": "Detailed explanation"
}"""
    elif question_type == "Open-Ended":
        return base_prompt + """
Format response as valid JSON:
{
  "question": "Question text",
  "sample_answer": "A detailed sample answer",
  "key_points": ["Point 1", "Point 2"],
  "explanation": "Detailed explanation"
}"""
    elif question_type == "True/False":
        return base_prompt + """
Format response as valid JSON:
{
  "statement": "Statement text",
  "answer": true,
  "explanation": "Detailed explanation"
}"""
    elif question_type == "Fill-in-the-Blank":
        return base_prompt + """
Format response as valid JSON:
{
  "question": "Question text with [BLANK] placeholder",
  "answer": "The correct word or phrase for the blank",
  "explanation": "Detailed explanation"
}"""
    return base_prompt
