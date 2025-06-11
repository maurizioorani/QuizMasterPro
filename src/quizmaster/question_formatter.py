from typing import Dict, Optional

class QuestionFormatter:
    def format_question(self, question_type: str, parsed_question: Dict) -> Optional[Dict]:
        """Formats the parsed question into the internal representation."""
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
