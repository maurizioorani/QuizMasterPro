import random
from typing import List, Dict, Optional

class ContextManager:
    def __init__(self, context_data: List[Dict]):
        self.context_data = context_data

    def select_relevant_context(self, question_type: str, focus_topics: Optional[List[str]] = None) -> str:
        """Select relevant context for question generation, prioritizing focus_topics."""
        if not self.context_data:
            return ""
        
        if focus_topics:
            relevant_contexts_set = set() # Use a set to store content to avoid duplicates
            for topic in focus_topics:
                topic_lower = topic.lower()
                for ctx in self.context_data:
                    if topic_lower in ctx['content'].lower():
                        relevant_contexts_set.add(ctx['content'])
            
            if relevant_contexts_set:
                # Convert set to list, maybe shuffle or take top N based on some criteria
                # For now, take up to 3 relevant unique contexts
                return "\n\n".join(list(relevant_contexts_set)[:3])
        
        # Fallback logic if no focus_topics or no matches found
        if question_type == "Multiple Choice":
            concepts = [ctx for ctx in self.context_data if ctx['type'] in ['Key Definition', 'Important Fact']]
            if concepts:
                return "\n\n".join([ctx['content'] for ctx in concepts[:2]])
        
        elif question_type == "True/False":
            facts = [ctx for ctx in self.context_data if ctx['type'] == 'Important Fact']
            if facts:
                return facts[0]['content']
        
        elif question_type == "Fill-in-the-Blank":
            definitions = [ctx for ctx in self.context_data if ctx['type'] == 'Key Definition']
            if definitions:
                return definitions[0]['content']
        
        selected = random.sample(self.context_data, min(3, len(self.context_data)))
        return "\n\n".join([ctx['content'] for ctx in selected])
