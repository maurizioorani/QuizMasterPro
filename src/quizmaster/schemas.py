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
