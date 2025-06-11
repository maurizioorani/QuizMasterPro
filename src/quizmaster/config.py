from dataclasses import dataclass, field

@dataclass
class QuizConfig:
    """Configuration for quiz generation."""
    ollama_base_url: str = "http://localhost:11434"
    max_retries: int = 3
    retry_delay: float = 1.0
    pull_timeout: int = 1800
    default_temperature: float = 0.7
    cache_size: int = 100
    current_model: str = "llama3" # Default model for the application if nothing else is chosen
    openai_models: list = field(default_factory=lambda: ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"])
