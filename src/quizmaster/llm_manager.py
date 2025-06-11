import os
import time
import logging
import requests
from typing import Optional, List
from litellm import completion
from .config import QuizConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, config: Optional[QuizConfig] = None):
        self.config = config or QuizConfig()

    def get_current_model(self) -> str:
        """Returns the current model from the configuration."""
        return self.config.current_model

    def set_model(self, model_name: str):
        """Sets the current model in the configuration."""
        self.config.current_model = model_name
        logger.info(f"LLM model set to: {model_name}")

    def get_available_models(self) -> List[str]:
        """Returns a list of predefined Ollama models plus OpenAI models from config."""
        # Predefined list of Ollama models to offer in the UI
        predefined_ollama_models = [
            "llama3.3:8b",
            "mistral:7b",
            "qwen2.5:7b",
            "deepseek-coder:6.7b",
            # Add other desired models here, e.g., "gemma2:9b"
        ]
        openai_models_from_config = self.config.openai_models
        
        combined_models = list(set(predefined_ollama_models + openai_models_from_config))
        logger.info(f"Offering models in UI: {combined_models}")
        return combined_models

    def get_openai_models(self) -> List[str]:
        """Returns the list of OpenAI models from the configuration."""
        return self.config.openai_models

    def get_local_models(self) -> List[str]:
        """Retrieves a list of locally available Ollama models."""
        try:
            response = requests.get(f"{self.config.ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            else:
                logger.error(f"Failed to get local models from Ollama: {response.status_code} - {response.text}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama to get local models: {str(e)}")
            return []

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
            estimated_chars = max_tokens * 4
            if len(content) > estimated_chars:
                return content[:estimated_chars] + "..."
        return content

    def make_llm_request(self, prompt: str, model: str, openai_models: list, max_tokens: int = 500, temperature: Optional[float] = None) -> Optional[str]:
        """Make LLM request with retry logic using litellm."""
        temperature = temperature or self.config.default_temperature
        
        logger.info(f"Making LLM request with model: {model}")
        
        for attempt in range(self.config.max_retries):
            try:
                if model in openai_models:
                    logger.info(f"Using OpenAI API with model {model}")
                    response = completion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                else:
                    logger.info(f"Using Ollama API with model {model}")
                    response = completion(
                        model=f"ollama/{model}",
                        messages=[{"role": "user", "content": prompt}],
                        api_base=self.config.ollama_base_url,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False
                    )
                
                logger.info(f"LLM request successful with model {model}")
                return response.choices[0].message.content
                
            except RuntimeError as e:
                if "cannot schedule new futures after interpreter shutdown" in str(e):
                    logger.error("Interpreter shutdown detected - restart required")
                    return None
                logger.error(f"RuntimeError in LLM request: {str(e)}")
                raise
                
            except Exception as e:
                logger.warning(f"LLM request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {self.config.retry_delay * (attempt + 1)} seconds...")
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed for model {model}")
                    
        return None

    def test_ollama_connection(self) -> bool: # Removed model argument, will use current_model
        """Test connection to Ollama server with retries using the current model."""
        current_model_to_test = self.get_current_model()
        if not current_model_to_test or current_model_to_test in self.config.openai_models:
            logger.info("Current model is OpenAI or not set; Ollama connection test skipped for this model.")
            # Optionally, try a default Ollama model if none is set, or just return True if not an Ollama model.
            # For now, let's try a generic connection test if current model is not Ollama.
            try:
                requests.get(self.config.ollama_base_url, timeout=5)
                logger.info("Successfully connected to Ollama base URL.")
                return True
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to Ollama base URL: {str(e)}")
                return False

        logger.info(f"Testing Ollama connection with model: {current_model_to_test}")
        for attempt in range(self.config.max_retries):
            try:
                response = completion(
                    model=f"ollama/{current_model_to_test}",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base=self.config.ollama_base_url,
                    max_tokens=10,
                    temperature=0.1
                )
                logger.info(f"Successfully connected to Ollama with model {current_model_to_test}")
                return True
            except Exception as e:
                logger.warning(f"Ollama connection attempt {attempt + 1} with model {current_model_to_test} failed: {str(e)}")
                if "model not found" in str(e).lower(): # Specific check for model not found
                    logger.error(f"Model '{current_model_to_test}' not found on Ollama server. Pull it first.")
                    return False # No point retrying if model is not there
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    
        logger.error(f"Failed to connect to Ollama with model {current_model_to_test} after all retries")
        return False

    def is_model_available(self, model_name: str) -> bool: # This is effectively get_local_models for a specific model
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
                    
        logger.error("Failed to connect to Ollama after all retries")
        return False

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            logger.info(f"Pulling model {model_name} from Ollama...")
            
            response = requests.post(
                f"{self.config.ollama_base_url}/api/pull",
                json={"name": model_name},
                timeout=self.config.pull_timeout,
                stream=True
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False
