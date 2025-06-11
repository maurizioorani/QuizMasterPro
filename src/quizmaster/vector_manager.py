import os
import json
import re
import hashlib
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from .config import QuizConfig # Added for QuizConfig type hint
import requests # Ensure requests is imported if not already (it's used in _initialize_contextgem)

# Set up logging
logger = logging.getLogger(__name__)

try:
    from contextgem import (
        Document,
        DocumentLLM,
        Aspect,
        StringConcept,
        JsonObjectConcept,
        DateConcept,
        NumericalConcept,
        BooleanConcept,
        DocumentPipeline,
        Image,
        image_to_base64
    )
    CONTEXTGEM_AVAILABLE = True
    logger.info("ContextGem library loaded successfully")
except ImportError as e:
    CONTEXTGEM_AVAILABLE = False
    logger.error(f"ContextGem not available: {e}. Please install with: pip install contextgem")

from database_manager import DatabaseManager

class VectorManager:
    """Enhanced Vector Manager using ContextGem for intelligent document processing"""
    
    def __init__(self, config: Optional[QuizConfig] = None):
        self.embeddings_available = CONTEXTGEM_AVAILABLE
        self.db = DatabaseManager()
        self.config = config # Store the passed config
        self.llm: Optional[DocumentLLM] = None # Ensure self.llm is defined
        
        # Initialize ContextGem components
        if CONTEXTGEM_AVAILABLE:
            self._initialize_contextgem()
            self._create_enhanced_documents_table()
            self._setup_extraction_pipeline()
        else:
            logger.warning("Vector storage disabled - ContextGem not available")

    def update_llm_configuration(self):
        """
        Re-initializes the ContextGem DocumentLLM instance based on the current
        configuration. This should be called if the global model config changes.
        """
        if CONTEXTGEM_AVAILABLE: # Only try if library is there
            logger.info("VectorManager attempting to update LLM configuration for ContextGem.")
            if not self.config:
                logger.warning("VectorManager has no config set; LLM re-initialization might use defaults.")
            self._initialize_contextgem()
        else:
            logger.warning("VectorManager: ContextGem not available, cannot update LLM configuration.")


    def _initialize_contextgem(self):
        """Initialize ContextGem DocumentLLM instance, preferring UI selected model if OpenAI."""
        self.llm = None # Reset llm instance before re-initializing
        self.embeddings_available = False # Reset status

        try:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

            # Priority 1: Use globally selected OpenAI model if available and an API key is present
            if self.config and self.config.current_model and self.config.current_model in self.config.openai_models:
                if openai_api_key:
                    cg_model_name = self.config.current_model
                    if not cg_model_name.startswith("openai/"):
                        cg_model_name = f"openai/{cg_model_name}"
                    
                    self.llm = DocumentLLM(model=cg_model_name, api_key=openai_api_key)
                    logger.info(f"VectorManager using UI selected OpenAI model '{cg_model_name}' for ContextGem.")
                    self.embeddings_available = True # Mark as available since we successfully configured an LLM
                    return
                else:
                    logger.error(f"VectorManager: UI selected OpenAI model {self.config.current_model} but OPENAI_API_KEY not found. Cannot use this model.")
                    self.llm = None
                    self.embeddings_available = False # Explicitly mark as unavailable
                    logger.warning("VectorManager: ContextGem LLM set to None due to missing API key for selected OpenAI model.")
                    return # Explicitly stop further model search

            # Priority 2: Try Ollama
            ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            try:
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    models_data = response.json()
                    available_ollama_models = [model['name'] for model in models_data.get('models', [])]
                    logger.info(f"VectorManager: Available local Ollama models: {available_ollama_models}")

                    selected_ollama_model = None # Ensure it's reset
                    ui_selected_model = self.config.current_model if self.config else None
                    logger.info(f"VectorManager: UI selected model (config.current_model): {ui_selected_model}")

                    # Check if UI selected model is an Ollama model and is available
                    # Assuming openai_models list is also in self.config if config is present
                    is_ui_model_openai = self.config and self.config.openai_models and ui_selected_model in self.config.openai_models

                    if ui_selected_model and not is_ui_model_openai and ui_selected_model in available_ollama_models:
                        selected_ollama_model = ui_selected_model
                        logger.info(f"VectorManager: UI selected Ollama model '{selected_ollama_model}' is available locally.")
                    else:
                        if ui_selected_model and not is_ui_model_openai:
                             logger.info(f"VectorManager: UI selected Ollama model '{ui_selected_model}' is NOT available locally or not an Ollama model. Checking preferred list.")
                        # Fallback to preferred Ollama models
                        preferred_ollama_models = ["mistral:7b", "qwen2.5:7b", "llama3.3:8b", "deepseek-coder:6.7b"]
                        for model in preferred_ollama_models:
                            if model in available_ollama_models:
                                selected_ollama_model = model
                                logger.info(f"VectorManager: Found preferred Ollama model '{selected_ollama_model}' locally.")
                                break
                        if not selected_ollama_model and available_ollama_models:
                            selected_ollama_model = available_ollama_models[0]
                            logger.info(f"VectorManager: Using first available Ollama model '{selected_ollama_model}' locally.")
                    
                    if selected_ollama_model:
                        self.llm = DocumentLLM(model=f"ollama/{selected_ollama_model}", api_base=ollama_base_url)
                        logger.info(f"VectorManager using Ollama model '{selected_ollama_model}' for ContextGem.")
                        self.embeddings_available = True
                        return
            except Exception as e:
                logger.info(f"VectorManager: Ollama not available or error selecting model: {e}")

            # Priority 3: Fallback to hardcoded OpenAI model if key exists
            if openai_api_key:
                self.llm = DocumentLLM(model="openai/gpt-4o-mini", api_key=openai_api_key)
                logger.info("VectorManager using fallback OpenAI model 'gpt-4o-mini' for ContextGem.")
                self.embeddings_available = True
                return
            
            # Priority 4: Fallback to Anthropic
            if anthropic_api_key:
                self.llm = DocumentLLM(model="anthropic/claude-3-5-sonnet", api_key=anthropic_api_key)
                logger.info("VectorManager using fallback Anthropic model 'claude-3-5-sonnet' for ContextGem.")
                self.embeddings_available = True
                return

            logger.warning("VectorManager: ContextGem LLM could not be initialized. No suitable model configuration found. Embeddings/ContextGem features might be limited.")
            self.embeddings_available = False # Ensure this is set if no LLM is configured
            self.llm = None

        except Exception as e:
            logger.error(f"VectorManager: Failed to initialize ContextGem LLM: {e}", exc_info=True)
            self.embeddings_available = False
            self.llm = None

    def _create_enhanced_documents_table(self):
        """Create enhanced table for storing documents with extracted concepts"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents_enhanced (
                        id TEXT PRIMARY KEY,
                        filename TEXT,
                        format TEXT,
                        raw_content TEXT,
                        processed_content TEXT,
                        extracted_concepts JSONB,
                        extracted_aspects JSONB,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better search performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_documents_enhanced_filename 
                    ON documents_enhanced(filename)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_documents_enhanced_format 
                    ON documents_enhanced(format)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_documents_enhanced_concepts 
                    ON documents_enhanced USING GIN (extracted_concepts)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_documents_enhanced_aspects 
                    ON documents_enhanced USING GIN (extracted_aspects)
                ''')
            conn.commit()

    def _setup_extraction_pipeline(self):
        """Setup simple extraction pipeline for fast document processing"""
        # Expanded default concepts for richer extraction
        self.default_concepts = [
            StringConcept(name="Main Topic", description="Primary topic or subject of the document", singular_occurrence=True, add_references=False),
            StringConcept(name="Document Type", description="Type or category of the document (e.g., report, article, manual)", singular_occurrence=True, add_references=False),
            StringConcept(name="Key Arguments", description="Main arguments or theses presented", singular_occurrence=False, add_references=False),
            StringConcept(name="Conclusions", description="Main conclusions or outcomes reported", singular_occurrence=False, add_references=False),
            StringConcept(name="Technical Terms", description="Specific technical terms or jargon used", singular_occurrence=False, add_references=False),
            StringConcept(name="Key People", description="Important individuals mentioned", singular_occurrence=False, add_references=False),
            StringConcept(name="Key Organizations", description="Important organizations or entities mentioned", singular_occurrence=False, add_references=False),
            StringConcept(name="Locations", description="Geographical places mentioned", singular_occurrence=False, add_references=False),
            DateConcept(name="Key Dates", description="Significant dates or time periods mentioned", singular_occurrence=False, add_references=False),
            NumericalConcept(name="Key Figures", description="Important numbers or statistics presented", singular_occurrence=False, add_references=False)
        ]
        
        # Enhanced pipeline for aspects
        self.simple_pipeline = DocumentPipeline()
        key_info_concepts = [
            StringConcept(name="Key Topics", description="Main topics covered in the document", add_references=False, singular_occurrence=False),
            StringConcept(name="Summary Points", description="Brief summary points or takeaways", add_references=False, singular_occurrence=False)
        ]
        # Add some of the default concepts to Key Information aspect if relevant
        for concept in self.default_concepts:
            if concept.name in ["Key Arguments", "Conclusions", "Technical Terms"]:
                key_info_concepts.append(StringConcept(name=concept.name, description=concept.description, add_references=False, singular_occurrence=False))

        self.simple_pipeline.aspects = [
            Aspect(
                name="Key Information",
                description="Core informational content of the document",
                concepts=key_info_concepts
            ),
            Aspect(
                name="Contextual Elements",
                description="People, places, and organizations providing context",
                concepts=[
                    StringConcept(name="Mentioned People", description="Individuals discussed", add_references=False, singular_occurrence=False),
                    StringConcept(name="Mentioned Organizations", description="Entities discussed", add_references=False, singular_occurrence=False),
                    StringConcept(name="Mentioned Locations", description="Places discussed", add_references=False, singular_occurrence=False)
                ]
            ),
            Aspect(
                name="Document Purpose",
                description="The intended purpose and audience of the document",
                concepts=[
                    StringConcept(name="Stated Purpose", description="The explicit or implicit purpose of the document", add_references=False, singular_occurrence=True),
                    StringConcept(name="Target Audience", description="The intended audience for the document", add_references=False, singular_occurrence=True)
                ]
            )
        ]

    def store_document(self, processed_doc: Dict) -> str:
        """Store document with fast ContextGem extraction in PostgreSQL"""
        if not self.embeddings_available:
            raise ConnectionError("ContextGem not available. Please install contextgem and configure API keys.")
        
        # Get the processed content - it should already be extracted text
        content = processed_doc["content"]
        
        # Validate that we have proper text content
        if not content or not isinstance(content, str):
            raise ValueError("Invalid content: expected processed text string from document processor")
        
        # Check if we somehow still have binary PDF data (shouldn't happen with fixed processor)
        if content.startswith("%PDF-") and re.search(r'stream|endobj|xref|startxref', content[:1000]):
            logger.error("CRITICAL: Binary PDF data detected in vector manager - document processor failed to extract text properly")
            raise ValueError("Document processor failed to extract text from PDF. Please check document_processor.py")
        
        # Clean the content of any remaining problematic characters
        if content:
            content = content.replace('\x00', '')  # Remove null bytes
            content = content.replace('\0', '')    # Remove null characters
            # Also clean other problematic characters while preserving normal text formatting
            content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
        
        # Create a consistent ID based on the original document
        original_content = processed_doc.get("original_content", content)
        doc_id = hashlib.sha256(original_content.encode()).hexdigest()
        
        # Store document immediately, then try extraction in background
        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        INSERT INTO documents_enhanced (
                            id, filename, format, raw_content, processed_content,
                            extracted_concepts, extracted_aspects, metadata, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO UPDATE SET
                            filename = EXCLUDED.filename,
                            format = EXCLUDED.format,
                            raw_content = EXCLUDED.raw_content,
                            processed_content = EXCLUDED.processed_content,
                            metadata = EXCLUDED.metadata,
                            updated_at = CURRENT_TIMESTAMP
                    ''', (
                        doc_id,
                        processed_doc["filename"],
                        processed_doc["format"],
                        content,
                        content,
                        json.dumps({}),  # Start with empty concepts
                        json.dumps({}),  # Start with empty aspects
                        json.dumps(processed_doc.get("metadata", {}))
                    ))
                conn.commit()
            
            logger.info(f"Document {doc_id} stored successfully")
            
            # First, check if the document processor already extracted concepts
            extracted_concepts = {}
            
            # Check if document processor provided concepts
            if "concepts" in processed_doc and processed_doc["concepts"]:
                logger.info(f"Using concepts from document processor: {len(processed_doc['concepts'])} concepts found")
                
                # Convert document processor concepts to ContextGem format
                for concept in processed_doc["concepts"]:
                    concept_name = concept.get("concept_name", "Extracted Concept")
                    concept_content = concept.get("content", "")
                    
                    if concept_content and concept_content.strip():
                        if concept_name not in extracted_concepts:
                            extracted_concepts[concept_name] = {"items": []}
                        
                        extracted_concepts[concept_name]["items"].append({
                            "value": concept_content,
                            "references": [],
                            "justification": concept.get("metadata", {}).get("extraction_method", "Document processor extraction")
                        })
                
                logger.info(f"Converted document processor concepts: {list(extracted_concepts.keys())}")
            
            # If no concepts from document processor, try ContextGem extraction
            if not extracted_concepts:
                try:
                    logger.info(f"No concepts from document processor, trying ContextGem extraction for {doc_id}")
                    extracted_concepts = self._fast_concept_extraction(content)
                    logger.info(f"ContextGem extraction result: {extracted_concepts}")
                    
                except Exception as extraction_error:
                    logger.warning(f"ContextGem extraction failed for {doc_id}: {str(extraction_error)}")
            
            # If still no concepts, use fallback
            if not extracted_concepts:
                logger.info(f"Using fallback concept extraction for {doc_id}")
                try:
                    extracted_concepts = self._create_fallback_concepts(content, processed_doc.get("filename", "Unknown"))
                    logger.info(f"Fallback extraction result: {extracted_concepts}")
                except Exception as fallback_error:
                    logger.error(f"Fallback concept creation failed: {str(fallback_error)}")
                    # Create minimal concepts as last resort
                    extracted_concepts = {
                        "Main Topic": {
                            "items": [{"value": "Document Content", "references": [], "justification": "Default fallback"}]
                        },
                        "Document Type": {
                            "items": [{"value": "Text Document", "references": [], "justification": "Default fallback"}]
                        }
                    }
            
            # Always update with some concepts
            try:
                with self.db._get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute('''
                            UPDATE documents_enhanced 
                            SET extracted_concepts = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        ''', (json.dumps(extracted_concepts), doc_id))
                    conn.commit()
                logger.info(f"Successfully updated document {doc_id} with concepts: {list(extracted_concepts.keys())}")
            except Exception as update_error:
                logger.error(f"Failed to update document {doc_id} with concepts: {str(update_error)}")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to store document: {str(e)}")
            raise ConnectionError(f"Failed to store document: {str(e)}")

    def _fast_concept_extraction(self, content: str) -> Dict:
        """Fast concept extraction with minimal processing"""
        try:
            # Limit content size for speed
            if len(content) > 20000:  # Smaller limit for speed
                content = content[:20000] + "..."
            
            # Create minimal document
            cg_doc = Document(raw_text=content)
            
            # Add all default concepts for a richer fast extraction
            cg_doc.add_concepts(self.default_concepts)
            
            # Quick extraction with timeout
            # This will attempt to extract all concepts defined in self.default_concepts
            extracted_doc = self.llm.extract_concepts_from_document(cg_doc)
            
            # Process results
            extracted_concepts = {}
            concepts_list = []
            
            if hasattr(extracted_doc, 'concepts'):
                concepts_list = extracted_doc.concepts
            elif isinstance(extracted_doc, list):
                concepts_list = extracted_doc
            else:
                concepts_list = getattr(cg_doc, 'concepts', [])
            
            for concept in concepts_list:
                if hasattr(concept, 'name') and hasattr(concept, 'extracted_items'):
                    if concept.extracted_items:  # Only add if items exist
                        extracted_concepts[concept.name] = {
                            "items": [
                                {
                                    "value": getattr(item, 'value', str(item)),
                                    "references": [],  # Skip references for speed
                                    "justification": None  # Skip justification for speed
                                }
                                for item in concept.extracted_items
                            ]
                        }
            
            return extracted_concepts
            
        except Exception as e:
            logger.warning(f"Fast extraction failed: {str(e)}")
            return {}

    def _process_large_document(self, content: str) -> tuple:
        """Process large documents by chunking"""
        import time
        
        # Split content into chunks
        max_chunk_size = 30000  # Conservative size for processing
        chunks = []
        
        # Try to split by paragraphs first
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_chunk_size:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If chunks are still too large, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                sentences = chunk.split('. ')
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_size:
                        current_chunk += sentence + '. '
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        logger.info(f"Processing document in {len(final_chunks)} chunks")
        
        # Process each chunk and merge results
        all_concepts = {}
        all_aspects = {}
        
        for i, chunk in enumerate(final_chunks):
            logger.info(f"Processing chunk {i+1}/{len(final_chunks)}")
            try:
                # Add delay to avoid rate limiting
                if i > 0:
                    time.sleep(2)  # 2 second delay between chunks
                
                chunk_concepts, chunk_aspects = self._process_document_chunk(chunk)
                
                # Merge concepts
                for concept_name, concept_data in chunk_concepts.items():
                    if concept_name not in all_concepts:
                        all_concepts[concept_name] = {"items": []}
                    
                    # Add unique items
                    existing_values = {item["value"] for item in all_concepts[concept_name]["items"]}
                    for item in concept_data["items"]:
                        if item["value"] not in existing_values:
                            all_concepts[concept_name]["items"].append(item)
                
                # Merge aspects
                for aspect_name, aspect_data in chunk_aspects.items():
                    if aspect_name not in all_aspects:
                        all_aspects[aspect_name] = {"items": [], "concepts": {}}
                    
                    # Add unique aspect items
                    existing_values = {item["value"] for item in all_aspects[aspect_name]["items"]}
                    for item in aspect_data["items"]:
                        if item["value"] not in existing_values:
                            all_aspects[aspect_name]["items"].append(item)
                    
                    # Merge aspect concepts
                    for concept_name, concept_data in aspect_data.get("concepts", {}).items():
                        if concept_name not in all_aspects[aspect_name]["concepts"]:
                            all_aspects[aspect_name]["concepts"][concept_name] = {"items": []}
                        
                        existing_values = {item["value"] for item in all_aspects[aspect_name]["concepts"][concept_name]["items"]}
                        for item in concept_data["items"]:
                            if item["value"] not in existing_values:
                                all_aspects[aspect_name]["concepts"][concept_name]["items"].append(item)
                
            except Exception as e:
                logger.warning(f"Failed to process chunk {i+1}: {str(e)}")
                continue
        
        return all_concepts, all_aspects

    def _process_document_chunk(self, content: str) -> tuple:
        """Process a single document chunk with ContextGem"""
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create ContextGem Document
                cg_doc = Document(raw_text=content)
                
                # First extract concepts only (simpler pipeline)
                cg_doc.add_concepts(self.default_concepts)
                
                # Extract concepts first
                extracted_doc = self.llm.extract_concepts_from_document(cg_doc)
                
                # Debug logging
                logger.debug(f"Extracted doc type: {type(extracted_doc)}")
                logger.debug(f"Extracted doc attributes: {dir(extracted_doc) if hasattr(extracted_doc, '__dict__') else 'No attributes'}")
                
                # Process extracted concepts - handle different return types
                extracted_concepts = {}
                
                # Check if extracted_doc is the document itself or has concepts attribute
                if hasattr(extracted_doc, 'concepts'):
                    concepts_list = extracted_doc.concepts
                elif isinstance(extracted_doc, list):
                    # If it returns a list, it might be the concepts directly
                    concepts_list = extracted_doc
                else:
                    # Try to get concepts from the original document
                    concepts_list = getattr(cg_doc, 'concepts', [])
                
                logger.debug(f"Concepts list type: {type(concepts_list)}, length: {len(concepts_list) if hasattr(concepts_list, '__len__') else 'unknown'}")
                
                for concept in concepts_list:
                    logger.debug(f"Processing concept: {type(concept)}, name: {getattr(concept, 'name', 'no name')}")
                    if hasattr(concept, 'name') and hasattr(concept, 'extracted_items'):
                        extracted_concepts[concept.name] = {
                            "items": [
                                {
                                    "value": getattr(item, 'value', str(item)),
                                    "references": getattr(item, 'reference_sentences', []) or getattr(item, 'reference_paragraphs', []),
                                    "justification": getattr(item, 'justification', None)
                                }
                                for item in concept.extracted_items
                            ]
                        }
                
                # Now extract aspects separately - with better error handling
                extracted_aspects = {}
                try:
                    # Create new document for aspects
                    cg_doc_aspects = Document(raw_text=content)
                    cg_doc_aspects.assign_pipeline(self.default_pipeline)
                    
                    # Extract aspects
                    extracted_doc_aspects = self.llm.extract_aspects_from_document(cg_doc_aspects)
                    
                    # Handle different return types for aspects
                    aspects_list = []
                    if hasattr(extracted_doc_aspects, 'aspects'):
                        aspects_list = extracted_doc_aspects.aspects
                    elif isinstance(extracted_doc_aspects, list):
                        aspects_list = extracted_doc_aspects
                    else:
                        # Try to get aspects from the document
                        aspects_list = getattr(cg_doc_aspects, 'aspects', [])
                    
                    for aspect in aspects_list:
                        if hasattr(aspect, 'name') and hasattr(aspect, 'extracted_items'):
                            aspect_data = {
                                "items": [
                                    {
                                        "value": getattr(item, 'value', str(item)),
                                        "references": getattr(item, 'reference_sentences', []) or getattr(item, 'reference_paragraphs', [])
                                    }
                                    for item in aspect.extracted_items
                                ],
                                "concepts": {}
                            }
                            
                            # Extract aspect concepts
                            if hasattr(aspect, 'concepts'):
                                for concept in aspect.concepts:
                                    if hasattr(concept, 'name') and hasattr(concept, 'extracted_items'):
                                        aspect_data["concepts"][concept.name] = {
                                            "items": [
                                                {
                                                    "value": getattr(item, 'value', str(item)),
                                                    "references": getattr(item, 'reference_sentences', []) or getattr(item, 'reference_paragraphs', []),
                                                    "justification": getattr(item, 'justification', None)
                                                }
                                                for item in concept.extracted_items
                                            ]
                                        }
                            
                            extracted_aspects[aspect.name] = aspect_data
                
                except Exception as aspect_error:
                    logger.warning(f"Failed to extract aspects: {str(aspect_error)}")
                    # Continue with just concepts
                
                logger.info(f"Successfully extracted {len(extracted_concepts)} concepts and {len(extracted_aspects)} aspects")
                return extracted_concepts, extracted_aspects
                
            except Exception as e:
                error_msg = str(e).lower()
                logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}")
                
                if "context window" in error_msg or "token" in error_msg:
                    logger.warning(f"Attempt {attempt + 1}: Context window exceeded, content too large")
                    if attempt < max_retries - 1:
                        # Try with smaller content
                        content = content[:len(content)//2]  # Reduce content size by half
                        logger.info(f"Reducing content size to {len(content)} characters")
                        continue
                elif "rate limit" in error_msg or "too many requests" in error_msg:
                    logger.warning(f"Attempt {attempt + 1}: Rate limit hit, waiting...")
                    if attempt < max_retries - 1:
                        time.sleep(10 * (attempt + 1))  # Exponential backoff
                        continue
                elif "'list' object has no attribute" in str(e):
                    logger.warning(f"Attempt {attempt + 1}: ContextGem API returned unexpected format")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                        continue
                
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    # Final attempt failed, return empty results
                    logger.error(f"All attempts failed for chunk processing: {str(e)}")
                    return {}, {}
        
        return {}, {}

    def _create_fallback_concepts(self, content: str, filename: str) -> Dict:
        """Create simple fallback concepts when ContextGem extraction fails"""
        try:
            # Extract simple concepts from content
            words = content.split()
            
            # Get first few meaningful words for main topic
            meaningful_words = []
            for word in words[:50]:  # Look at first 50 words
                word = word.strip('.,!?;:"()[]{}').lower()
                if len(word) > 3 and word.isalpha():  # Only meaningful words
                    meaningful_words.append(word.title())
                if len(meaningful_words) >= 5:
                    break
            
            main_topic = " ".join(meaningful_words[:3]) if meaningful_words else "Unknown Topic"
            
            # Determine document type from filename or content
            doc_type = "Document"
            filename_lower = filename.lower()
            
            if any(word in filename_lower for word in ['python', 'programming', 'code']):
                doc_type = "Programming Guide"
            elif any(word in filename_lower for word in ['web', 'html', 'css', 'javascript']):
                doc_type = "Web Development"
            elif any(word in filename_lower for word in ['report', 'analysis']):
                doc_type = "Report"
            elif any(word in filename_lower for word in ['manual', 'guide', 'tutorial']):
                doc_type = "Manual"
            elif filename_lower.endswith('.pdf'):
                doc_type = "PDF Document"
            
            # Create fallback concepts
            fallback_concepts = {
                "Main Topic": {
                    "items": [
                        {
                            "value": main_topic,
                            "references": [],
                            "justification": "Extracted from document content"
                        }
                    ]
                },
                "Document Type": {
                    "items": [
                        {
                            "value": doc_type,
                            "references": [],
                            "justification": "Determined from filename and content analysis"
                        }
                    ]
                }
            }
            
            logger.info(f"Created fallback concepts: {list(fallback_concepts.keys())}")
            return fallback_concepts
            
        except Exception as e:
            logger.error(f"Failed to create fallback concepts: {str(e)}")
            return {}

    def list_documents(self) -> List[Dict]:
        """List all stored documents with metadata and extraction summaries"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT id, filename, format, metadata, extracted_concepts, created_at
                    FROM documents_enhanced
                    ORDER BY created_at DESC
                ''')
                results = cursor.fetchall()
                
                docs_list = []
                for row in results:
                    doc_info = {
                        "id": row[0],
                        "filename": row[1] or "Unknown Document",
                        "format": row[2] or "Unknown",
                        "metadata": row[3] or {},
                        "created_at": row[5].isoformat() if row[5] else None
                    }
                    
                    # Add summary of extracted concepts
                    if row[4] and isinstance(row[4], dict) and len(row[4]) > 0:  # extracted_concepts
                        concepts = row[4]
                        
                        # Look for Main Topic (new) or Main Subject (legacy)
                        main_topic_data = (
                            concepts.get("Main Topic") or 
                            concepts.get("Main Subject") or 
                            {}
                        )
                        main_topic_value = "Not processed yet"
                        if main_topic_data and main_topic_data.get("items"):
                            main_topic_value = main_topic_data["items"][0].get("value", "Unknown")
                        
                        # Look for Document Type
                        doc_type_data = concepts.get("Document Type", {})
                        doc_type_value = "Unknown"
                        if doc_type_data and doc_type_data.get("items"):
                            doc_type_value = doc_type_data["items"][0].get("value", "Unknown")
                        
                        # Determine if extraction was successful
                        has_meaningful_concepts = any(
                            concept_data.get("items") and len(concept_data["items"]) > 0 
                            and any(item.get("value", "").strip() for item in concept_data["items"])
                            for concept_data in concepts.values() 
                            if isinstance(concept_data, dict)
                        )
                        
                        # Simplified status determination - if we have meaningful concepts, it's completed
                        extraction_status = "completed" if has_meaningful_concepts else "pending"
                        
                        doc_info["extracted_summary"] = {
                            "concept_count": len(concepts),
                            "main_topic": main_topic_value,
                            "document_type": doc_type_value,
                            "extraction_status": extraction_status
                        }
                    else:
                        # Try to trigger concept extraction for old documents without concepts
                        try:
                            logger.info(f"Triggering background concept extraction for document {row[0]}")
                            # Get the document content and try extraction
                            doc_data = self.get_document(row[0])
                            if doc_data and doc_data.get("content"):
                                # Use fallback concept creation immediately
                                fallback_concepts = self._create_fallback_concepts(
                                    doc_data["content"], 
                                    doc_data.get("filename", "Unknown")
                                )
                                
                                if fallback_concepts:
                                    # Update the database immediately
                                    with self.db._get_connection() as conn:
                                        with conn.cursor() as cursor:
                                            cursor.execute('''
                                                UPDATE documents_enhanced 
                                                SET extracted_concepts = %s, updated_at = CURRENT_TIMESTAMP
                                                WHERE id = %s
                                            ''', (json.dumps(fallback_concepts), row[0]))
                                        conn.commit()
                                    
                                    # Return the updated status
                                    main_topic = fallback_concepts.get("Main Topic", {}).get("items", [{}])[0].get("value", "Unknown")
                                    doc_type = fallback_concepts.get("Document Type", {}).get("items", [{}])[0].get("value", "Unknown")
                                    
                                    doc_info["extracted_summary"] = {
                                        "concept_count": len(fallback_concepts),
                                        "main_topic": main_topic,
                                        "document_type": doc_type,
                                        "extraction_status": "completed"
                                    }
                                else:
                                    doc_info["extracted_summary"] = {
                                        "concept_count": 0,
                                        "main_topic": "No content available",
                                        "document_type": "Unknown",
                                        "extraction_status": "failed"
                                    }
                            else:
                                doc_info["extracted_summary"] = {
                                    "concept_count": 0,
                                    "main_topic": "No content available",
                                    "document_type": "Unknown",
                                    "extraction_status": "failed"
                                }
                        except Exception as e:
                            logger.error(f"Failed to process document {row[0]} concepts: {str(e)}")
                            doc_info["extracted_summary"] = {
                                "concept_count": 0,
                                "main_topic": "Processing failed",
                                "document_type": "Unknown",
                                "extraction_status": "failed"
                            }
                    
                    docs_list.append(doc_info)
                
                return docs_list

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document by ID with full extraction data"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT raw_content, filename, format, metadata, 
                           extracted_concepts, extracted_aspects
                    FROM documents_enhanced
                    WHERE id = %s
                ''', (doc_id,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        "content": result[0],
                        "filename": result[1],
                        "format": result[2],
                        "metadata": result[3],
                        "extracted_concepts": result[4],
                        "extracted_aspects": result[5]
                    }
                return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    DELETE FROM documents_enhanced
                    WHERE id = %s
                    RETURNING id
                ''', (doc_id,))
                result = cursor.fetchone()
                conn.commit()
                return result is not None

    def concept_search(self, query: str, concept_type: str = None, k: int = 5) -> List[Dict]:
        """Search documents by extracted concepts"""
        if not self.embeddings_available:
            return []
        
        search_conditions = []
        params = []
        
        # Add text search in concepts
        search_conditions.append("extracted_concepts::text ILIKE %s")
        params.append(f"%{query}%")
        
        # Add concept type filter if specified
        if concept_type:
            search_conditions.append("extracted_concepts ? %s")
            params.append(concept_type)
        
        params.append(k)
        
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f'''
                    SELECT id, filename, format, extracted_concepts, extracted_aspects,
                           ts_rank(to_tsvector('english', raw_content), plainto_tsquery('english', %s)) as relevance
                    FROM documents_enhanced
                    WHERE {' AND '.join(search_conditions)}
                    ORDER BY relevance DESC
                    LIMIT %s
                ''', [query] + params)
                
                results = cursor.fetchall()
                return [{
                    "id": row[0],
                    "filename": row[1],
                    "format": row[2],
                    "extracted_concepts": row[3],
                    "extracted_aspects": row[4],
                    "relevance": float(row[5]) if row[5] else 0.0
                } for row in results]

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Find similar documents using concept-based search"""
        return self.concept_search(query, k=k)

    def aspect_search(self, aspect_name: str, query: str = None, k: int = 5) -> List[Dict]:
        """Search documents by specific aspect"""
        if not self.embeddings_available:
            return []
        
        conditions = ["extracted_aspects ? %s"]
        params = [aspect_name]
        
        if query:
            conditions.append("extracted_aspects::text ILIKE %s")
            params.append(f"%{query}%")
        
        params.append(k)
        
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f'''
                    SELECT id, filename, format, extracted_aspects->>%s as aspect_data,
                           metadata
                    FROM documents_enhanced
                    WHERE {' AND '.join(conditions)}
                    ORDER BY created_at DESC
                    LIMIT %s
                ''', [aspect_name] + params)
                
                results = cursor.fetchall()
                return [{
                    "id": row[0],
                    "filename": row[1],
                    "format": row[2],
                    "aspect_data": json.loads(row[3]) if row[3] else {},
                    "metadata": row[4]
                } for row in results]

    def extract_custom_concepts(self, doc_id: str, custom_concepts: List[Dict]) -> Dict:
        """Extract custom concepts from an existing document"""
        if not self.embeddings_available:
            raise ConnectionError("ContextGem not available")
        
        # Get document
        doc_data = self.get_document(doc_id)
        if not doc_data:
            raise ValueError(f"Document {doc_id} not found")
        
        # Create ContextGem document
        cg_doc = Document(raw_text=doc_data["content"])
        
        # Add custom concepts
        concepts = []
        for concept_def in custom_concepts:
            concept_type = concept_def.get("type", "string")
            if concept_type == "string":
                concept = StringConcept(
                    name=concept_def["name"],
                    description=concept_def["description"],
                    add_references=True,
                    reference_depth="sentences"
                )
            elif concept_type == "json":
                concept = JsonObjectConcept(
                    name=concept_def["name"],
                    description=concept_def["description"],
                    structure=concept_def.get("structure", {}),
                    add_references=True,
                    reference_depth="sentences"
                )
            elif concept_type == "date":
                concept = DateConcept(
                    name=concept_def["name"],
                    description=concept_def["description"],
                    add_references=True,
                    reference_depth="sentences"
                )
            elif concept_type == "number":
                concept = NumericalConcept(
                    name=concept_def["name"],
                    description=concept_def["description"],
                    numeric_type=concept_def.get("numeric_type", "float"),
                    add_references=True,
                    reference_depth="sentences"
                )
            else:
                concept = StringConcept(
                    name=concept_def["name"],
                    description=concept_def["description"],
                    add_references=True,
                    reference_depth="sentences"
                )
            concepts.append(concept)
        
        cg_doc.add_concepts(concepts)
        
        # Extract
        extracted_doc = self.llm.extract_all(cg_doc, use_concurrency=True)
        
        # Format results
        results = {}
        for concept in extracted_doc.concepts:
            results[concept.name] = {
                "items": [
                    {
                        "value": item.value,
                        "references": getattr(item, 'reference_sentences', []),
                        "justification": getattr(item, 'justification', None)
                    }
                    for item in concept.extracted_items
                ]
            }
        
        return results

    def get_retriever(self):
        """Return a retriever compatible with LangChain's interface"""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document as LangchainDocument
        
        class ContextGemRetriever(BaseRetriever):
            def __init__(self, vector_manager):
                self.vector_manager = vector_manager
                super().__init__()
            
            def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
                results = self.vector_manager.concept_search(query)
                documents = []
                
                for doc in results:
                    # Create metadata from extracted concepts
                    metadata = {
                        "id": doc["id"],
                        "filename": doc["filename"],
                        "format": doc["format"],
                        "relevance": doc.get("relevance", 0.0)
                    }
                    
                    # Add concept summaries to metadata
                    if doc.get("extracted_concepts"):
                        concepts = doc["extracted_concepts"]
                        for concept_name, concept_data in concepts.items():
                            if concept_data.get("items"):
                                metadata[f"concept_{concept_name.lower().replace(' ', '_')}"] = concept_data["items"][0]["value"]
                    
                    # Get document content
                    full_doc = self.vector_manager.get_document(doc["id"])
                    content = full_doc["content"] if full_doc else ""
                    
                    documents.append(LangchainDocument(
                        page_content=content,
                        metadata=metadata
                    ))
                
                return documents
        
        return ContextGemRetriever(self)

    def get_stats(self) -> Dict:
        """Get statistics about stored documents and extracted concepts"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                # Get basic counts
                cursor.execute("SELECT COUNT(*) FROM documents_enhanced")
                doc_count = cursor.fetchone()[0]
                
                # Get format distribution
                cursor.execute('''
                    SELECT format, COUNT(*) 
                    FROM documents_enhanced 
                    GROUP BY format
                ''')
                format_dist = dict(cursor.fetchall())
                
                # Get concept statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as docs_with_concepts,
                        AVG(jsonb_array_length(jsonb_path_query_array(extracted_concepts, '$.*[*].items'))) as avg_concepts_per_doc
                    FROM documents_enhanced 
                    WHERE extracted_concepts IS NOT NULL
                ''')
                concept_stats = cursor.fetchone()
                
                return {
                    "total_documents": doc_count,
                    "format_distribution": format_dist,
                    "docs_with_concepts": concept_stats[0] if concept_stats else 0,
                    "avg_concepts_per_doc": float(concept_stats[1]) if concept_stats and concept_stats[1] else 0.0,
                    "contextgem_available": self.embeddings_available
                }
