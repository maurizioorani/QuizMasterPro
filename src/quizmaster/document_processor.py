import os
import io
from typing import Dict, List, Optional
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
import requests
import re
import logging
import json
import tiktoken
from contextgem import Document as ContextGemDocument, DocumentLLM, StringConcept, NumericalConcept, BooleanConcept, Aspect

# Configure logging to suppress pdfminer warnings
pdfminer_logger = logging.getLogger('pdfminer')
pdfminer_logger.setLevel(logging.ERROR)

# Specifically suppress pdfminer.pdfpage warnings
pdfpage_logger = logging.getLogger('pdfminer.pdfpage')
pdfpage_logger.setLevel(logging.ERROR)

# Configure general logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, persist_directory: str = "documents"):
        self.supported_formats = ['pdf', 'docx', 'txt', 'html']
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ContextGem LLM - will be set when model is selected
        self.contextgem_llm = None
        self.current_model = None
        self.contextgem_enabled = True  # Enable ContextGem by default
        self.fallback_extraction_enabled = True  # Enable fallback extraction
        self.openai_models = [
            "gpt-4.1-nano",
            "gpt-4o-mini"
        ]
        self.ollama_models = [
            "mistral:7b",
            "qwen2.5:7b",
            "gemma2:9b",
            "deepseek-coder:6.7b",
            "codellama:7b",
            "llama3.3:8b",
            "极速3.3:70b",
            "gemma3:9b",
            "phi4:latest",
            "deepseek-r1:latest",
            "granite3.3:latest",
            "llama3.1:8b",
            "mistral-nemo:latest"
        ]
        self.available_models = self.openai_models + self.ollama_models
        
        # Token limits and chunking settings
        self.max_tokens = 6000  # Leave some buffer below 8192 limit
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-style encoding for approximate counting
    
    def set_model(self, model_name: str):
        """Set the model for ContextGem extraction."""
        if model_name != self.current_model:
            self.current_model = model_name
            logger.info(f"DocumentProcessor: Attempting to set model to {model_name}")
            if model_name in self.openai_models:
                # Skip ContextGem initialization for OpenAI models
                self.contextgem_llm = None # Ensure ContextGem is None for OpenAI
                logger.info(f"DocumentProcessor: Set OpenAI model to {model_name}. ContextGem disabled.")
            else:
                # Initialize ContextGem for Ollama models
                try:
                    self.contextgem_llm = DocumentLLM(
                        model=f"ollama/{model_name}",
                        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"), # Use OLLAMA_API_KEY if available, default to "ollama"
                        api_base="http://localhost:11434"
                    )
                    logger.info(f"DocumentProcessor: Set Ollama model to {model_name}. ContextGem initialized.")
                except Exception as e:
                    self.contextgem_llm = None
                    logger.error(f"DocumentProcessor: Failed to initialize ContextGem for {model_name}: {str(e)}", exc_info=True)
                    logger.warning("DocumentProcessor: ContextGem is not available. Falling back to basic extraction.")
                
    def get_current_model(self) -> str:
        """Get the currently set model."""
        return self.current_model if self.current_model else "gpt-4.1-nano"
        
    def store_document(self, processed_doc: Dict) -> str:
        """Store processed document as JSON file and return its ID"""
        import hashlib
        doc_id = hashlib.sha256(processed_doc["content"].encode()).hexdigest()
        
        # Store as JSON file
        doc_path = os.path.join(self.persist_directory, f"{doc_id}.json")
        with open(doc_path, 'w', encoding='utf-8') as f:
            # Don't store the full contextgem_document object (not serializable)
            storable_doc = {k: v for k, v in processed_doc.items() if k != 'contextgem_document'}
            json.dump(storable_doc, f, ensure_ascii=False, indent=2)
        
        return doc_id
    
    def list_documents(self) -> List[Dict]:
        """List all stored documents with metadata"""
        docs_list = []
        if not os.path.exists(self.persist_directory):
            return docs_list
            
        for filename in os.listdir(self.persist_directory):
            if filename.endswith('.json'):
                doc_id = filename[:-5]  # Remove .json extension
                try:
                    doc_path = os.path.join(self.persist_directory, filename)
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                    
                    docs_list.append({
                        "id": doc_id,
                        "filename": doc_data.get("filename", "unknown"),
                        "format": doc_data.get("format", "txt"),
                        "metadata": doc_data.get("metadata", {})
                    })
                except Exception as e:
                    print(f"Error reading document {doc_id}: {e}")
                    
        return docs_list
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document by ID"""
        doc_path = os.path.join(self.persist_directory, f"{doc_id}.json")
        if os.path.exists(doc_path):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading document {doc_id}: {e}")
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        doc_path = os.path.join(self.persist_directory, f"{doc_id}.json")
        if os.path.exists(doc_path):
            try:
                os.remove(doc_path)
                return True
            except Exception as e:
                print(f"Error deleting document {doc_id}: {e}")
        return False

    def validate_file(self, filename: str) -> bool:
        """Validate the uploaded file format."""
        file_format = os.path.splitext(filename)[1][1:].lower()
        return file_format in self.supported_formats

    def convert_to_text(self, file_content: bytes, file_format: str) -> str:
        """Convert the uploaded file to plain text."""
        try:
            if file_format == 'pdf':
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            elif file_format == 'docx':
                doc = Document(io.BytesIO(file_content))
                text = '\n'.join([para.text for para in doc.paragraphs])
            elif file_format == 'html':
                soup = BeautifulSoup(file_content, 'html.parser')
                text = soup.get_text()
            else:  # txt
                text = file_content.decode('utf-8', errors='ignore')
            return text
        except Exception as e:
            raise ValueError(f"Error processing {file_format} file: {str(e)}")

    def intelligent_segmentation(self, text: str) -> List[Dict]:
        """Segment document into logical units with ContextGem or fallback to basic segmentation"""
        if self.contextgem_llm is None:
            logger.warning("ContextGem not available for intelligent segmentation, using basic segmentation.")
            return self._basic_segmentation(text)
            
        try:
            logger.info("Using ContextGem for intelligent segmentation.")
            cg_doc = ContextGemDocument(raw_text=text)
            
            # Define aspects for segmentation
            cg_doc.add_aspects([
                Aspect(name="Chapter", description="A major chapter or section"),
                Aspect(name="Paragraph", description="A coherent paragraph of text"),
                Aspect(name="KeyPoint", description="An important point or idea")
            ])
            
            # Extract structured segments
            extracted_doc = self.contextgem_llm.extract_all(cg_doc)
            
            segments = []
            if extracted_doc and hasattr(extracted_doc, 'aspects'):
                for aspect in extracted_doc.aspects:
                    if hasattr(aspect, 'extracted_items') and aspect.extracted_items:
                        for item in aspect.extracted_items:
                            if hasattr(item, 'value') and item.value:
                                segments.append({
                                    'text': item.value,
                                    'type': aspect.name,
                                    'sentence_number': len(segments) + 1,
                                    'word_count': len(item.value.split()),
                                    'metadata': {
                                        'source_context': item.reference_sentences[0] if hasattr(item, 'reference_sentences') and item.reference_sentences else ""
                                    }
                                })
            
            # Fallback to basic segmentation if ContextGem returns nothing
            if not segments:
                logger.warning("ContextGem segmentation returned no segments, falling back to basic segmentation.")
                return self._basic_segmentation(text)
            
            logger.info(f"ContextGem intelligent segmentation successful. Generated {len(segments)} segments.")
            return segments
            
        except Exception as e:
            logger.error(f"Error in ContextGem intelligent segmentation: {str(e)}", exc_info=True)
            logger.warning("Falling back to basic segmentation.")
            return self._basic_segmentation(text)

    def count_tokens(self, text: str) -> int:
        """Count approximate tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks that fit within token limits."""
        token_count = self.count_tokens(text)
        
        if token_count <= self.max_tokens:
            return [text]
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed token limit
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current paragraph
                # If single paragraph is too large, split by sentences
                if self.count_tokens(paragraph) > self.max_tokens:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        test_sentence_chunk = temp_chunk + (" " if temp_chunk else "") + sentence
                        
                        if self.count_tokens(test_sentence_chunk) <= self.max_tokens:
                            temp_chunk = test_sentence_chunk
                        else:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                    
                    current_chunk = temp_chunk
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def process_chunk_with_contextgem(self, chunk_text: str, filename: str, file_format: str, chunk_index: int) -> List[Dict]:
        """Process a single chunk with ContextGem and return extracted concepts."""
        try:
            logger.info(f"Attempting ContextGem extraction for chunk {chunk_index}")
            
            if self.current_model in self.openai_models:
                # Skip ContextGem for OpenAI models
                logger.info("Skipping ContextGem extraction for OpenAI model.")
                return []
                
            if not self.contextgem_llm:
                logger.error("ContextGem LLM not initialized for extraction.")
                return [] # Return empty list instead of raising error
                
            cg_doc = ContextGemDocument(raw_text=chunk_text)
            
            # Define concepts for extraction from ContextGem docs
            cg_doc.add_concepts([
                StringConcept(name="Key Definition", description="A key term or concept defined in the text"), 
                StringConcept(name="Important Fact", description="A significant piece of information or data"),
                StringConcept(name="Main Idea", description="The central theme or argument of a section"),
                NumericalConcept(name="Key Number", description="Important numerical data in the text"),
                BooleanConcept(name="Has Important Fact", description="Whether the text contains an important fact")
            ])
            
            # Define document aspects for better segmentation  
            cg_doc.add_aspects([
                Aspect(name="Section", description="A distinct section of the document"),
                Aspect(name="Chapter", description="A major chapter or part of the document")
            ])
            
            # Extract information using ContextGem with timeout and error handling
            extracted_cg_doc = self.contextgem_llm.extract_all(cg_doc)
            
            # Create structured data from ContextGem extraction
            extracted_concepts = []
            if extracted_cg_doc and hasattr(extracted_cg_doc, 'concepts'):
                for concept in extracted_cg_doc.concepts:
                    if hasattr(concept, 'extracted_items') and concept.extracted_items:
                        for item in concept.extracted_items:
                            if hasattr(item, 'value') and item.value:
                                extracted_concepts.append({
                                    "content": item.value,
                                    "concept_name": concept.name,
                                    "type": "concept",
                                    "source_sentence": item.reference_sentences[0] if hasattr(item, 'reference_sentences') and item.reference_sentences else "",
                                    "chunk_index": chunk_index,
                                    "metadata": {
                                        "filename": filename,
                                        "format": file_format,
                                        "concept_name": concept.name,
                                        "chunk_index": chunk_index
                                    }
                                })
            
            logger.info(f"ContextGem extracted {len(extracted_concepts)} concepts from chunk {chunk_index}")
            return extracted_concepts
            
        except Exception as e:
            logger.error(f"ContextGem extraction failed for chunk {chunk_index}: {str(e)}", exc_info=True)
            logger.warning("This is likely due to the model not producing valid JSON. Document processing will continue without ContextGem extraction for this chunk.")
            return []

    def extract_concepts_fallback(self, text: str, filename: str, file_format: str) -> List[Dict]:
        """Fallback concept extraction using direct LLM calls without strict JSON requirements."""
        try:
            from litellm import completion
            
            # Chunk the text if it's too large
            chunks = self.chunk_text(text)
            all_concepts = []
            
            for chunk_index, chunk in enumerate(chunks):
                logger.info(f"Attempting fallback concept extraction from chunk {chunk_index + 1}/{len(chunks)}")
                
                prompt = f"""Analyze the following text and extract key concepts. For each concept, provide:
- The concept content/definition
- What type it is (Key Definition, Important Fact, or Main Idea)
- A brief source context

Text to analyze:
{chunk}

Please extract concepts in this format:
CONCEPT: [the actual concept content]
TYPE: [Key Definition/Important Fact/Main Idea]
SOURCE: [brief context from the text]
---

CONCEPT: [next concept]
TYPE: [type]
SOURCE: [source context]
---

Extract 5-10 key concepts from this text."""

                try:
                    response = completion(
                        model=f"ollama/{self.current_model}",
                        messages=[{"role": "user", "content": prompt}],
                        api_base="http://localhost:11434",
                        max_tokens=1500,
                        temperature=0.3
                    )
                    
                    response_text = response.choices[0].message.content
                    chunk_concepts = self._parse_concept_response(response_text, filename, file_format, chunk_index)
                    all_concepts.extend(chunk_concepts)
                    logger.info(f"Fallback extraction found {len(chunk_concepts)} concepts in chunk {chunk_index + 1}.")
                    
                except Exception as e:
                    logger.error(f"Error extracting concepts from chunk {chunk_index + 1}: {str(e)}", exc_info=True)
                    continue
            
            logger.info(f"Fallback extraction finished. Total concepts found: {len(all_concepts)}")
            return all_concepts
            
        except Exception as e:
            logger.error(f"Fallback concept extraction failed: {str(e)}", exc_info=True)
            return []
    
    def _parse_concept_response(self, response_text: str, filename: str, file_format: str, chunk_index: int) -> List[Dict]:
        """Parse the LLM response to extract concepts without requiring JSON."""
        concepts = []
        
        # Split by concept separators
        concept_blocks = response_text.split('---')
        
        for block in concept_blocks:
            block = block.strip()
            if not block:
                continue
                
            # Extract concept, type, and source using regex
            concept_match = re.search(r'CONCEPT:\s*(.+?)(?=TYPE:|$)', block, re.DOTALL | re.IGNORECASE)
            type_match = re.search(r'TYPE:\s*(.+?)(?=SOURCE:|$)', block, re.DOTALL | re.IGNORECASE)
            source_match = re.search(r'SOURCE:\s*(.+?)$', block, re.DOTALL | re.IGNORECASE)
            
            if concept_match:
                concept_content = concept_match.group(1).strip()
                concept_type = type_match.group(1).strip() if type_match else "Important Fact"
                source_context = source_match.group(1).strip() if source_match else ""
                
                # Clean up and validate the concept type
                if concept_type not in ["Key Definition", "Important Fact", "Main Idea"]:
                    concept_type = "Important Fact"  # Default fallback
                
                # Only add if we have substantial content
                if len(concept_content) > 10:
                    concepts.append({
                        "content": concept_content,
                        "concept_name": concept_type,
                        "type": "concept",
                        "source_sentence": source_context,
                        "chunk_index": chunk_index,
                        "metadata": {
                            "filename": filename,
                            "format": file_format,
                            "concept_name": concept_type,
                            "chunk_index": chunk_index,
                            "extraction_method": "fallback"
                        }
                    })
        
        return concepts

    def extract_metadata(self, segments: List[Dict]) -> Dict:
        """Extract comprehensive metadata from document segments."""
        metadata = {
            'total_paragraphs': len(segments),
            'total_words': sum(seg['word_count'] for seg in segments),
            'chapters': [],
            'sections': [],
            'content_types': {}
        }
        return metadata

    def process_chunk_with_openai(self, chunk_text: str, filename: str, file_format: str, chunk_index: int) -> List[Dict]:
        """Process a chunk using OpenAI API with JSON response formatting."""
        from litellm import completion
        import json
        
        try:
            prompt = f"""Analyze the following text and extract key information in JSON format:
{chunk_text}

Return a JSON object with these fields:
- "concepts": Array of objects with:
  - "content": The extracted concept text
  - "concept_name": One of ["Key Definition", "Important Fact", "Main Idea"]
  - "source_sentence": Context sentence where concept appears
- "aspects": Array of objects with:
  - "name": One of ["Section", "Chapter"]
  - "value": The section/chapter content

Example response format:
{{
  "concepts": [
    {{
      "content": "Machine learning is...",
      "concept_name": "Key Definition",
      "source_sentence": "The text defines machine learning as..."
    }}
  ],
  "aspects": [
    {{
      "name": "Section",
      "value": "Introduction to AI"
    }}
  ]
}}"""

            response = completion(
                model=self.current_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse and validate JSON
            try:
                result = json.loads(response.choices[0].message.content)
                concepts = []
                
                # Process concepts
                for concept in result.get("concepts", []):
                    concepts.append({
                        "content": concept.get("content", ""),
                        "concept_name": concept.get("concept_name", "Important Fact"),
                        "type": "concept",
                        "source_sentence": concept.get("source_sentence", ""),
                        "chunk_index": chunk_index,
                        "metadata": {
                            "filename": filename,
                            "format": file_format,
                            "extraction_method": "openai"
                        }
                    })
                
                return concepts
                
            except json.JSONDecodeError as e:
                print(f"Invalid JSON from OpenAI: {str(e)}")
                return []
                
        except Exception as e:
            print(f"OpenAI processing failed: {str(e)}")
            return []

    def process(self, content: bytes, source: str) -> Dict:
        """Process the document or text using appropriate method based on model type."""
        logger.info(f"Starting document processing for source: {source}")
        try:
            # Determine if content is a file upload or direct text input
            if os.path.exists(source):
                # It's a file path
                filename = os.path.basename(source)
                if not self.validate_file(filename):
                    raise ValueError(f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}")
                
                file_format = os.path.splitext(filename)[1][1:].lower()
                with open(source, 'rb') as f:
                    file_content = f.read()
                text = self.convert_to_text(file_content, file_format)
            else:
                # It's direct text input
                filename = "direct_input.txt"
                file_format = "txt"
                text = content.decode('utf-8') if isinstance(content, bytes) else content
            
            if not text.strip():
                raise ValueError("The document appears to be empty or could not be processed.")
            
            token_count = self.count_tokens(text)
            logger.info(f"Document token count: {token_count}")
            
            extracted_concepts = []
            
            if self.current_model in self.openai_models:
                logger.info("Using OpenAI for document processing and concept extraction.")
                if token_count <= self.max_tokens:
                    extracted_concepts = self.process_chunk_with_openai(text, filename, file_format, 0)
                else:
                    chunks = self.chunk_text(text)
                    logger.info(f"Document chunked into {len(chunks)} chunks for OpenAI processing.")
                    for i, chunk in enumerate(chunks):
                        concepts = self.process_chunk_with_openai(chunk, filename, file_format, i)
                        extracted_concepts.extend(concepts)
            else:
                # Use ContextGem if initialized, otherwise fallback
                if self.contextgem_llm:
                    logger.info("Using ContextGem for document processing and concept extraction.")
                    if token_count <= self.max_tokens:
                        extracted_concepts = self.process_chunk_with_contextgem(text, filename, file_format, 0)
                    else:
                        chunks = self.chunk_text(text)
                        logger.info(f"Document chunked into {len(chunks)} chunks for ContextGem processing.")
                        for i, chunk in enumerate(chunks):
                            concepts = self.process_chunk_with_contextgem(chunk, filename, file_format, i)
                            extracted_concepts.extend(concepts)
                
                # Fallback extraction if ContextGem extraction failed or was not used
                if not extracted_concepts and self.fallback_extraction_enabled: # Check if fallback is enabled
                     logger.warning("ContextGem extraction did not yield concepts or was not used. Attempting fallback extraction.")
                     extracted_concepts = self.extract_concepts_fallback(text, filename, file_format)
                     if not extracted_concepts:
                         logger.error("Fallback extraction also failed to extract concepts.")
                elif not extracted_concepts and not self.fallback_extraction_enabled:
                     logger.warning("ContextGem extraction did not yield concepts or was not used. Fallback extraction is disabled.")


            if not extracted_concepts:
                # Only raise error if no concepts were extracted by any method
                raise ValueError("Failed to extract any concepts from the document using available methods.")
            
            # Basic segmentation for all models (used for metadata and general structure)
            segments = self._basic_segmentation(text)
            logger.info(f"Basic segmentation resulted in {len(segments)} segments.")
            
            # Extract metadata including word counts
            metadata = self.extract_metadata(segments)
            metadata.update({
                'extracted_concepts_count': len(extracted_concepts),
                'concept_types': list(set([c['concept_name'] for c in extracted_concepts])),
                'original_tokens': token_count,
                'was_chunked': token_count > self.max_tokens,
                'processing_method': 'openai' if self.current_model in self.openai_models else ('contextgem' if self.contextgem_llm else 'fallback')
            })
            
            logger.info(f"Document processing complete. Extracted {len(extracted_concepts)} concepts.")

            return {
                'content': text,
                'segments': segments,
                'metadata': metadata,
                'source': source,
                'extracted_concepts': extracted_concepts,
            }
        except Exception as e:
            # Log the error and re-raise with additional context
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    def _basic_segmentation(self, text: str) -> List[Dict]:
        """Basic text segmentation fallback."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [{
            'text': s.strip(),
            'sentence_number': i+1,
            'word_count': len(s.split())
        } for i, s in enumerate(sentences) if s.strip()]

    def get_segments_by_section(self, processed_content: Dict, section_name: str) -> List[Dict]:
        """Filter segments by section name."""
        if not section_name:
            return processed_content['segments']
        
        relevant_segments = []
        for segment in processed_content['segments']:
            if (section_name.lower() in segment['text'].lower() or 
                (segment['section'] and section_name.lower() in segment['section'].lower()) or
                (segment['chapter'] and section_name.lower() in segment['chapter'].lower())):
                relevant_segments.append(segment)
        
        return relevant_segments

    def get_content_summary(self, processed_content: Dict) -> str:
        """Generate a summary of the document content."""
        metadata = processed_content['metadata']
        
        summary = f"Document: {processed_content['filename']}\n"
        summary += f"Format: {processed_content['format'].upper()}\n"
        summary += f"Total Words: {metadata['total_words']}\n"
        summary += f"Total Paragraphs: {metadata['total_paragraphs']}\n"
        
        if metadata['chapters']:
            summary += f"Chapters: {len(metadata['chapters'])}\n"
        
        if metadata['sections']:
            summary += f"Sections: {len(metadata['sections'])}\n"
        
        return summary
