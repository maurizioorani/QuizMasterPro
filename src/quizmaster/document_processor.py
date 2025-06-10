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
from contextgem import Document as ContextGemDocument, DocumentLLM, StringConcept, Aspect

# Configure logging to suppress pdfminer warnings
pdfminer_logger = logging.getLogger('pdfminer')
pdfminer_logger.setLevel(logging.ERROR)
pdfpage_logger = logging.getLogger('pdfminer.pdfpage')
pdfpage_logger.setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, persist_directory: str = "documents"):
        self.supported_formats = ['pdf', 'docx', 'txt', 'html']
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ContextGem LLM
        self.contextgem_llm = None
        self.current_model = "mistral:7b"  # Default to local model
        
        # Define model lists
        self.openai_models = [
            "gpt-4.1-nano",
            "gpt-4o-mini"
        ]
        self.ollama_models = [
            "llama3.3:8b",
            "mistral:7b",
            "deepseek-coder:6.7b",
            "deepseek-r1:latest"
        ]
        self.available_models = self.openai_models + self.ollama_models
        
        # Simplified chunking settings for ContextGem
        self.max_chunk_size = 8000  # ContextGem can handle larger chunks efficiently
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def get_current_model(self):
        """Return the currently selected model"""
        return self.current_model

    def set_model(self, model_name: str):
        """Set the current model for document processing"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not in available models")
            
        self.current_model = model_name
        logger.info(f"Model set to: {model_name} (using optimized direct extraction)")

    def count_tokens(self, text: str) -> int:
        """Count approximate tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: rough estimation
            return len(text) // 4

    def smart_chunk_text(self, text: str) -> List[str]:
        """Efficiently chunk text for ContextGem processing."""
        token_count = self.count_tokens(text)
        
        if token_count <= self.max_chunk_size:
            return [text]
        
        # Simple paragraph-based chunking - let ContextGem handle the complexity
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if self.count_tokens(test_chunk) <= self.max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]

    def extract_concepts_with_direct_llm(self, text: str, filename: str, file_format: str) -> List[Dict]:
        """Extract concepts using direct LLM calls - fast and reliable."""
        try:
            logger.info(f"Starting optimized concept extraction for {filename}")
            
            # Validate input
            if not isinstance(text, str) or not text.strip():
                logger.warning("Empty or invalid text provided")
                return []
            
            # Ensure we have a model set - respect user's selection
            if not self.current_model:
                # Only set default if no model has been explicitly selected
                try:
                    # Test if Ollama is available
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        self.current_model = "mistral:7b"  # Default local model
                        logger.info("No model set, defaulting to local model: mistral:7b")
                    else:
                        raise requests.RequestException("Ollama not responding")
                except Exception:
                    # Fallback to OpenAI if Ollama not available
                    if os.environ.get("OPENAI_API_KEY"):
                        self.current_model = "gpt-4o-mini"
                        logger.info("Ollama unavailable, defaulting to OpenAI: gpt-4o-mini")
                    else:
                        # Last resort
                        self.current_model = "mistral:7b"
                        logger.warning("No Ollama connection and no OpenAI key, defaulting to mistral:7b (may fail)")
            
            logger.info(f"Using model: {self.current_model}")
            
            # Use direct LLM call with optimized prompt
            return self._extract_concepts_fast(text, filename, file_format)
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {str(e)}")
            # Try fallback with different approach
            try:
                logger.info("Trying fallback concept extraction")
                return self._fallback_extract_basic(text, filename, file_format)
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed: {str(fallback_error)}")
                return []

    def _extract_concepts_fast(self, text: str, filename: str, file_format: str) -> List[Dict]:
        """Fast concept extraction using optimized direct LLM calls."""
        try:
            from litellm import completion
            
            # First, let's extract a short summary to better understand the document
            summary_prompt = f"""Provide a ONE PARAGRAPH summary of what this document is about:

{text[:2000]}

Summary (max 3 sentences):"""

            # Configure for the current model
            if self.current_model in self.ollama_models:
                model_name = f"ollama/{self.current_model}"
                api_base = "http://localhost:11434"
            else:
                model_name = self.current_model
                api_base = None
                
            # Get document summary first
            try:
                summary_response = completion(
                    model=model_name,
                    messages=[{"role": "user", "content": summary_prompt}],
                    api_base=api_base,
                    max_tokens=200,
                    temperature=0.1
                )
                document_summary = summary_response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Failed to get document summary: {str(e)}")
                document_summary = "Unknown document content"
            
            # Now create a more informed concept extraction prompt using the summary
            if self.current_model in self.ollama_models:
                # Improved prompt for Ollama models with clearer instructions
                prompt = f"""Based on this document summary: "{document_summary}"

Extract 6-8 key topics from this document. Focus ONLY on the knowledge domains, subjects, and concepts covered.

Document content (excerpt):
{text[:3000]}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
1. [First subject domain or concept] 
2. [Second subject domain or concept]
3. [Third subject domain or concept]
...

Example good topics:
- Machine Learning Algorithms
- Data Preprocessing Techniques
- Neural Network Architecture
- Web Security Fundamentals
- JavaScript Frameworks
- Database Normalization
- Literary Analysis Methods

DO NOT mention document format or features. ONLY extract actual knowledge topics."""
            else:
                # Completely redesigned prompt for OpenAI models with two-stage extraction
                prompt = f"""This document is about: "{document_summary}"

You are an expert knowledge extractor. Your task is to identify the MAIN KNOWLEDGE DOMAINS and SPECIFIC CONCEPTS covered in this document.

Document content (excerpt):
{text[:3500]}

Approach this in 2 steps:
1) First, identify the primary knowledge domains (subjects) in this document
2) For each domain, extract specific concepts that are discussed

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
DOMAIN: [knowledge area 1]
- [specific concept 1.1]
- [specific concept 1.2]

DOMAIN: [knowledge area 2]
- [specific concept 2.1]
- [specific concept 2.2]

Example domains: Computer Science, Quantum Physics, Financial Analysis
Example concepts: Binary Search Trees, Wave-Particle Duality, Risk Assessment Methods

STRICT REQUIREMENTS:
- Extract ONLY real knowledge topics that appear in the text
- NEVER mention document format (PDF, DOCX, etc.) 
- NEVER include document features (images, tables, headers)
- NEVER include metadata (author, file size, page count)
- Focus EXCLUSIVELY on the actual subject matter knowledge
- Include specialized terminology from the document
- Extract 3-5 domains with 2-3 concepts each

The extracted topics will be used for creating quiz questions, so they must accurately represent the document's ACTUAL SUBJECT MATTER."""

            # Configure for the current model
            if self.current_model in self.ollama_models:
                model_name = f"ollama/{self.current_model}"
                api_base = "http://localhost:11434"
                logger.info(f"Using Ollama model: {self.current_model}")
            else:
                model_name = self.current_model
                api_base = None
                logger.info(f"Using OpenAI model: {self.current_model}")
            
            # Log before sending request
            logger.info(f"Sending extraction request to {model_name}")
            
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_base=api_base,
                max_tokens=1000,  # Increased for more complete responses
                temperature=0.2  # Low temperature for consistent results
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"Got response: {response_text[:100]}...")
            
            # Use appropriate parser based on model type
            if self.current_model in self.ollama_models:
                return self._parse_ollama_response(response_text, filename, file_format)
            else:
                return self._parse_domain_concepts_response(response_text, filename, file_format)
            
        except Exception as e:
            logger.error(f"Fast extraction failed: {str(e)}")
            # Create some default concepts based on filename
            return self._generate_default_concepts(filename, file_format)

    def _parse_domain_concepts_response(self, response_text: str, filename: str, file_format: str) -> List[Dict]:
        """Parse the domain-based concept extraction response."""
        concepts = []
        lines = response_text.split('\n')
        
        current_domain = "General"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for domain headers
            if line.startswith("DOMAIN:"):
                current_domain = line.replace("DOMAIN:", "").strip()
                continue
                
            # Check for concept bullet points
            if line.startswith("-") or line.startswith("•"):
                concept_text = line.lstrip("-•").strip()
                if len(concept_text) > 3:  # Ensure concept has meaningful content
                    concepts.append({
                        "content": concept_text,
                        "concept_name": current_domain,
                        "type": "concept",
                        "source_sentence": concept_text,
                        "metadata": {
                            "filename": filename,
                            "format": file_format,
                            "extraction_method": "domain_based"
                        }
                    })
        
        # Fallback for numbered lists if domain format wasn't used
        if not concepts:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Try to match numbered items (e.g., "1. Machine Learning")
                if re.match(r'^\d+\.\s', line):
                    concept_text = re.sub(r'^\d+\.\s', '', line).strip()
                    if len(concept_text) > 3:
                        concepts.append({
                            "content": concept_text,
                            "concept_name": "Key Topic",
                            "type": "concept",
                            "source_sentence": concept_text,
                            "metadata": {
                                "filename": filename,
                                "format": file_format,
                                "extraction_method": "numbered_list"
                            }
                        })
        
        return concepts
        
    def _parse_ollama_response(self, response_text: str, filename: str, file_format: str) -> List[Dict]:
        """Parse the response from Ollama models."""
        concepts = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match numbered items (e.g., "1. Machine Learning")
            if re.match(r'^\d+\.\s', line):
                concept_text = re.sub(r'^\d+\.\s', '', line).strip()
                if len(concept_text) > 3:  # Ensure concept has meaningful content
                    concepts.append({
                        "content": concept_text,
                        "concept_name": "Subject Matter",
                        "type": "concept",
                        "source_sentence": concept_text,
                        "metadata": {
                            "filename": filename,
                            "format": file_format,
                            "extraction_method": "ollama_direct"
                        }
                    })
            # Match bullet points
            elif line.startswith("-") or line.startswith("•"):
                concept_text = line.lstrip("-•").strip()
                if len(concept_text) > 3:  # Ensure concept has meaningful content
                    concepts.append({
                        "content": concept_text,
                        "concept_name": "Subject Matter",
                        "type": "concept",
                        "source_sentence": concept_text,
                        "metadata": {
                            "filename": filename,
                            "format": file_format,
                            "extraction_method": "ollama_direct"
                        }
                    })
        
        return concepts
        
    def _generate_default_concepts(self, filename: str, file_format: str) -> List[Dict]:
        """Generate default concepts based on filename when extraction fails."""
        # Extract meaningful terms from filename
        name_without_ext = os.path.splitext(filename)[0]
        words = re.findall(r'\w+', name_without_ext)
        
        concepts = []
        
        # Use filename as a basic concept
        if len(words) > 1:
            concepts.append({
                "content": " ".join(words),
                "concept_name": "Document Topic",
                "type": "concept",
                "source_sentence": filename,
                "metadata": {
                    "filename": filename,
                    "format": file_format,
                    "extraction_method": "filename_fallback"
                }
            })
        
        # Add file format as a technical concept
        concepts.append({
            "content": f"Document in {file_format.upper()} format",
            "concept_name": "Technical Information",
            "type": "concept",
            "source_sentence": filename,
            "metadata": {
                "filename": filename,
                "format": file_format,
                "extraction_method": "filename_fallback"
            }
        })
        
        return concepts

    def _parse_fast_response(self, response_text: str, filename: str, file_format: str) -> List[Dict]:
        """Parse the fast extraction response."""
        concepts = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse numbered format: "1. TYPE: content"
            if '. ' in line and ':' in line:
                try:
                    # Extract number, type, and content
                    parts = line.split('. ', 1)
                    if len(parts) == 2:
                        type_content = parts[1]
                        if ':' in type_content:
                            type_part, content = type_content.split(':', 1)
                            type_part = type_part.strip().upper()
                            content = content.strip()
                            
                            # Map types
                            if type_part == 'DEFINITION':
                                concept_type = 'Key Definition'
                            elif type_part == 'IDEA':
                                concept_type = 'Main Idea'
                            elif type_part == 'FACT':
                                concept_type = 'Important Fact'
                            else:
                                concept_type = 'Important Fact'
                            
                            # Only add if content is substantial
                            if len(content) > 15:
                                concepts.append({
                                    "content": content,
                                    "concept_name": concept_type,
                                    "type": "concept",
                                    "source_sentence": content[:100] + "...",
                                    "metadata": {
                                        "filename": filename,
                                        "format": file_format,
                                        "extraction_method": "optimized_direct"
                                    }
                                })
                except Exception:
                    continue
        
        return concepts

    def _fallback_extract_basic(self, text: str, filename: str, file_format: str) -> List[Dict]:
        """Basic concept extraction fallback using direct LLM calls."""
        try:
            from litellm import completion
            
            logger.info(f"Using fallback extraction for {filename}")
            
            # Use a simpler, more direct prompt
            prompt = f"""Extract key concepts from the following text. For each concept, provide the concept content and categorize it.

Text:
{text}

Please identify and extract concepts in this exact format:
CONCEPT_TYPE: Key Definition
CONTENT: [actual definition or term]
SOURCE: [brief context]

CONCEPT_TYPE: Main Idea  
CONTENT: [main concept or theme]
SOURCE: [brief context]

CONCEPT_TYPE: Important Fact
CONTENT: [significant fact or information]
SOURCE: [brief context]

Extract 5-10 concepts total."""

            # Determine model configuration
            if self.current_model in self.ollama_models:
                model_name = f"ollama/{self.current_model}"
                api_base = "http://localhost:11434"
            else:
                model_name = self.current_model
                api_base = None

            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_base=api_base,
                max_tokens=1500,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            return self._parse_fallback_response(response_text, filename, file_format)
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {str(e)}")
            return []

    def _parse_fallback_response(self, response_text: str, filename: str, file_format: str) -> List[Dict]:
        """Parse the fallback LLM response to extract concepts."""
        concepts = []
        
        # Split response into blocks and parse each one
        lines = response_text.split('\n')
        current_concept = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('CONCEPT_TYPE:'):
                # Save previous concept if it exists
                if current_concept.get('content'):
                    concepts.append(self._create_concept_dict(current_concept, filename, file_format))
                
                # Start new concept
                concept_type = line.replace('CONCEPT_TYPE:', '').strip()
                current_concept = {'concept_name': concept_type}
                
            elif line.startswith('CONTENT:'):
                content = line.replace('CONTENT:', '').strip()
                current_concept['content'] = content
                
            elif line.startswith('SOURCE:'):
                source = line.replace('SOURCE:', '').strip()
                current_concept['source_sentence'] = source
        
        # Add the last concept
        if current_concept.get('content'):
            concepts.append(self._create_concept_dict(current_concept, filename, file_format))
        
        return concepts

    def _create_concept_dict(self, concept_data: Dict, filename: str, file_format: str) -> Dict:
        """Create a standardized concept dictionary."""
        return {
            "content": concept_data.get('content', ''),
            "concept_name": concept_data.get('concept_name', 'Important Fact'),
            "type": "concept",
            "source_sentence": concept_data.get('source_sentence', ''),
            "metadata": {
                "filename": filename,
                "format": file_format,
                "extraction_method": "fallback"
            }
        }

    def validate_file(self, filename: str) -> bool:
        """Validate if the file has a supported format."""
        if not filename:
            return False
        file_extension = os.path.splitext(filename)[1][1:].lower()
        return file_extension in self.supported_formats

    def process(self, content: bytes, source: str, custom_filename: str = None) -> Dict:
        """Main processing method - simplified and optimized."""
        logger.info(f"Processing document: {source}")
        
        try:
            # Determine file type and extract text
            if os.path.exists(source):
                filename = os.path.basename(source)
                if not self.validate_file(filename):
                    raise ValueError(f"Unsupported file format. Supported: {', '.join(self.supported_formats)}")
                
                file_format = os.path.splitext(filename)[1][1:].lower()
                with open(source, 'rb') as f:
                    file_content = f.read()
                text = self.convert_to_text(file_content, file_format)
            else:
                # Direct input processing
                # First, determine the file format
                file_format = "txt"  # default
                
                # Check if it's a PDF by looking at source filename or content
                if source.lower().endswith('.pdf') or (isinstance(content, bytes) and content.startswith(b'%PDF')):
                    file_format = "pdf"
                elif source.lower().endswith('.docx'):
                    file_format = "docx"
                elif source.lower().endswith('.html'):
                    file_format = "html"
                
                # Extract text based on detected format
                if isinstance(content, bytes):
                    # Always convert bytes to text using the appropriate method
                    text = self.convert_to_text(content, file_format)
                else:
                    # String content
                    text = str(content)
                
                # Generate filename if not provided
                if custom_filename and custom_filename.strip():
                    filename = custom_filename.strip()
                    # Ensure filename has correct extension
                    if not filename.lower().endswith(f'.{file_format}'):
                        filename += f'.{file_format}'
                else:
                    # Generate filename from first few words of extracted text
                    words = text.split()[:5]  # Take first 5 words
                    if words:
                        filename = "_".join(word.strip('.,!?;:"()[]{}') for word in words)
                        filename = re.sub(r'[^\w\s-]', '', filename)  # Remove special chars
                        filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
                        filename = filename[:50] + f".{file_format}"  # Limit length and add extension
                    else:
                        filename = f"document.{file_format}"
            
            if not text.strip():
                raise ValueError("Document appears to be empty or could not be processed.")
            
            # Log processing stats
            token_count = self.count_tokens(text)
            word_count = len(text.split())
            logger.info(f"Document stats - Tokens: {token_count}, Words: {word_count}")
            
            # Extract concepts using optimized direct method
            concepts = self.extract_concepts_with_direct_llm(text, filename, file_format)
            
            # Create metadata compatible with Streamlit app
            paragraphs = text.split('\n\n')
            metadata = {
                'total_words': word_count,
                'total_tokens': token_count,
                'total_paragraphs': len([p for p in paragraphs if p.strip()]),
                'concept_count': len(concepts),
                'filename': filename,
                'format': file_format,
                'chapters': [],  # Simple implementation - could be enhanced
                'sections': []   # Simple implementation - could be enhanced
            }
            
            return {
                "content": text,
                "filename": filename,
                "format": file_format,
                "concepts": concepts,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
            
    def convert_to_text(self, content: bytes, file_format: str) -> str:
        """Convert various file formats to plain text - optimized."""
        if not content:
            return ""
            
        if not isinstance(content, bytes):
            if isinstance(content, str):
                return content
            try:
                content = bytes(content)
            except Exception as e:
                logger.error(f"Failed to convert content to bytes: {str(e)}")
                return ""
        
        try:
            if file_format == 'pdf':
                return self._extract_text_from_pdf(content)
            elif file_format == 'docx':
                return self._extract_text_from_docx(content)
            elif file_format == 'html':
                return self._extract_text_from_html(content.decode('utf-8', errors='ignore'))
            elif file_format == 'txt':
                return content.decode('utf-8', errors='ignore')
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            logger.error(f"Error converting {file_format} to text: {str(e)}")
            raise
            
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content."""
        text_parts = []
        
        # Check if content might be binary PDF data
        if content.startswith(b'%PDF'):
            # Process as binary PDF
            with io.BytesIO(content) as pdf_file:
                try:
                    with pdfplumber.open(pdf_file) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                            
                            # Log progress for large documents
                            if (page_num + 1) % 10 == 0:
                                logger.debug(f"Processed {page_num + 1} pages")
                                
                except Exception as e:
                    logger.error(f"PDF extraction error with pdfplumber: {str(e)}")
                    # Try fallback method if pdfplumber fails
                    try:
                        # If pdfplumber fails, try to decode as UTF-8 text
                        # This would happen if the PDF contains mostly text that's already UTF-8 encoded
                        decoded_text = content.decode('utf-8', errors='ignore')
                        # Only return the decoded text if it looks like valid text (not binary garbage)
                        if '%PDF' in decoded_text and not re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]{10,}', decoded_text):
                            return decoded_text
                    except Exception as fallback_error:
                        logger.error(f"Fallback PDF text extraction also failed: {str(fallback_error)}")
                    
                    # If all else fails, raise the original error
                    raise ValueError(f"Failed to extract text from PDF: {str(e)}")
        else:
            # If it doesn't start with %PDF, try to decode as text directly
            try:
                decoded_text = content.decode('utf-8', errors='ignore')
                return decoded_text
            except Exception as e:
                logger.error(f"Failed to decode content as UTF-8: {str(e)}")
                # Continue with pdfplumber as a fallback
                with io.BytesIO(content) as pdf_file:
                    try:
                        with pdfplumber.open(pdf_file) as pdf:
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text_parts.append(page_text)
                    except Exception as pdf_error:
                        logger.error(f"PDF extraction fallback error: {str(pdf_error)}")
                        raise ValueError(f"Failed to extract text from content: {str(e)}")
        
        # If we extracted text using pdfplumber, join the parts
        if text_parts:
            return "\n\n".join(text_parts)
        else:
            # If no text was extracted but no errors were raised, return empty string
            logger.warning("No text extracted from PDF content")
            return ""
        
    def _extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content."""
        text_parts = []
        
        with io.BytesIO(content) as docx_file:
            try:
                doc = Document(docx_file)
                for para in doc.paragraphs:
                    if para.text.strip():
                        text_parts.append(para.text)
            except Exception as e:
                logger.error(f"DOCX extraction error: {str(e)}")
                raise ValueError(f"Failed to extract text from DOCX: {str(e)}")
        
        return "\n".join(text_parts)
        
    def _extract_text_from_html(self, content: str) -> str:
        """Extract text from HTML content."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "meta", "link"]):
                element.extract()
            
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
            
        except Exception as e:
            logger.error(f"HTML extraction error: {str(e)}")
            raise ValueError(f"Failed to extract text from HTML: {str(e)}")

    def get_content_summary(self, processed_content: Dict) -> str:
        """Generate a content summary for display purposes."""
        try:
            content = processed_content.get('content', '')
            metadata = processed_content.get('metadata', {})
            concepts = processed_content.get('concepts', [])
            
            # Basic statistics
            word_count = metadata.get('total_words', len(content.split()))
            concept_count = len(concepts)
            
            # Generate summary
            lines = [
                f"Document: {processed_content.get('filename', 'Unknown')}",
                f"Format: {processed_content.get('format', 'Unknown').upper()}",
                f"Total Words: {word_count:,}",
                f"Concepts Extracted: {concept_count}",
                ""
            ]
            
            # Add concept breakdown if available
            if concepts:
                concept_types = {}
                for concept in concepts:
                    concept_type = concept.get('concept_name', 'Unknown')
                    concept_types[concept_type] = concept_types.get(concept_type, 0) + 1
                
                lines.append("Concept Breakdown:")
                for concept_type, count in concept_types.items():
                    lines.append(f"  • {concept_type}: {count}")
                lines.append("")
            
            # Add content preview
            preview_length = min(200, len(content))
            lines.append(f"Content Preview ({preview_length} chars):")
            lines.append(f"{content[:preview_length]}{'...' if len(content) > preview_length else ''}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error generating content summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def intelligent_segmentation(self, content: str) -> List[Dict]:
        """Provide intelligent segmentation of content into logical sections."""
        try:
            if not content or not isinstance(content, str):
                return []
            
            # Simple paragraph-based segmentation
            paragraphs = content.split('\n\n')
            segments = []
            
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    segments.append({
                        'id': i,
                        'type': 'paragraph',
                        'content': paragraph.strip(),
                        'word_count': len(paragraph.split()),
                        'start_position': content.find(paragraph),
                        'metadata': {
                            'segment_number': i + 1,
                            'total_segments': len([p for p in paragraphs if p.strip()])
                        }
                    })
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in intelligent segmentation: {str(e)}")
            return []
