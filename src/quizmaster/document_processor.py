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

# Specifically suppress pdfminer.pdfpage warnings
pdfpage_logger = logging.getLogger('pdfminer.pdfpage')
pdfpage_logger.setLevel(logging.ERROR)

class DocumentProcessor:
    def __init__(self, persist_directory: str = "documents"):
        self.supported_formats = ['pdf', 'docx', 'txt', 'html']
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ContextGem LLM - will be set when model is selected
        self.contextgem_llm = None
        self.current_model = None
        self.contextgem_enabled = False  # Disable by default due to model compatibility issues
        self.fallback_extraction_enabled = True  # Enable robust fallback concept extraction
        
        # Token limits and chunking settings
        self.max_tokens = 6000  # Leave some buffer below 8192 limit
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-style encoding for approximate counting
    
    def set_model(self, model_name: str):
        """Set the model for ContextGem extraction."""
        if model_name != self.current_model:
            self.current_model = model_name
            self.contextgem_llm = DocumentLLM(
                model=f"ollama/{model_name}",
                api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
                api_base="http://localhost:11434"
            )
            print(f"DocumentProcessor: Set model to {model_name}")
        
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
        """Segment document into logical units (sentences) with metadata and filter irrelevant content."""
        # This method will be refactored to use ContextGem for more intelligent segmentation/extraction
        # For now, keep a basic sentence segmentation for compatibility
        segments = []
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        sentences = [s + (re.search(r'[.!?]', text[text.find(s) + len(s):]).group(0) if re.search(r'[.!?]', text[text.find(s) + len(s):]) else '') for s in sentences if s.strip()]
        if not sentences and text.strip():
            sentences = [text.strip()]
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            segments.append({'text': sentence, 'sentence_number': i + 1, 'word_count': len(sentence.split())})
        return segments

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
            print(f"Attempting ContextGem extraction for chunk {chunk_index}")
            cg_doc = ContextGemDocument(raw_text=chunk_text)
            
            # Define concepts for extraction
            cg_doc.add_concepts([
                StringConcept(name="Key Definition", description="A key term or concept defined in the text."),
                StringConcept(name="Important Fact", description="A significant piece of information or data."),
                StringConcept(name="Main Idea", description="The central theme or argument of a section."),
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
            
            print(f"ContextGem extracted {len(extracted_concepts)} concepts from chunk {chunk_index}")
            return extracted_concepts
            
        except Exception as e:
            print(f"ContextGem extraction failed for chunk {chunk_index}: {str(e)}")
            print("This is likely due to the model not producing valid JSON. Document processing will continue without ContextGem extraction.")
            return []

    def extract_concepts_fallback(self, text: str, filename: str, file_format: str) -> List[Dict]:
        """Fallback concept extraction using direct LLM calls without strict JSON requirements."""
        try:
            from litellm import completion
            
            # Chunk the text if it's too large
            chunks = self.chunk_text(text)
            all_concepts = []
            
            for chunk_index, chunk in enumerate(chunks):
                print(f"Extracting concepts from chunk {chunk_index + 1}/{len(chunks)}")
                
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
                    
                except Exception as e:
                    print(f"Error extracting concepts from chunk {chunk_index}: {str(e)}")
                    continue
            
            print(f"Fallback extraction found {len(all_concepts)} concepts")
            return all_concepts
            
        except Exception as e:
            print(f"Fallback concept extraction failed: {str(e)}")
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

    def process(self, file_content: bytes, filename: str) -> Dict:
        """Process the document and return structured content."""
        if not self.validate_file(filename):
            raise ValueError(f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}")

        file_format = os.path.splitext(filename)[1][1:].lower()
        
        try:
            # Convert to text
            text = self.convert_to_text(file_content, file_format)
            
            if not text.strip():
                raise ValueError("The document appears to be empty or could not be processed.")
            
            # Check if document needs to be chunked
            token_count = self.count_tokens(text)
            print(f"Document token count: {token_count}")
            
            extracted_concepts = []
            
            # Try ContextGem extraction first, then fallback to robust method
            if self.contextgem_enabled and self.contextgem_llm is not None:
                print("ContextGem extraction enabled - attempting concept extraction")
                if token_count <= self.max_tokens:
                    # Process entire document at once
                    print("Processing document as single chunk")
                    extracted_concepts = self.process_chunk_with_contextgem(text, filename, file_format, 0)
                else:
                    # Chunk the document and process each chunk
                    print(f"Document too large ({token_count} tokens), chunking...")
                    chunks = self.chunk_text(text)
                    print(f"Split into {len(chunks)} chunks")
                    
                    for i, chunk in enumerate(chunks):
                        print(f"Processing chunk {i+1}/{len(chunks)}")
                        chunk_concepts = self.process_chunk_with_contextgem(chunk, filename, file_format, i)
                        extracted_concepts.extend(chunk_concepts)
            
            # Use fallback extraction if ContextGem is disabled or if we have a model available
            if self.fallback_extraction_enabled and self.current_model and len(extracted_concepts) == 0:
                print("Using robust fallback concept extraction")
                extracted_concepts = self.extract_concepts_fallback(text, filename, file_format)
            
            # Enhanced metadata with ContextGem data
            enhanced_metadata = self.extract_metadata(self.intelligent_segmentation(text))
            enhanced_metadata.update({
                'extracted_concepts_count': len(extracted_concepts),
                'concept_types': list(set([concept['concept_name'] for concept in extracted_concepts])),
                'contextgem_processed': True,
                'original_tokens': token_count,
                'was_chunked': token_count > self.max_tokens,
                'chunks_processed': len(self.chunk_text(text)) if token_count > self.max_tokens else 1
            })

            return {
                'content': text,
                'segments': self.intelligent_segmentation(text),
                'metadata': enhanced_metadata,
                'filename': filename,
                'format': file_format,
                'extracted_concepts': extracted_concepts,  # ContextGem extracted data
            }
            
        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")

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
