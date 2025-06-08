import os
import io
from typing import Dict, List, Optional
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
import requests
import re
from chroma_manager import ChromaManager

class DocumentProcessor:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.supported_formats = ['pdf', 'docx', 'txt', 'html']
        self.chroma = ChromaManager(persist_directory)
        
    def store_document(self, processed_doc: Dict) -> str:
        """Store processed document in ChromaDB and return its ID"""
        return self.chroma.store_document(processed_doc)
    
    def list_documents(self) -> List[Dict]:
        """List all stored documents with metadata"""
        return self.chroma.list_documents()
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document by ID"""
        doc = self.chroma.get_document(doc_id)
        if doc:
            # Extract filename and format from metadata
            metadata = doc.get('metadata', {})
            return {
                'content': doc['content'],
                'filename': metadata.get('filename', 'unknown'),
                'format': metadata.get('format', 'txt'),
                'metadata': {k: v for k, v in metadata.items() if k not in ['filename', 'format']}
            }
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        return self.chroma.delete_document(doc_id)

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
        segments = []
        
        # Split by sentences for more granularity
        # A simpler regex to split by common sentence endings, followed by whitespace
        # This might be less precise with abbreviations but avoids the look-behind error.
        # We'll rely more on the filtering logic to handle less-than-perfect splits.
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        # Add back the terminator to the end of each sentence for context, if it was split off
        sentences = [s + (re.search(r'[.!?]', text[text.find(s) + len(s):]).group(0) if re.search(r'[.!?]', text[text.find(s) + len(s):]) else '') for s in sentences if s.strip()]
        if not sentences and text.strip(): # Handle case where text has no terminators
            sentences = [text.strip()]
        
        current_section = None
        current_chapter = None
        
        # Keywords to identify potentially irrelevant segments (summary, author info)
        irrelevant_keywords = [
            "author", "summary", "introduction to", "overview of", "about this book",
            "copyright", "all rights reserved", "published by", "version", "edition",
            "dedication", "acknowledgements", "preface", "foreword", "table of contents",
            "list of figures", "list of tables", "index", "bibliography", "glossary",
            "contact information", "website", "email", "phone", "address", "disclaimer",
            "license", "terms of use", "chapter", "section", "part", "appendix", "annex",
            "preamble", "abstract", "executive summary", "conclusion", "final thoughts",
            "future work", "references", "citations", "further reading", "about the author",
            "about the publisher", "imprint", "isbn", "doi", "url", "http", "www", ".com", ".org", ".net"
        ]
        
        # Compile regex for faster matching
        irrelevant_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in irrelevant_keywords) + r')\b', re.IGNORECASE)

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Skip very short sentences that are unlikely to contain meaningful content
            if len(sentence.split()) < 5:
                continue

            # Heuristic to filter out irrelevant segments
            # Check if the sentence contains too many irrelevant keywords or looks like metadata
            if irrelevant_pattern.search(sentence) and len(sentence.split()) < 30: # Short sentences with keywords are suspicious
                continue
            
            # Detect chapters and sections (still useful for context, but not for filtering questions)
            chapter_match = re.match(r'^(chapter|ch\.?)\s+(\d+|[ivxlc]+)', sentence.lower())
            section_match = re.match(r'^(section|sec\.?)\s+(\d+)', sentence.lower())
            
            if chapter_match:
                current_chapter = sentence
            elif section_match:
                current_section = sentence
            
            # Create segment with metadata
            segment = {
                'text': sentence,
                'sentence_number': i + 1, # Changed from paragraph_number
                'chapter': current_chapter,
                'section': current_section,
                'word_count': len(sentence.split()),
                'type': self._classify_segment_type(sentence)
            }
            segments.append(segment)
        
        return segments

    def _classify_segment_type(self, segment_text: str) -> str:
        """Classify segment type based on content."""
        segment_lower = segment_text.lower()
        
        if re.match(r'^(chapter|ch\.?)', segment_lower):
            return 'chapter_header'
        elif re.match(r'^(section|sec\.?)', segment_lower):
            return 'section_header'
        elif segment_text.endswith('?'):
            return 'question'
        elif len(segment_text.split()) < 10:
            return 'short_text'
        else:
            return 'content'

    def extract_metadata(self, segments: List[Dict]) -> Dict:
        """Extract comprehensive metadata from document segments."""
        metadata = {
            'total_paragraphs': len(segments),
            'total_words': sum(seg['word_count'] for seg in segments),
            'chapters': [],
            'sections': [],
            'content_types': {}
        }
        
        # Extract unique chapters and sections
        for segment in segments:
            if segment['chapter'] and segment['chapter'] not in metadata['chapters']:
                metadata['chapters'].append(segment['chapter'])
            if segment['section'] and segment['section'] not in metadata['sections']:
                metadata['sections'].append(segment['section'])
        
        # Count content types
        for segment in segments:
            content_type = segment['type']
            metadata['content_types'][content_type] = metadata['content_types'].get(content_type, 0) + 1
        
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
            
            # Intelligent segmentation
            segments = self.intelligent_segmentation(text)
            
            # Extract metadata
            metadata = self.extract_metadata(segments)
            
            return {
                'content': text,
                'segments': segments,
                'metadata': metadata,
                'filename': filename,
                'format': file_format
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
