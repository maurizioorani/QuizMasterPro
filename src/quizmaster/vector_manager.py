import os
import json
import hashlib
from typing import Dict, List, Optional
import numpy as np
from psycopg2 import sql
from psycopg2.extras import execute_values
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document as LangchainDocument
from database_manager import DatabaseManager

class VectorManager:
    def __init__(self):
        self.db = DatabaseManager()
        self._setup_vector_extension()
        self._create_documents_table()
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.embedding_dim = 768  # Default for nomic-embed-text

    def _setup_vector_extension(self):
        """Ensure pgvector extension is enabled in PostgreSQL"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()

    def _create_documents_table(self):
        """Create table for storing documents with vector embeddings"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql.SQL('''
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        filename TEXT,
                        format TEXT,
                        metadata JSONB,
                        embedding vector({})
                    )
                ''').format(sql.SQL(str(self.embedding_dim))))
                
                # Create index for vector search
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON documents USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                ''')
            conn.commit()

    def store_document(self, processed_doc: Dict) -> str:
        """Store document with embeddings in PostgreSQL"""
        content = processed_doc["content"]
        doc_id = hashlib.sha256(content.encode()).hexdigest()
        
        # Generate embedding
        embedding = self.embeddings.embed_documents([content])[0]
        
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO documents (id, content, filename, format, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        filename = EXCLUDED.filename,
                        format = EXCLUDED.format,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                ''', (
                    doc_id,
                    content,
                    processed_doc["filename"],
                    processed_doc["format"],
                    json.dumps(processed_doc.get("metadata", {})),
                    np.array(embedding).tobytes()
                ))
            conn.commit()
        return doc_id

    def list_documents(self) -> List[Dict]:
        """List all stored documents with metadata"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT id, filename, format, metadata
                    FROM documents
                ''')
                results = cursor.fetchall()
                
                docs_list = []
                for row in results:
                    docs_list.append({
                        "id": row[0],
                        "filename": row[1],
                        "format": row[2],
                        "metadata": row[3]
                    })
                return docs_list

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document by ID"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT content, filename, format, metadata
                    FROM documents
                    WHERE id = %s
                ''', (doc_id,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        "content": result[0],
                        "filename": result[1],
                        "format": result[2],
                        "metadata": result[3]
                    }
                return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    DELETE FROM documents
                    WHERE id = %s
                    RETURNING id
                ''', (doc_id,))
                result = cursor.fetchone()
                conn.commit()
                return result is not None

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Find similar documents using vector search"""
        query_embedding = self.embeddings.embed_documents([query])[0]
        
        with self.db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT id, content, filename, format, metadata,
                        1 - (embedding <=> %s) as similarity
                    FROM documents
                    ORDER BY embedding <=> %s
                    LIMIT %s
                ''', (np.array(query_embedding).tobytes(), 
                     np.array(query_embedding).tobytes(), 
                     k))
                
                results = cursor.fetchall()
                return [{
                    "id": row[0],
                    "content": row[1],
                    "filename": row[2],
                    "format": row[3],
                    "metadata": row[4],
                    "similarity": float(row[5])
                } for row in results]

    def get_retriever(self):
        """Return a retriever compatible with LangChain's interface"""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        
        class PgVectorRetriever(BaseRetriever):
            def __init__(self, vector_manager):
                self.vector_manager = vector_manager
                super().__init__()
            
            def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
                results = self.vector_manager.similarity_search(query)
                return [
                    Document(
                        page_content=doc["content"],
                        metadata={
                            "id": doc["id"],
                            "filename": doc["filename"],
                            "format": doc["format"],
                            **doc["metadata"]
                        }
                    ) for doc in results
                ]
        
        return PgVectorRetriever(self)
