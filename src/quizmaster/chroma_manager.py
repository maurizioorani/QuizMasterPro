import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
import hashlib
import os

class ChromaManager:
    def __init__(self, persist_directory: str = "chroma_db"):
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Use the new PersistentClient API
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            if "Could not connect to tenant default_tenant" in str(e):
                print(f"ChromaDB error detected: {e}. Attempting to reset ChromaDB directory.")
                # Clean up potentially corrupted ChromaDB directory
                import shutil
                if os.path.exists(persist_directory):
                    shutil.rmtree(persist_directory)
                os.makedirs(persist_directory, exist_ok=True)
                
                # Re-initialize client and collection
                self.client = chromadb.PersistentClient(path=persist_directory)
                self.collection = self.client.get_or_create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                raise # Re-raise other exceptions

    def store_document(self, processed_doc: Dict) -> str:
        """Store processed document and return its ID"""
        content = processed_doc["content"]
        doc_id = hashlib.sha256(content.encode()).hexdigest()
        
        # ChromaDB requires scalar metadata values - convert complex types to JSON strings
        flat_metadata = {}
        for key, value in processed_doc["metadata"].items():
            if isinstance(value, (dict, list)):
                # Convert to JSON string for complex types
                import json
                flat_metadata[key] = json.dumps(value)
            else:
                flat_metadata[key] = value
        
        metadata = {
            "filename": processed_doc["filename"],
            "format": processed_doc["format"],
            **flat_metadata
        }
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id

    def list_documents(self) -> List[Dict]:
        """List all stored documents with metadata"""
        results = self.collection.get(include=["metadatas"])
        return [
            {"id": id_, **metadata}
            for id_, metadata in zip(results["ids"], results["metadatas"])
        ]

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document by ID"""
        try:
            result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
            if result["documents"]:
                return {
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
        except Exception:
            return None
        return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
