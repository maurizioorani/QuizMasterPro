import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
import hashlib
import os
import shutil
import time
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document as LangchainDocument

class ChromaManager:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        
        # Clear all potential environment variables that might force http-only mode
        env_vars_to_clear = [
            "CHROMA_SERVER_NO_RECOIL",
            "CHROMA_SERVER_HOST",
            "CHROMA_SERVER_HTTP_PORT",
            "CHROMA_API_IMPL",
            "CHROMA_DB_IMPL"
        ]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]

        # Initialize OllamaEmbeddings for LangChain with dedicated embedding model
        embedding_model = "nomic-embed-text:latest"
        
        # Ensure the embedding model is available
        self._ensure_embedding_model(embedding_model)
        
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        # Try different ChromaDB initialization approaches
        self.client = None
        self.vectorstore = None
        
        # Approach 1: Try basic PersistentClient without custom settings
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.vectorstore = Chroma(
                client=self.client,
                collection_name="documents",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("Successfully initialized ChromaDB with PersistentClient.")
        except Exception as e1:
            print(f"PersistentClient failed: {e1}")
            
            # Approach 2: Try basic in-memory client
            try:
                self.client = chromadb.Client()
                self.vectorstore = Chroma(
                    client=self.client,
                    collection_name="documents",
                    embedding_function=self.embeddings
                )
                print("Successfully initialized ChromaDB with in-memory Client.")
            except Exception as e2:
                print(f"In-memory Client failed: {e2}")
                
                # Approach 3: Try HTTPClient (requires ChromaDB server to be running)
                try:
                    self.client = chromadb.HttpClient(host="localhost", port=8000)
                    self.vectorstore = Chroma(
                        client=self.client,
                        collection_name="documents",
                        embedding_function=self.embeddings
                    )
                    print("Successfully initialized ChromaDB with HttpClient connecting to localhost:8000.")
                except Exception as e3:
                    print(f"HttpClient connection failed: {e3}")
                    
                    # Final approach: Completely bypass ChromaDB for now and use a mock implementation
                    print("All ChromaDB initialization methods failed. Using mock implementation.")
                    self._use_mock_implementation()

    def _ensure_embedding_model(self, model_name: str):
        """Ensure the embedding model is available, pull if necessary"""
        import subprocess
        import requests
        
        try:
            # Check if model is available
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                if model_name in available_models:
                    print(f"✅ Embedding model {model_name} is already available")
                    return
            
            # Model not available, try to pull it
            print(f"🔄 Pulling embedding model {model_name}...")
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully pulled embedding model {model_name}")
            else:
                print(f"❌ Failed to pull embedding model {model_name}: {result.stderr}")
                
        except Exception as e:
            print(f"⚠️ Error checking/pulling embedding model {model_name}: {e}")

    def _use_mock_implementation(self):
        """Use a mock implementation that doesn't actually store anything"""
        self.client = None
        self.vectorstore = None
        self._mock_storage = {}  # Simple dict to store documents temporarily
        print("WARNING: Using mock ChromaDB implementation. No vector search capabilities available.")
        print("To fix this issue:")
        print("1. Install ChromaDB properly: pip install chromadb")
        print("2. Or start a ChromaDB server: chroma run --host localhost --port 8000 --path ./chroma_db")

    def store_document(self, processed_doc: Dict) -> str:
        """Store processed document using LangChain's Chroma and return its ID"""
        content = processed_doc["content"]
        doc_id = hashlib.sha256(content.encode()).hexdigest()
        
        if self.vectorstore is None:
            # Mock implementation
            self._mock_storage[doc_id] = {
                "content": content,
                "filename": processed_doc["filename"],
                "format": processed_doc["format"],
                "metadata": processed_doc.get("metadata", {})
            }
            return doc_id
        
        # LangChain Document expects metadata to be flat
        metadata = {
            "filename": processed_doc["filename"],
            "format": processed_doc["format"],
            "source": doc_id # Use doc_id as source for retrieval
        }
        # Add other metadata from processed_doc if available and flat
        for key, value in processed_doc.get("metadata", {}).items():
            if isinstance(value, (str, int, float, bool)): # Only store scalar types directly
                metadata[key] = value
            elif isinstance(value, (dict, list)): # Convert complex types to JSON string
                import json
                metadata[key] = json.dumps(value)

        langchain_doc = LangchainDocument(page_content=content, metadata=metadata)
        
        # Add document to Chroma. LangChain handles embedding and storage.
        self.vectorstore.add_documents([langchain_doc])
        return doc_id

    def list_documents(self) -> List[Dict]:
        """List all stored documents with metadata"""
        if self.vectorstore is None:
            # Mock implementation
            docs_list = []
            for doc_id, doc_data in self._mock_storage.items():
                docs_list.append({
                    "id": doc_id,
                    "filename": doc_data["filename"],
                    "format": doc_data["format"]
                })
            return docs_list
        
        # Access the underlying chromadb client collection
        collection = self.client.get_collection(name="documents")
        results = collection.get(ids=collection.get()['ids'], include=["metadatas"])
        
        docs_list = []
        for i in range(len(results["ids"])):
            doc_id = results["ids"][i]
            metadata = results["metadatas"][i]
            
            # Reconstruct original metadata if it was JSON stringified
            import json
            for key, value in metadata.items():
                try:
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass # Not a JSON string
            
            docs_list.append({"id": doc_id, **metadata})
        return docs_list

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document by ID"""
        if self.vectorstore is None:
            # Mock implementation
            if doc_id in self._mock_storage:
                doc_data = self._mock_storage[doc_id]
                return {
                    "content": doc_data["content"],
                    "filename": doc_data["filename"],
                    "format": doc_data["format"],
                    "metadata": doc_data["metadata"]
                }
            return None
        
        try:
            # LangChain Chroma's get_by_id is not directly exposed,
            # so we use the underlying client.
            collection = self.client.get_collection(name="documents")
            result = collection.get(ids=[doc_id], include=["documents", "metadatas"])
            
            if result["documents"]:
                content = result["documents"][0]
                metadata = result["metadatas"][0]
                
                # Reconstruct original metadata if it was JSON stringified
                import json
                for key, value in metadata.items():
                    try:
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            metadata[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass # Not a JSON string

                return {
                    "content": content,
                    "filename": metadata.get("filename", "unknown"),
                    "format": metadata.get("format", "txt"),
                    "metadata": {k: v for k, v in metadata.items() if k not in ["filename", "format", "source"]}
                }
        except Exception as e:
            print(f"Error getting document {doc_id}: {e}")
            return None
        return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        if self.vectorstore is None:
            # Mock implementation
            if doc_id in self._mock_storage:
                del self._mock_storage[doc_id]
                return True
            return False
        
        try:
            # LangChain Chroma's delete method is not directly exposed for specific IDs,
            # so we use the underlying client.
            collection = self.client.get_collection(name="documents")
            collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False

    def get_retriever(self):
        """Return a LangChain retriever for the stored documents."""
        if self.vectorstore is None:
            # Mock implementation - return None or a simple mock retriever
            print("WARNING: No retriever available with mock implementation. Quiz generation may be limited.")
            return None
        return self.vectorstore.as_retriever()
