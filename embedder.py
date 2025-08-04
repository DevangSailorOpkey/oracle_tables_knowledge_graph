import requests
import logging
from typing import Dict, List, Any, Optional, Union
import json
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaEmbedder:
    """Class to generate embeddings using Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        """Initialize the embedder with Ollama API settings
        
        Args:
            base_url: Base URL for Ollama API
            model: Embedding model to use
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.embed_endpoint = f"{self.base_url}/api/embeddings"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Ollama API"""
        try:
            # Simple request to check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                logger.warning(f"Ollama API returned status code {response.status_code}")
            else:
                available_models = [model.get('name') for model in response.json().get('models', [])]
                if self.model not in available_models:
                    logger.warning(f"Model {self.model} not found in available models: {available_models}")
                else:
                    logger.info(f"Successfully connected to Ollama API, model {self.model} is available")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama API: {str(e)}")
            logger.error("Make sure Ollama is running and the URL is correct")
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using Ollama API
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding vector, or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(self.embed_endpoint, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.status_code} {response.text}")
                return None
            
            result = response.json()
            embedding = result.get('embedding')
            
            if not embedding:
                logger.error(f"No embedding returned from Ollama API: {result}")
                return None
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return None
    
    def create_table_embedding_text(self, 
                                  table_name: str, 
                                  module: str, 
                                  submodule: str, 
                                  description: str, 
                                  primary_key: Optional[Union[Dict[str, Any], str]] = None,
                                  columns: Optional[List[Dict[str, Any]]] = None) -> str:
        """Create text for table embedding
        
        Args:
            table_name: Name of the table
            module: Module name
            submodule: Submodule name
            description: Table description
            primary_key: Primary key info
            columns: List of important columns
            
        Returns:
            Text string to be embedded
        """
        # Start with basic table info
        text_parts = [
            f"Table: {table_name}",
            f"Module: {module}",
            f"Submodule: {submodule}",
            f"Description: {description}"
        ]
        
        # Add primary key if available
        if primary_key:
            if isinstance(primary_key, dict):
                pk_columns = primary_key.get('columns', '')
            else:
                pk_columns = primary_key
                
            text_parts.append(f"Primary Key: {pk_columns}")
        
        # Add important columns (up to 10)
        if columns and isinstance(columns, list):
            column_texts = []
            for i, col in enumerate(columns[:10]):  # Limit to 10 columns
                if isinstance(col, dict):
                    col_name = col.get('name', '')
                    col_type = col.get('datatype', '')
                    col_desc = col.get('comments', '')
                    
                    if col_name:
                        col_text = f"{col_name} ({col_type})"
                        if col_desc:
                            col_text += f": {col_desc}"
                        column_texts.append(col_text)
                        
            if column_texts:
                text_parts.append("Important Columns: " + "; ".join(column_texts))
        
        # Join all parts with newlines
        return "\n".join(text_parts)
    
    def create_column_embedding_text(self,
                                   column_name: str,
                                   datatype: str,
                                   table_name: str,
                                   description: str = "",
                                   is_primary_key: bool = False,
                                   is_foreign_key: bool = False,
                                   references_column: str = "") -> str:
        """Create text for column embedding
        
        Args:
            column_name: Name of the column
            datatype: Data type of the column
            table_name: Name of the parent table
            description: Column description or comments
            is_primary_key: Whether the column is part of a primary key
            is_foreign_key: Whether the column is a foreign key
            references_column: Referenced column if foreign key
            
        Returns:
            Text string to be embedded
        """
        # Start with basic column info
        text_parts = [
            f"Column: {column_name}",
            f"Data Type: {datatype}",
            f"Table: {table_name}"
        ]
        
        # Add description if available
        if description:
            text_parts.append(f"Description: {description}")
        
        # Add key information
        if is_primary_key:
            text_parts.append("This is a primary key column")
        
        if is_foreign_key and references_column:
            text_parts.append(f"This is a foreign key referencing: {references_column}")
        
        # Join all parts with newlines
        return "\n".join(text_parts)
    
    def embed_table(self, 
                   table_name: str, 
                   module: str, 
                   submodule: str, 
                   description: str, 
                   primary_key: Optional[Union[Dict[str, Any], str]] = None,
                   columns: Optional[List[Dict[str, Any]]] = None) -> Optional[List[float]]:
        """Create embedding for a table
        
        Args:
            table_name: Name of the table
            module: Module name
            submodule: Submodule name
            description: Table description
            primary_key: Primary key info
            columns: List of important columns
            
        Returns:
            Embedding vector or None if failed
        """
        # Create text for embedding
        text = self.create_table_embedding_text(
            table_name, module, submodule, description, primary_key, columns
        )
        
        # Get embedding
        return self.get_embedding(text)
    
    def embed_column(self,
                    column_name: str,
                    datatype: str,
                    table_name: str,
                    description: str = "",
                    is_primary_key: bool = False,
                    is_foreign_key: bool = False,
                    references_column: str = "") -> Optional[List[float]]:
        """Create embedding for a column
        
        Args:
            column_name: Name of the column
            datatype: Data type of the column
            table_name: Name of the parent table
            description: Column description or comments
            is_primary_key: Whether the column is part of a primary key
            is_foreign_key: Whether the column is a foreign key
            references_column: Referenced column if foreign key
            
        Returns:
            Embedding vector or None if failed
        """
        # Create text for embedding
        text = self.create_column_embedding_text(
            column_name, datatype, table_name, description,
            is_primary_key, is_foreign_key, references_column
        )
        
        # Get embedding
        return self.get_embedding(text)

# Example usage
if __name__ == "__main__":
    embedder = OllamaEmbedder()
    sample_text = "This is a test text for embedding"
    embedding = embedder.get_embedding(sample_text)
    if embedding:
        print(f"Got embedding with {len(embedding)} dimensions")
    else:
        print("Failed to get embedding")