import logging
from typing import Dict, List, Any, Optional
from graph_builder import TableGraphBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TableRAGEngine:
    """RAG engine for Oracle tables knowledge graph"""
    
    def __init__(self, graph_builder: TableGraphBuilder):
        """Initialize the RAG engine
        
        Args:
            graph_builder: TableGraphBuilder instance
        """
        self.graph_builder = graph_builder
    
    def query(self, query_text: str, top_k: int = 5, include_related: bool = True) -> Dict[str, Any]:
        """Execute a RAG query against the knowledge graph
        
        Args:
            query_text: Natural language query
            top_k: Number of most relevant tables to return
            include_related: Whether to include related tables
            
        Returns:
            Dictionary with query results
        """
        # Step 1: Find relevant tables using vector search
        relevant_tables = self.graph_builder.vector_search(query_text, limit=top_k)
        
        # Step 2: For each relevant table, find related tables if requested
        if include_related and relevant_tables:
            for i, table in enumerate(relevant_tables[:3]):  # Limit to top 3 for related lookups
                related = self.graph_builder.find_related_tables(table['id'])
                relevant_tables[i]['related_tables'] = related
        
        # Step 3: Get full details for the top table
        if relevant_tables:
            top_table_id = relevant_tables[0]['id']
            top_table_details = self.graph_builder.get_table_details(top_table_id)
            
            if top_table_details:
                relevant_tables[0]['details'] = top_table_details
                
                # Get columns for the top table (new)
                columns = self.graph_builder.get_columns_for_table(top_table_id)
                if columns:
                    relevant_tables[0]['columns'] = columns
        
        # Step 4: Prepare response (without SQL generation)
        result = {
            'query': query_text,
            'tables': relevant_tables
        }
        
        return result
    
    def column_query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute a RAG query specifically for columns
        
        Args:
            query_text: Natural language query
            top_k: Number of most relevant columns to return
            
        Returns:
            Dictionary with query results
        """
        # Step 1: Find relevant columns using vector search
        relevant_columns = self.graph_builder.vector_search_columns(query_text, limit=top_k)
        
        # Step 2: Get full details for each column
        for i, column in enumerate(relevant_columns):
            column_details = self.graph_builder.get_column_details(column['id'])
            if column_details:
                relevant_columns[i].update(column_details)
        
        # Step 3: Prepare response
        result = {
            'query': query_text,
            'columns': relevant_columns
        }
        
        return result
    
    def hybrid_query(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """Execute a RAG query against both tables and columns
        
        Args:
            query_text: Natural language query
            top_k: Number of most relevant items to return for each type
            
        Returns:
            Dictionary with both table and column results
        """
        # Get table results
        table_results = self.query(query_text, top_k=top_k, include_related=True)
        
        # Get column results
        column_results = self.column_query(query_text, top_k=top_k)
        
        # Combine results
        result = {
            'query': query_text,
            'tables': table_results['tables'],
            'columns': column_results['columns']
        }
        
        return result
    
    def get_table_columns(self, table_id: str) -> Dict[str, Any]:
        """Get all columns for a specific table with details
        
        Args:
            table_id: ID of the table
            
        Returns:
            Dictionary with table details and its columns
        """
        # Get table details
        table_details = self.graph_builder.get_table_details(table_id)
        
        # Get columns for the table
        columns = self.graph_builder.get_columns_for_table(table_id)
        
        # Prepare response
        result = {
            'table': table_details,
            'columns': columns
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize graph builder
    builder = TableGraphBuilder(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    
    # Initialize RAG engine
    rag_engine = TableRAGEngine(builder)
    
    # Test table query
    result = rag_engine.query("Find customer sales information")
    print(f"Query: {result['query']}")
    print(f"Found {len(result['tables'])} relevant tables")
    
    # Display top result
    if result['tables']:
        top = result['tables'][0]
        print(f"\nTop table: {top['name']} ({top['similarity']:.4f})")
        print(f"Description: {top['description']}")
        
        if 'columns' in top:
            print(f"\nColumns: {len(top['columns'])}")
            for i, col in enumerate(top['columns'][:3]):
                pk_marker = "🔑 " if col['is_primary_key'] else "  "
                fk_marker = "🔗 " if col['is_foreign_key'] else "  "
                print(f"  {i+1}. {pk_marker}{fk_marker}{col['name']} ({col['datatype']})")
        
        if 'related_tables' in top:
            print(f"\nRelated tables: {len(top['related_tables'])}")
            for i, related in enumerate(top['related_tables'][:3]):
                print(f"  {i+1}. {related['name']}: {related['description']}")
    
    # Test column query
    print("\n--- Column Query Test ---")
    col_result = rag_engine.column_query("Find customer identifier fields")
    print(f"Query: {col_result['query']}")
    print(f"Found {len(col_result['columns'])} relevant columns")
    
    # Display top column results
    for i, col in enumerate(col_result['columns'][:3]):
        print(f"\n{i+1}. {col['name']} ({col['similarity']:.4f})")
        print(f"   Table: {col['table_id']}")
        print(f"   Data Type: {col['datatype']}")
        if col['description']:
            print(f"   Description: {col['description']}")
    
    # Clean up
    builder.close()