import logging
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
import json
import traceback
from datetime import datetime
from models import TableNode, Relationship, ColumnNode, ViewNode
from embedder import OllamaEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TableGraphBuilder:
    """Builder for knowledge graph of Oracle tables with vector embeddings"""
    
    def __init__(self, uri: str, username: str, password: str, 
                 vector_dimensions: int = 768,
                 ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text"):
        """Initialize the graph builder
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            vector_dimensions: Dimensions of the embedding vectors
            ollama_url: URL for Ollama API
            embedding_model: Name of embedding model to use
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.vector_dimensions = vector_dimensions
        
        # Initialize embedder
        self.embedder = OllamaEmbedder(base_url=ollama_url, model=embedding_model)
        
        # Initialize Neo4j schema
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes"""
        try:
            with self.driver.session() as session:
                # Create constraint for TABLE nodes
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:TABLE)
                    REQUIRE n.id IS UNIQUE
                """)
                
                # Create constraint for COLUMN nodes
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:COLUMN)
                    REQUIRE n.id IS UNIQUE
                """)
                
                # Create constraint for VIEW nodes
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:VIEW)
                    REQUIRE n.id IS UNIQUE
                """)
                
                # Create vector index for table embeddings
                # Check Neo4j version first (vector indexes require Neo4j 5.11+)
                result = session.run("RETURN apoc.version()")
                neo4j_version = result.single()[0]
                
                if neo4j_version.startswith('5.') and float(neo4j_version.split('.')[1]) >= 11:
                    # Create vector index for tables
                    try:
                        session.run(f"""
                            CREATE VECTOR INDEX table_embedding IF NOT EXISTS
                            FOR (n:TABLE)
                            ON n.embedding
                            OPTIONS {{indexConfig: {{
                                `vector.dimensions`: {self.vector_dimensions},
                                `vector.similarity_function`: 'cosine'
                            }}}}
                        """)
                        logger.info("Created vector index for table embeddings")
                        
                        # Create vector index for columns
                        session.run(f"""
                            CREATE VECTOR INDEX column_embedding IF NOT EXISTS
                            FOR (n:COLUMN)
                            ON n.embedding
                            OPTIONS {{indexConfig: {{
                                `vector.dimensions`: {self.vector_dimensions},
                                `vector.similarity_function`: 'cosine'
                            }}}}
                        """)
                        logger.info("Created vector index for column embeddings")
                        
                        # Create vector index for views
                        session.run(f"""
                            CREATE VECTOR INDEX view_embedding IF NOT EXISTS
                            FOR (n:VIEW)
                            ON n.embedding
                            OPTIONS {{indexConfig: {{
                                `vector.dimensions`: {self.vector_dimensions},
                                `vector.similarity_function`: 'cosine'
                            }}}}
                        """)
                        logger.info("Created vector index for view embeddings")
                    except Exception as e:
                        logger.warning(f"Failed to create vector indexes: {str(e)}")
                        logger.warning("Vector search will not be available")
                else:
                    logger.warning(f"Neo4j version {neo4j_version} may not support vector indexes")
                    logger.warning("Vector search will not be available")
                
                logger.info("Initialized Neo4j schema")
                
        except Exception as e:
            logger.error(f"Error initializing schema: {str(e)}")
            logger.error(traceback.format_exc())

    def create_table_node(self, table: TableNode) -> bool:
        """Create a table node in Neo4j
        
        Args:
            table: TableNode instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding if not already present
            if not table.embedding:
                # Extract columns for embedding
                columns = None
                if table.columns:
                    # For embedding purposes, we'll use a list of dicts
                    columns = [
                        {
                            'name': col.name,
                            'datatype': col.datatype,
                            'comments': col.comments
                        }
                        for col in table.columns[:10]  # Limit to 10 columns
                    ]
                
                # Get embedding
                embedding = self.embedder.embed_table(
                    table_name=table.name,
                    module=table.module,
                    submodule=table.submodule,
                    description=table.description or "",
                    primary_key=table.primary_key.dict() if table.primary_key else None,
                    columns=columns
                )
                
                if embedding:
                    table.embedding = embedding
                    logger.info(f"Generated embedding for table {table.name}")
                else:
                    logger.warning(f"Failed to generate embedding for table {table.name}")
            
            # Convert table to Neo4j-compatible dictionary
            properties = table.dict(exclude={'type', 'created_at', 'updated_at'})
            
            with self.driver.session() as session:
                cypher = """
                MERGE (n:TABLE {id: $id})
                ON CREATE SET 
                    n = $properties,
                    n.created_at = datetime($created_at),
                    n.updated_at = datetime($updated_at)
                ON MATCH SET 
                    n += $properties,
                    n.updated_at = datetime($updated_at)
                RETURN n
                """
                
                result = session.run(
                    cypher,
                    id=table.id,
                    properties=properties,
                    created_at=table.created_at.isoformat(),
                    updated_at=datetime.utcnow().isoformat()
                )
                
                # Check result
                record = result.single()
                if record:
                    logger.info(f"Created/updated table node: {table.id}")
                    return True
                else:
                    logger.warning(f"Failed to create/update table node: {table.id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating table node {table.id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def create_column_node(self, column: ColumnNode) -> bool:
        """Create a column node in Neo4j
        
        Args:
            column: ColumnNode instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding if not already present
            if not column.embedding:
                # Get embedding
                embedding = self.embedder.embed_column(
                    column_name=column.name,
                    datatype=column.datatype,
                    table_name=column.table_id,  # Using table_id, assuming it contains table name
                    description=column.description or "",
                    is_primary_key=column.is_primary_key,
                    is_foreign_key=column.is_foreign_key,
                    references_column=column.references_column or ""
                )
                
                if embedding:
                    column.embedding = embedding
                    logger.info(f"Generated embedding for column {column.name}")
                else:
                    logger.warning(f"Failed to generate embedding for column {column.name}")
            
            # Convert column to Neo4j-compatible dictionary
            properties = column.dict(exclude={'type', 'created_at', 'updated_at'})
            
            with self.driver.session() as session:
                # Create column node
                cypher = """
                MERGE (c:COLUMN {id: $id})
                ON CREATE SET 
                    c = $properties,
                    c.created_at = datetime($created_at),
                    c.updated_at = datetime($updated_at)
                ON MATCH SET 
                    c += $properties,
                    c.updated_at = datetime($updated_at)
                RETURN c
                """
                
                result = session.run(
                    cypher,
                    id=column.id,
                    properties=properties,
                    created_at=column.created_at.isoformat(),
                    updated_at=datetime.utcnow().isoformat()
                )
                
                # Check if column was created
                column_record = result.single()
                if not column_record:
                    logger.warning(f"Failed to create/update column node: {column.id}")
                    return False
                
                logger.info(f"Created/updated column node: {column.id}")
                
                # Create relationship between table and column
                cypher = """
                MATCH (t:TABLE {id: $table_id})
                MATCH (c:COLUMN {id: $column_id})
                MERGE (t)-[r:HAS_COLUMN]->(c)
                RETURN r
                """
                
                result = session.run(
                    cypher,
                    table_id=column.table_id,
                    column_id=column.id
                )
                
                # Check if relationship was created/matched
                rel_record = result.single()
                if rel_record:
                    logger.info(f"Connected column {column.id} to table {column.table_id}")
                    return True
                else:
                    logger.warning(f"Failed to connect column {column.id} to table {column.table_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating column node {column.id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def create_column_relationships(self, column: ColumnNode) -> bool:
        """Create relationships between columns (foreign key relationships)
        
        Args:
            column: ColumnNode instance with foreign key information
            
        Returns:
            True if successful, False otherwise
        """
        if not column.is_foreign_key or not column.references_column:
            return True  # Nothing to do
            
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (source:COLUMN {id: $source_id})
                MATCH (target:COLUMN {id: $target_id})
                MERGE (source)-[r:REFERENCES]->(target)
                RETURN r
                """
                
                result = session.run(
                    cypher,
                    source_id=column.id,
                    target_id=column.references_column
                )
                
                # Check if relationship exists (created or matched)
                record = result.single()
                if record:
                    logger.info(f"Created/matched column relationship: {column.id} -> {column.references_column}")
                    return True
                else:
                    logger.warning(f"Failed to create column relationship: {column.id} -> {column.references_column}")
                    return False
                
        except Exception as e:
            logger.error(f"Error creating column relationship {column.id} -> {column.references_column}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def create_view_node(self, view: ViewNode) -> bool:
        """Create a view node in Neo4j
        
        Args:
            view: ViewNode instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding if not already present
            if not view.embedding:
                # Only embed the description
                if view.description:
                    embedding = self.embedder.get_embedding(view.description)
                    if embedding:
                        view.embedding = embedding
                        logger.info(f"Generated embedding for view {view.name}")
                    else:
                        logger.warning(f"Failed to generate embedding for view {view.name}")
            
            # Convert view to Neo4j-compatible dictionary
            properties = view.dict(exclude={'type', 'created_at', 'updated_at'})
            
            with self.driver.session() as session:
                cypher = """
                MERGE (n:VIEW {id: $id})
                ON CREATE SET 
                    n = $properties,
                    n.created_at = datetime($created_at),
                    n.updated_at = datetime($updated_at)
                ON MATCH SET 
                    n += $properties,
                    n.updated_at = datetime($updated_at)
                RETURN n
                """
                
                result = session.run(
                    cypher,
                    id=view.id,
                    properties=properties,
                    created_at=view.created_at.isoformat(),
                    updated_at=datetime.utcnow().isoformat()
                )
                
                # Check result
                record = result.single()
                if record:
                    logger.info(f"Created/updated view node: {view.id}")
                    return True
                else:
                    logger.warning(f"Failed to create/update view node: {view.id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating view node {view.id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def create_view_relationships(self, view_id: str, table_ids: List[str]) -> bool:
        """Create relationships between a view and its underlying tables
        
        Args:
            view_id: ID of the view
            table_ids: List of table IDs used by the view
            
        Returns:
            True if all relationships created successfully
        """
        success_count = 0
        
        for table_id in table_ids:
            try:
                with self.driver.session() as session:
                    # Create the relationship in a single query
                    cypher = """
                    MATCH (v:VIEW {id: $view_id})
                    MATCH (t:TABLE {id: $table_id})
                    MERGE (v)-[r:USES_TABLE]->(t)
                    RETURN count(r) as rel_count
                    """
                    
                    result = session.run(
                        cypher,
                        view_id=view_id,
                        table_id=table_id.lower()  # Ensure lowercase for consistency
                    )
                    
                    # Check if relationship exists (created or matched)
                    record = result.single()
                    if record and record['rel_count'] > 0:
                        logger.info(f"Created/matched view relationship: {view_id} -> {table_id}")
                        success_count += 1
                    else:
                        logger.warning(f"Failed to create view relationship: {view_id} -> {table_id} (nodes might not exist)")
                        
            except Exception as e:
                logger.error(f"Error creating view relationship {view_id} -> {table_id}: {str(e)}")
                logger.error(traceback.format_exc())
        
        return success_count == len(table_ids)

    def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship between tables
        
        Args:
            relationship: Relationship instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (source:TABLE {id: $source_id})
                MATCH (target:TABLE {id: $target_id})
                MERGE (source)-[r:REFERENCES]->(target)
                ON CREATE SET 
                    r = $properties,
                    r.created_at = datetime($created_at)
                ON MATCH SET 
                    r += $properties
                RETURN r
                """
                
                result = session.run(
                    cypher,
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    properties=relationship.properties,
                    created_at=relationship.created_at.isoformat()
                )
                
                # Check if relationship exists (created or matched)
                record = result.single()
                if record:
                    logger.info(f"Created/matched relationship: {relationship.source_id} -> {relationship.target_id}")
                    return True
                else:
                    logger.warning(f"Failed to create relationship: {relationship.source_id} -> {relationship.target_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Error creating relationship {relationship.source_id} -> {relationship.target_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def vector_search(self, query_text: str, limit: int = 5, node_type: str = "TABLE") -> List[Dict[str, Any]]:
        """Search for nodes by vector similarity
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            node_type: Type of node to search (TABLE, COLUMN, or VIEW)
            
        Returns:
            List of matching nodes with similarity scores
        """
        try:
            # Get embedding for query
            query_embedding = self.embedder.get_embedding(query_text)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            with self.driver.session() as session:
                # Check if vector index exists
                if node_type == "TABLE":
                    index_name = "table_embedding"
                elif node_type == "COLUMN":
                    index_name = "column_embedding"
                elif node_type == "VIEW":
                    index_name = "view_embedding"
                else:
                    logger.error(f"Invalid node type: {node_type}")
                    return []
                
                index_check = session.run("""
                    SHOW INDEXES
                    YIELD name, type
                    WHERE name = $index_name AND type = 'VECTOR'
                    RETURN count(*) > 0 AS exists
                """, index_name=index_name)
                
                vector_index_exists = index_check.single()[0]
                
                if vector_index_exists:
                    # Use vector index
                    cypher = f"""
                    CALL db.index.vector.queryNodes('{index_name}', $limit, $embedding)
                    YIELD node, score
                    RETURN node.id AS id, node.name AS name, 
                        {
                            'node.module AS module, node.submodule AS submodule' 
                            if node_type in ['TABLE', 'VIEW'] else 
                            'node.datatype AS datatype, node.table_id AS table_id'
                        },
                        node.description AS description,
                        {'node.sql_query AS sql_query,' if node_type == 'VIEW' else ''}
                        score AS similarity
                    ORDER BY similarity DESC
                    """
                else:
                    # Fallback to manual similarity calculation
                    logger.warning(f"Vector index not found, using manual similarity calculation (slower)")
                    cypher = f"""
                    MATCH (n:{node_type})
                    WHERE n.embedding IS NOT NULL
                    WITH n, gds.similarity.cosine(n.embedding, $embedding) AS similarity
                    WHERE similarity > 0.5
                    RETURN n.id AS id, n.name AS name, 
                        {
                            'n.module AS module, n.submodule AS submodule' 
                            if node_type in ['TABLE', 'VIEW'] else 
                            'n.datatype AS datatype, n.table_id AS table_id'
                        },
                        n.description AS description,
                        {'n.sql_query AS sql_query,' if node_type == 'VIEW' else ''}
                        similarity
                    ORDER BY similarity DESC
                    LIMIT $limit
                    """
                
                result = session.run(
                    cypher,
                    embedding=query_embedding,
                    limit=limit
                )
                
                nodes = []
                for record in result:
                    node_data = {
                        'id': record['id'],
                        'name': record['name'],
                        'description': record['description'],
                        'similarity': record['similarity']
                    }
                    
                    # Add node type specific fields
                    if node_type in ['TABLE', 'VIEW']:
                        node_data.update({
                            'module': record['module'],
                            'submodule': record['submodule']
                        })
                        if node_type == 'VIEW':
                            node_data['sql_query'] = record.get('sql_query')
                    else:  # COLUMN
                        node_data.update({
                            'datatype': record['datatype'],
                            'table_id': record['table_id']
                        })
                    
                    nodes.append(node_data)
                
                return nodes
                
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def vector_search_columns(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for columns by vector similarity
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            
        Returns:
            List of matching columns with similarity scores
        """
        return self.vector_search(query_text, limit, node_type="COLUMN")
    
    def get_columns_for_table(self, table_id: str) -> List[Dict[str, Any]]:
        """Get all columns for a specific table
        
        Args:
            table_id: ID of the table
            
        Returns:
            List of columns with details
        """
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (t:TABLE {id: $table_id})-[:HAS_COLUMN]->(c:COLUMN)
                RETURN 
                    c.id AS id,
                    c.name AS name,
                    c.datatype AS datatype,
                    c.description AS description,
                    c.is_primary_key AS is_primary_key,
                    c.is_foreign_key AS is_foreign_key,
                    c.references_column AS references_column
                ORDER BY c.is_primary_key DESC, c.name
                """
                
                result = session.run(
                    cypher,
                    table_id=table_id
                )
                
                columns = []
                for record in result:
                    columns.append({
                        'id': record['id'],
                        'name': record['name'],
                        'datatype': record['datatype'],
                        'description': record['description'],
                        'is_primary_key': record['is_primary_key'],
                        'is_foreign_key': record['is_foreign_key'],
                        'references_column': record['references_column'],
                    })
                
                return columns
                
        except Exception as e:
            logger.error(f"Error getting columns for table {table_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_column_details(self, column_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific column
        
        Args:
            column_id: ID of the column
            
        Returns:
            Dictionary with column details or None if not found
        """
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (c:COLUMN {id: $column_id})
                OPTIONAL MATCH (c)-[:REFERENCES]->(ref:COLUMN)
                RETURN
                    c.id AS id,
                    c.name AS name,
                    c.datatype AS datatype,
                    c.table_id AS table_id,
                    c.description AS description,
                    c.length AS length,
                    c.precision AS precision,
                    c.is_nullable AS is_nullable,
                    c.is_primary_key AS is_primary_key,
                    c.is_foreign_key AS is_foreign_key,
                    c.references_column AS references_column,
                    ref.name AS referenced_column_name,
                    ref.table_id AS referenced_table_id
                """
                
                result = session.run(
                    cypher,
                    column_id=column_id
                )
                
                record = result.single()
                if not record:
                    return None
                
                return dict(record)
                
        except Exception as e:
            logger.error(f"Error getting column details for {column_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def find_related_tables(self, table_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """Find tables related to the given table through relationships
        
        Args:
            table_id: ID of the table
            depth: Traversal depth (1 = direct connections only)
            
        Returns:
            List of related tables with relationship information
        """
        try:
            with self.driver.session() as session:
                # Query for related tables
                cypher = """
                MATCH path = (source:TABLE {id: $table_id})-[r:REFERENCES*1..2]-(related:TABLE)
                WHERE related.id <> $table_id
                WITH related, [rel in relationships(path) | {
                    source: startNode(rel).id,
                    target: endNode(rel).id,
                    foreign_key: rel.foreign_key_column
                }] AS rels
                RETURN DISTINCT
                    related.id AS id,
                    related.name AS name,
                    related.module AS module,
                    related.submodule AS submodule,
                    related.description AS description,
                    rels AS relationships
                LIMIT 20
                """
                
                result = session.run(
                    cypher,
                    table_id=table_id
                )
                
                related_tables = []
                for record in result:
                    related_tables.append({
                        'id': record['id'],
                        'name': record['name'],
                        'module': record['module'],
                        'submodule': record['submodule'],
                        'description': record['description'],
                        'relationships': record['relationships']
                    })
                
                return related_tables
                
        except Exception as e:
            logger.error(f"Error finding related tables for {table_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def vector_search_views(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for views by vector similarity
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            
        Returns:
            List of matching views with similarity scores
        """
        return self.vector_search(query_text, limit, node_type="VIEW")

    def get_table_details(self, table_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific table
        
        Args:
            table_id: ID of the table
            
        Returns:
            Dictionary with table details or None if not found
        """
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (t:TABLE {id: $table_id})
                RETURN
                    t.id AS id,
                    t.name AS name,
                    t.module AS module,
                    t.submodule AS submodule,
                    t.description AS description,
                    t.columns AS columns,
                    t.primary_key AS primary_key,
                    t.indexes AS indexes,
                    t.details AS details
                """
                
                result = session.run(
                    cypher,
                    table_id=table_id
                )
                
                record = result.single()
                if not record:
                    return None
                
                # Parse JSON strings back to objects
                table_details = dict(record)
                for field in ['columns', 'primary_key', 'indexes', 'details']:
                    if table_details.get(field):
                        try:
                            table_details[field] = json.loads(table_details[field])
                        except:
                            pass
                
                return table_details
                
        except Exception as e:
            logger.error(f"Error getting table details for {table_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

# Example usage
if __name__ == "__main__":
    builder = TableGraphBuilder(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    
    # Test vector search
    results = builder.vector_search("customer sales data")
    print(f"Found {len(results)} tables")
    for table in results:
        print(f"{table['name']} ({table['similarity']:.4f}): {table['description']}")
    
    builder.close()