import json
import logging
import asyncio
from typing import Dict, List, Set, Tuple, Any, Optional
from graph_builder import TableGraphBuilder
import os
from dotenv import load_dotenv
from qwen import llm

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedViewColumnRelationshipBuilder:
    """Build REFERENCES_COLUMN relationships with KG vector search and LLM verification"""
    
    def __init__(self, neo4j_uri: str, username: str, password: str, 
                 ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text"):
        """Initialize the relationship builder"""
        # Use TableGraphBuilder for all operations
        self.graph_builder = TableGraphBuilder(
            uri=neo4j_uri,
            username=username,
            password=password,
            ollama_url=ollama_url,
            embedding_model=embedding_model
        )
    
    def check_column_exists(self, column_id: str) -> bool:
        """Check if a column exists in the knowledge graph"""
        try:
            with self.graph_builder.driver.session() as session:
                result = session.run(
                    "MATCH (c:COLUMN {id: $column_id}) RETURN count(c) > 0 as exists",
                    column_id=column_id
                )
                return result.single()['exists']
        except Exception as e:
            logger.error(f"Error checking column {column_id}: {e}")
            return False
    
    async def verify_column_match_with_llm(self, extracted_column: str, matched_column: str, 
                                          similarity: float, tables_used: List[str]) -> bool:
        """Use LLM to verify if the column match is correct"""
        prompt = f"""
        You are a database expert. Determine if these two column names refer to the same column:
        
        Column from SQL query: {extracted_column}
        Column found in KG: {matched_column}
        Similarity score: {similarity:.3f}
        Context: This column should belong to one of these tables: {', '.join(tables_used)}
        
        Common patterns to consider:
        - Aliases: ROW_ID might be an alias for ROWID
        - Underscores vs no underscores: LAST_UPDATE_DATE vs LASTUPDATEDATE
        - Abbreviations: CUST_ID vs CUSTOMER_ID
        - Case differences should be ignored
        
        Respond with ONLY "true" if they refer to the same column, or "false" if they don't.
        """
        
        try:
            response = await llm.ainvoke(prompt)
            
            # Handle AIMessage object
            if hasattr(response, 'content'):
                response_text = response.content.strip().lower()
            else:
                response_text = str(response).strip().lower()
            
            return response_text == "true"
            
        except Exception as e:
            logger.error(f"Error verifying column match with LLM: {e}")
            # If LLM fails, use similarity threshold
            return similarity > 0.85
    
    def create_view_column_relationship(self, view_id: str, column_id: str) -> bool:
        """Create REFERENCES_COLUMN relationship between view and column"""
        try:
            with self.graph_builder.driver.session() as session:
                cypher = """
                MATCH (v:VIEW {id: $view_id})
                MATCH (c:COLUMN {id: $column_id})
                MERGE (v)-[r:REFERENCES_COLUMN]->(c)
                RETURN r
                """
                
                result = session.run(
                    cypher,
                    view_id=view_id,
                    column_id=column_id
                )
                
                record = result.single()
                if record:
                    logger.debug(f"Created/matched relationship: {view_id} -> {column_id}")
                    return True
                else:
                    logger.warning(f"Failed to create relationship: {view_id} -> {column_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating relationship {view_id} -> {column_id}: {e}")
            return False
    
    async def process_view_columns_enhanced(self, extracted_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Process extracted view columns with KG vector search and LLM verification"""
        stats = {
            'total_views': 0,
            'total_columns': 0,
            'direct_matches': 0,
            'vector_search_matches': 0,
            'llm_verified': 0,
            'relationships_created': 0,
            'columns_not_found': [],
            'vector_match_details': []
        }
        
        for view_id, view_data in extracted_data.items():
            stats['total_views'] += 1
            view_name = view_data['view_name']
            column_mappings = view_data['column_mappings']
            tables_used = view_data['tables_used']
            
            logger.info(f"Processing view: {view_name} ({view_id})")
            
            view_columns_found = set()
            view_columns_vector_matched = set()
            view_columns_not_found = set()
            
            for col_mapping in column_mappings:
                column_name = col_mapping['column_name']
                potential_ids = col_mapping['potential_column_ids']
                stats['total_columns'] += 1
                
                # First, try direct match
                column_found = False
                for potential in potential_ids:
                    column_id = potential['column_id']
                    
                    if self.check_column_exists(column_id):
                        if self.create_view_column_relationship(view_id, column_id):
                            stats['direct_matches'] += 1
                            stats['relationships_created'] += 1
                            view_columns_found.add(f"{column_name} ({column_id})")
                            column_found = True
                            break
                
                # If not found, use table-restricted vector search
                if not column_found:
                    logger.info(f"  Column '{column_name}' not found directly, using table-restricted vector search...")
                    logger.info(f"  Searching in tables: {tables_used}")
                    
                    # Use the new method to search only within the view's tables
                    search_results = self.graph_builder.vector_search_columns_in_tables(
                        query_text=column_name,
                        table_ids=tables_used,
                        limit=3  # Get top 1 match
                    )
                    
                    logger.info(search_results)

                    if search_results:
                        best_match = search_results[0]
                        matched_column_name = best_match['name']
                        matched_column_id = best_match['id']
                        matched_table_id = best_match['table_id']
                        similarity = best_match['similarity']
                        
                        logger.info(f"  Found match: {matched_column_name} in table {matched_table_id} (similarity: {similarity:.3f})")
                        
                        # Verify with LLM
                        llm_verified = await self.verify_column_match_with_llm(
                            column_name, 
                            matched_column_name,
                            similarity,
                            tables_used
                        )
                        
                        if llm_verified:
                            logger.info(f"  ✓ Vector search match verified: {column_name} → {matched_column_name} (similarity: {similarity:.3f})")
                            
                            if self.create_view_column_relationship(view_id, matched_column_id):
                                stats['vector_search_matches'] += 1
                                stats['llm_verified'] += 1
                                stats['relationships_created'] += 1
                                view_columns_vector_matched.add(f"{column_name} → {matched_column_name} ({matched_column_id})")
                                
                                stats['vector_match_details'].append({
                                    'view': view_name,
                                    'extracted_column': column_name,
                                    'matched_column': matched_column_name,
                                    'matched_column_id': matched_column_id,
                                    'matched_table_id': matched_table_id,
                                    'similarity': similarity,
                                    'llm_verified': True
                                })
                                column_found = True
                        else:
                            logger.info(f"  ✗ LLM rejected match: {column_name} ≠ {matched_column_name}")
                    else:
                        logger.info(f"  No matches found in tables {tables_used}")
                
                if not column_found:
                    view_columns_not_found.add(column_name)
                    stats['columns_not_found'].append({
                        'view': view_name,
                        'column': column_name,
                        'tried_ids': [p['column_id'] for p in potential_ids],
                        'searched_tables': tables_used
                    })
            
            # Log summary for this view
            if view_columns_found:
                logger.info(f"  Direct matches: {len(view_columns_found)}")
            if view_columns_vector_matched:
                logger.info(f"  Vector search matches: {len(view_columns_vector_matched)}")
            if view_columns_not_found:
                logger.warning(f"  Still missing: {view_columns_not_found}")
        
        return stats
    
    def close(self):
        """Close the Neo4j connection"""
        if self.graph_builder:
            self.graph_builder.close()

async def main():
    """Main function to update knowledge graph with enhanced matching"""
    # Load extracted columns data
    try:
        with open('view_columns_extracted.json', 'r') as f:
            extracted_data = json.load(f)
    except FileNotFoundError:
        logger.error("view_columns_extracted.json not found. Run extract_view_columns.py first.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in view_columns_extracted.json: {e}")
        return
    
    # Initialize relationship builder
    builder = EnhancedViewColumnRelationshipBuilder(
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD', 'password'),
        ollama_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        embedding_model=os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
    )
    
    try:
        # Process views and create relationships
        stats = await builder.process_view_columns_enhanced(extracted_data)
        
        # Save detailed results
        with open('view_column_update_enhanced_results.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\n=== Enhanced Update Summary ===")
        print(f"Total views processed: {stats['total_views']}")
        print(f"Total columns processed: {stats['total_columns']}")
        print(f"Direct matches: {stats['direct_matches']}")
        print(f"Vector search matches (LLM verified): {stats['vector_search_matches']}")
        print(f"Total relationships created: {stats['relationships_created']}")
        print(f"Success rate: {stats['relationships_created']/stats['total_columns']*100:.1f}%")
        
        if stats['vector_match_details']:
            print(f"\n=== Vector Search Match Examples (first 10) ===")
            for match in stats['vector_match_details'][:10]:
                print(f"\n{match['view']}:")
                print(f"  {match['extracted_column']} → {match['matched_column']} (similarity: {match['similarity']:.3f})")
        
        remaining_missing = len(stats['columns_not_found'])
        if remaining_missing > 0:
            print(f"\n=== Still Missing ({remaining_missing} columns) ===")
            # Group by view
            by_view = {}
            for item in stats['columns_not_found']:
                view = item['view']
                if view not in by_view:
                    by_view[view] = []
                by_view[view].append(item['column'])
            
            for view, columns in list(by_view.items())[:5]:
                print(f"\n{view}:")
                for col in columns[:5]:
                    print(f"  - {col}")
        
        print(f"\nDetailed results saved to: view_column_update_enhanced_results.json")
        
    finally:
        builder.close()

if __name__ == "__main__":
    asyncio.run(main())