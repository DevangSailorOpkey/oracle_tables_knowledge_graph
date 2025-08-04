import argparse
import os
import logging
import sys
from typing import List, Dict, Any, Optional
import json
from graph_builder import TableGraphBuilder
from json_parser import OracleTableParser
from rag_engine import TableRAGEngine
from models import TableNode, Relationship, ColumnNode, ViewNode
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OracleTablesCLI:
    """Command-line interface for Oracle Tables Knowledge Graph RAG"""
    
    def __init__(self):
        """Initialize the CLI with argument parser"""
        self.parser = self._create_parser()
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the command-line argument parser"""
        parser = argparse.ArgumentParser(
            description='Oracle Tables Knowledge Graph RAG CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Global options
        parser.add_argument(
            '--neo4j-uri',
            default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            help='Neo4j connection URI (default: bolt://localhost:7687)'
        )
        parser.add_argument(
            '--username',
            default=os.getenv('NEO4J_USERNAME', 'neo4j'),
            help='Neo4j username (default: neo4j)'
        )
        parser.add_argument(
            '--password',
            default=os.getenv('NEO4J_PASSWORD', 'password'),
            help='Neo4j password (default: password)'
        )
        parser.add_argument(
            '--data-dir',
            default='Tables/',
            help='Directory containing JSON data files (default: Tables/)'
        )
        parser.add_argument(
            '--ollama-url',
            default=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            help='Ollama API URL (default: http://localhost:11434)'
        )
        parser.add_argument(
            '--embed-model',
            default=os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text'),
            help='Embedding model to use (default: nomic-embed-text)'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose logging'
        )
        
        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Commands')
        
        # Load command
        # Table
        load_parser = subparsers.add_parser('load', help='Load data into knowledge graph')
        load_parser.add_argument(
            '--files',
            nargs='+',
            default=["Financials.json", "HCM.json", "SCM.json", "Project Management.json", "Sales and Fusion Service.json"],
            help='JSON files to load (default: all modules)'
        )
        # View
        load_views_parser = subparsers.add_parser('load-views', help='Load views from JSON')
        load_views_parser.add_argument('file', type=str, help='Path to JSON file')

        # Query command
        query_parser = subparsers.add_parser('query', help='Query the knowledge graph')
        query_parser.add_argument(
            'query_text',
            help='Natural language query'
        )
        query_parser.add_argument(
            '--top-k',
            type=int,
            default=5,
            help='Number of top results to return (default: 5)'
        )
        query_parser.add_argument(
            '--no-related',
            action='store_true',
            help='Disable inclusion of related tables'
        )
        query_parser.add_argument(
            '--format',
            choices=['text', 'json'],
            default='text',
            help='Output format (default: text)'
        )
        query_parser.add_argument(
            '--node-type',
            choices=['table', 'column', 'view', 'both'],
            default='table',
            help='Type of node to search (default: table)'
        )
        
        # Column command
        column_parser = subparsers.add_parser('column', help='Query for columns')
        column_subparsers = column_parser.add_subparsers(dest='column_command', help='Column commands')
        
        # Column search command
        column_search_parser = column_subparsers.add_parser('search', help='Search for columns')
        column_search_parser.add_argument(
            'query_text',
            help='Natural language query for columns'
        )
        column_search_parser.add_argument(
            '--top-k',
            type=int,
            default=100,
            help='Number of top results to return (default: 5)'
        )
        column_search_parser.add_argument(
            '--format',
            choices=['text', 'json'],
            default='text',
            help='Output format (default: text)'
        )

        # In cli.py, modify the column_search_parser:
        column_search_parser.add_argument(
            '--table-id',
            help='Optionally limit the search to a specific table'
        )
        
        # Column list command
        column_list_parser = column_subparsers.add_parser('list', help='List columns for a table')
        column_list_parser.add_argument(
            'table_id',
            help='ID of the table to list columns for'
        )
        column_list_parser.add_argument(
            '--format',
            choices=['text', 'json'],
            default='text',
            help='Output format (default: text)'
        )
        
        # Column details command
        column_details_parser = column_subparsers.add_parser('details', help='Get details for a column')
        column_details_parser.add_argument(
            'column_id',
            help='ID of the column to get details for'
        )
        column_details_parser.add_argument(
            '--format',
            choices=['text', 'json'],
            default='text',
            help='Output format (default: text)'
        )
        
        # View command
        view_parser = subparsers.add_parser('view', help='Query and manage views')
        view_subparsers = view_parser.add_subparsers(dest='view_command', help='View commands')

        # View search command
        view_search_parser = view_subparsers.add_parser('search', help='Search for views')
        view_search_parser.add_argument(
            'query_text',
            help='Natural language query for views'
        )
        view_search_parser.add_argument(
            '--top-k',
            type=int,
            default=5,
            help='Number of top results to return (default: 5)'
        )
        view_search_parser.add_argument(
            '--format',
            choices=['text', 'json'],
            default='text',
            help='Output format (default: text)'
        )

        # Add view command
        view_add_parser = view_subparsers.add_parser('add', help='Add a view to the knowledge graph')
        view_add_parser.add_argument(
            '--id',
            required=True,
            help='View ID'
        )
        view_add_parser.add_argument(
            '--name',
            required=True,
            help='View name'
        )
        view_add_parser.add_argument(
            '--module',
            required=True,
            help='Module name'
        )
        view_add_parser.add_argument(
            '--submodule',
            required=True,
            help='Submodule name'
        )
        view_add_parser.add_argument(
            '--description',
            required=True,
            help='View description'
        )
        view_add_parser.add_argument(
            '--sql-query',
            required=True,
            help='SQL query for the view'
        )
        view_add_parser.add_argument(
            '--tables-used',
            nargs='+',
            required=True,
            help='List of tables used by the view'
        )

        # Info command
        info_parser = subparsers.add_parser('info', help='Get information about the knowledge graph')
        
        return parser
    
    def run(self):
        """Run the CLI application"""
        args = self.parser.parse_args()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        if not args.command:
            self.parser.print_help()
            sys.exit(1)
            
        try:
            # Initialize graph builder
            builder = TableGraphBuilder(
                uri=args.neo4j_uri,
                username=args.username,
                password=args.password,
                ollama_url=args.ollama_url,
                embedding_model=args.embed_model
            )
            
            if args.command == 'load':
                self._handle_load(args, builder)
            elif args.command == 'query':
                self._handle_query(args, builder)
            elif args.command == 'column':
                self._handle_column(args, builder)
            elif args.command == 'view':
                self._handle_view(args, builder)
            elif args.command == 'info':
                self._handle_info(args, builder)
            elif args.command == 'load-views':
                self._handle_load_views(args, builder)

            # Clean up
            builder.close()
            
        except Exception as e:
            logger.error(f"Error executing command '{args.command}': {str(e)}")
            sys.exit(1)
    
    def _handle_load(self, args, builder: TableGraphBuilder):
        """Handle the 'load' command"""
        logger.info(f"Loading data from {args.data_dir}")
        
        # Initialize parser
        parser = OracleTableParser(data_dir=args.data_dir)
        
        # Parse files
        tables, columns, relationships, views = parser.parse_all_files(args.files)
        
        logger.info(f"Parsed {len(tables)} tables, {len(columns)} columns, {len(relationships)} relationships, and {len(views)} views")
        
        # Load tables into Neo4j
        table_success_count = 0
        for table_id, table in tables.items():
            if builder.create_table_node(table):
                table_success_count += 1
                
        logger.info(f"Loaded {table_success_count}/{len(tables)} tables into Neo4j")
        
        # Load columns into Neo4j
        column_success_count = 0
        for column_id, column in columns.items():
            if builder.create_column_node(column):
                column_success_count += 1
                
        logger.info(f"Loaded {column_success_count}/{len(columns)} columns into Neo4j")
        
        # Create column relationships (foreign keys)
        fk_success_count = 0
        for column_id, column in columns.items():
            if column.is_foreign_key and column.references_column:
                if builder.create_column_relationships(column):
                    fk_success_count += 1
        
        logger.info(f"Created {fk_success_count} column foreign key relationships")
        
        # Load table relationships into Neo4j
        rel_success_count = 0
        for rel in relationships:
            if builder.create_relationship(rel):
                rel_success_count += 1
                
        logger.info(f"Loaded {rel_success_count}/{len(relationships)} table relationships into Neo4j")
        
        # Load views into Neo4j
        view_success_count = 0
        for view_id, view in views.items():
            if builder.create_view_node(view):
                view_success_count += 1
                # Create relationships between view and tables
                if view.tables_used:
                    builder.create_view_relationships(view_id, view.tables_used)
                    
        logger.info(f"Loaded {view_success_count}/{len(views)} views into Neo4j")

    def _handle_query(self, args, builder: TableGraphBuilder):
        """Handle the 'query' command"""
        # Initialize RAG engine
        rag_engine = TableRAGEngine(builder)
        
        if args.node_type == 'table' or args.node_type == 'both':
            # Execute table query
            table_result = rag_engine.query(
                query_text=args.query_text,
                top_k=args.top_k,
                include_related=not args.no_related
            )
            
            # Output table results
            if args.format == 'json':
                if args.node_type == 'both':
                    print(json.dumps({"tables": table_result}, indent=2))
                else:
                    print(json.dumps(table_result, indent=2))
            else:
                if args.node_type == 'both':
                    print("\n=== TABLE RESULTS ===")
                self._print_query_results(table_result)
        
        if args.node_type == 'column' or args.node_type == 'both':
            # Execute column query
            column_results = builder.vector_search_columns(
                query_text=args.query_text,
                limit=args.top_k
            )
            
            # Output column results
            if args.format == 'json':
                if args.node_type == 'both':
                    print(json.dumps({"columns": column_results}, indent=2))
                else:
                    print(json.dumps(column_results, indent=2))
            else:
                if args.node_type == 'both':
                    print("\n=== COLUMN RESULTS ===")
                self._print_column_results(column_results)

        if args.node_type == 'view':
            # Execute view query
            view_results = builder.vector_search_views(
                query_text=args.query_text,
                limit=args.top_k
            )
            
            # Output view results
            if args.format == 'json':
                print(json.dumps(view_results, indent=2))
            else:
                self._print_view_results(view_results)
    
    def _handle_column(self, args, builder: TableGraphBuilder):
        """Handle the 'column' command and its subcommands"""
        if not hasattr(args, 'column_command') or not args.column_command:
            logger.error("No column subcommand specified")
            sys.exit(1)
            
        # In the _handle_column method of the OracleTablesCLI class
        if args.column_command == 'search':
            # Get the table ID if specified
            table_id = args.table_id if hasattr(args, 'table_id') else None
            
            if table_id:
                # Search for columns within a specific table
                columns = builder.get_columns_for_table(table_id)
                
                # Filter based on text search if columns are found
                if columns:
                    # If we have a query text, filter the columns
                    if args.query_text and args.query_text.strip():
                        # Simple text matching for now
                        filtered_columns = []
                        for col in columns:
                            if (args.query_text.lower() in col['name'].lower() or
                                (col['description'] and args.query_text.lower() in col['description'].lower())):
                                # Add a dummy similarity score
                                col['similarity'] = 1.0
                                filtered_columns.append(col)
                        columns = filtered_columns
                    
                    # Apply limit
                    columns = columns[:args.top_k]
            else:
                # Use existing vector search for all tables
                columns = builder.vector_search_columns(
                    query_text=args.query_text,
                    limit=args.top_k
                )
            
            # Output column results
            if args.format == 'json':
                print(json.dumps(columns, indent=2))
            else:
                print(f"\n=== Column Search Results ===")
                if table_id:
                    print(f"Searching in table: {table_id}")
                self._print_column_results(columns)
                
        elif args.column_command == 'list':
            # List columns for a table
            columns = builder.get_columns_for_table(args.table_id)
            
            # Output column list
            if args.format == 'json':
                print(json.dumps(columns, indent=2))
            else:
                print(f"\n=== Columns for table {args.table_id} ===")
                if not columns:
                    print("No columns found for this table.")
                else:
                    for i, column in enumerate(columns):
                        pk_marker = "🔑 " if column['is_primary_key'] else "  "
                        fk_marker = "🔗 " if column['is_foreign_key'] else "  "
                        print(f"{i+1}. {pk_marker}{fk_marker}{column['name']} ({column['datatype']})")
                        if column['description']:
                            print(f"   Description: {column['description']}")
                        if column['is_foreign_key'] and column['references_column']:
                            print(f"   References: {column['references_column']}")
                        print()
                
        elif args.column_command == 'details':
            # Get details for a column
            column_details = builder.get_column_details(args.column_id)
            
            # Output column details
            if args.format == 'json':
                print(json.dumps(column_details, indent=2))
            else:
                if column_details:
                    print(f"\n=== Column Details: {column_details['name']} ===")
                    print(f"ID: {column_details['id']}")
                    print(f"Data Type: {column_details['datatype']}")
                    print(f"Table: {column_details['table_id']}")
                    
                    if column_details['description']:
                        print(f"Description: {column_details['description']}")
                    
                    if column_details['length']:
                        print(f"Length: {column_details['length']}")
                    
                    if column_details['precision']:
                        print(f"Precision: {column_details['precision']}")
                    
                    print(f"Nullable: {'Yes' if column_details['is_nullable'] else 'No'}")
                    print(f"Primary Key: {'Yes' if column_details['is_primary_key'] else 'No'}")
                    print(f"Foreign Key: {'Yes' if column_details['is_foreign_key'] else 'No'}")
                    
                    if column_details['is_foreign_key'] and column_details['referenced_column_name']:
                        print(f"References: {column_details['referenced_table_id']}.{column_details['referenced_column_name']}")
                else:
                    print(f"Column with ID '{args.column_id}' not found.")
    
    def _handle_view(self, args, builder: TableGraphBuilder):
        """Handle the 'view' command and its subcommands"""
        if not hasattr(args, 'view_command') or not args.view_command:
            logger.error("No view subcommand specified")
            sys.exit(1)
            
        if args.view_command == 'search':
            # Search for views
            views = builder.vector_search_views(
                query_text=args.query_text,
                limit=args.top_k
            )
            
            # Output view results
            if args.format == 'json':
                print(json.dumps(views, indent=2))
            else:
                print(f"\n=== View Search Results ===")
                self._print_view_results(views)
                
        elif args.view_command == 'add':
            # Create view node
            view = ViewNode(
                id=args.id.lower(),
                name=args.name,
                module=args.module,
                submodule=args.submodule,
                description=args.description,
                sql_query=args.sql_query,
                tables_used=[t.lower() for t in args.tables_used]
            )
            
            # Add to Neo4j
            if builder.create_view_node(view):
                # Create relationships to tables
                if builder.create_view_relationships(view.id, view.tables_used):
                    print(f"Successfully added view '{view.name}' to the knowledge graph")
                else:
                    print(f"View '{view.name}' added but some table relationships failed")
            else:
                print(f"Failed to add view '{view.name}' to the knowledge graph")

    def _handle_info(self, args, builder: TableGraphBuilder):
        """Handle the 'info' command"""
        try:
            with builder.driver.session() as session:
                # Get table count
                result = session.run("MATCH (t:TABLE) RETURN COUNT(t) AS count")
                table_count = result.single()['count']
                
                # Get column count
                result = session.run("MATCH (c:COLUMN) RETURN COUNT(c) AS count")
                column_count = result.single()['count']
                
                # Get view count
                result = session.run("MATCH (v:VIEW) RETURN COUNT(v) AS count")
                view_count = result.single()['count']
                
                # Get relationship counts
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) AS type, COUNT(r) AS count
                    ORDER BY count DESC
                """)
                
                relationships = {}
                for record in result:
                    relationships[record['type']] = record['count']
                
                # Get modules
                result = session.run("""
                    MATCH (t:TABLE)
                    RETURN t.module AS module, COUNT(*) AS count
                    ORDER BY count DESC
                """)
                
                modules = {}
                for record in result:
                    modules[record['module']] = record['count']
                
                # Get primary key and foreign key counts
                result = session.run("""
                    MATCH (c:COLUMN)
                    WHERE c.is_primary_key = true
                    RETURN COUNT(c) AS pk_count
                """)
                pk_count = result.single()['pk_count']
                
                result = session.run("""
                    MATCH (c:COLUMN)
                    WHERE c.is_foreign_key = true
                    RETURN COUNT(c) AS fk_count
                """)
                fk_count = result.single()['fk_count']
                
                print("\n=== Knowledge Graph Statistics ===")
                print(f"Total Tables: {table_count}")
                print(f"Total Views: {view_count}")
                print(f"Total Columns: {column_count}")
                print(f"  - Primary Keys: {pk_count}")
                print(f"  - Foreign Keys: {fk_count}")
                
                print("\n=== Relationships ===")
                for rel_type, count in relationships.items():
                    print(f"{rel_type}: {count}")
                
                print("\n=== Tables by Module ===")
                for module, count in modules.items():
                    print(f"{module}: {count} tables")
                
        except Exception as e:
            logger.error(f"Error getting graph info: {str(e)}")

    def _handle_load_views(self, args, builder: TableGraphBuilder):
        """Handle the 'load-views' command"""
        logger.info(f"Loading views from {args.file}")
        
        try:
            with open(args.file, 'r') as f:
                views_data = json.load(f)
            
            view_success_count = 0
            relationship_success_count = 0
            
            for view_data in views_data:
                # Create ViewNode object from JSON data
                view = ViewNode(
                    id=view_data.get('id', '').lower(),
                    name=view_data.get('name', ''),
                    module=view_data.get('module', ''),
                    submodule=view_data.get('submodule', ''),
                    description=view_data.get('description', ''),
                    sql_query=view_data.get('sql_query', ''),
                    tables_used=[t.lower() for t in view_data.get('tables_used', [])]
                )
                
                # Create view node in Neo4j
                if builder.create_view_node(view):
                    view_success_count += 1
                    logger.info(f"Created view node: {view.name}")
                    
                    # Create relationships between view and tables
                    if view.tables_used:
                        if builder.create_view_relationships(view.id, view.tables_used):
                            relationship_success_count += 1
                            logger.info(f"Created relationships for view {view.name} to {len(view.tables_used)} tables")
                        else:
                            logger.warning(f"Failed to create some relationships for view {view.name}")
                else:
                    logger.error(f"Failed to create view node: {view.name}")
            
            logger.info(f"Successfully loaded {view_success_count}/{len(views_data)} views into Neo4j")
            logger.info(f"Created relationships for {relationship_success_count} views")
            
        except FileNotFoundError:
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {args.file}: {str(e)}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading views: {str(e)}")
            sys.exit(1)

    def _print_query_results(self, result: Dict[str, Any]):
        """Print table query results in text format"""
        print(f"\nQuery: {result['query']}")
        
        tables = result['tables']
        if not tables:
            print("\nNo matching tables found.")
            return
            
        print(f"\n=== Found {len(tables)} relevant tables ===")
        
        # Print top results
        for i, table in enumerate(tables):
            print(f"\n{i+1}. {table['name']} ({table['similarity']:.4f})")
            print(f"   Module: {table['module']}/{table['submodule']}")
            print(f"   Description: {table['description']}")
            
            # Print related tables if available
            if 'related_tables' in table and table['related_tables']:
                related = table['related_tables']
                print(f"\n   Related Tables ({len(related)}):")
                for j, rel in enumerate(related[:3]):  # Show top 3 related
                    print(f"   {j+1}. {rel['name']} - {rel['description']}")
                
                if len(related) > 3:
                    print(f"   ... and {len(related) - 3} more related tables")
            
            # Print columns if available
            if 'details' in table and 'columns' in table['details']:
                columns = table['details']['columns']
                if isinstance(columns, list) and columns:
                    print("\n   Key Columns:")
                    for j, col in enumerate(columns[:5]):  # Show top 5 columns
                        if isinstance(col, dict):
                            col_type = col.get('datatype', '')
                            print(f"   - {col.get('name', '')} ({col_type})")
                            
                    if len(columns) > 5:
                        print(f"   ... and {len(columns) - 5} more columns")
    
    def _print_column_results(self, columns: List[Dict[str, Any]]):
        """Print column search results in text format"""
        if not columns:
            print("\nNo matching columns found.")
            return
            
        print(f"\n=== Found {len(columns)} relevant columns ===")
        
        # Print top results
        for i, column in enumerate(columns):
            print(f"\n{i+1}. {column['name']} ({column['similarity']:.4f})")
            print(f"   Table: {column['table_id']}")
            print(f"   Data Type: {column['datatype']}")
            if column['description']:
                print(f"   Description: {column['description']}")

    def _print_view_results(self, views: List[Dict[str, Any]]):
        """Print view search results in text format"""
        if not views:
            print("\nNo matching views found.")
            return
            
        print(f"\n=== Found {len(views)} relevant views ===")
        
        # Print top results
        for i, view in enumerate(views):
            print(f"\n{i+1}. {view['name']} ({view['similarity']:.4f})")
            print(f"   Module: {view['module']}/{view['submodule']}")
            if view['description']:
                print(f"   Description: {view['description']}")
            if 'sql_query' in view and view['sql_query']:
                # Show first 200 chars of SQL
                sql_preview = view['sql_query'][:200] + "..." if len(view['sql_query']) > 200 else view['sql_query']
                print(f"   SQL Preview: {sql_preview}")

def main():
    """Main entry point"""
    cli = OracleTablesCLI()
    cli.run()

if __name__ == "__main__":
    main()