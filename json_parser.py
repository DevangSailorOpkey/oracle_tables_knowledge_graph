import json
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import re
from models import TableNode, TableDetails, PrimaryKey, Column, Index, ForeignKey, Relationship, ColumnNode, ViewNode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OracleTableParser:
    """Parser for Oracle Fusion table JSON data"""
    
    def __init__(self, data_dir: str = "Tables/"):
        """Initialize parser with directory containing JSON files"""
        self.data_dir = data_dir
        self.tables = {}  # Dictionary to store parsed tables by ID
        self.columns = {}  # Dictionary to store parsed columns by ID
        self.views = {}  # Dictionary to store parsed views by ID
        self.relationships = []  # List to store table relationships
        
    def parse_all_files(self, file_list: List[str]) -> Tuple[Dict[str, TableNode], Dict[str, ColumnNode], List[Relationship], Dict[str, ViewNode]]:
        """Parse all JSON files in the list and extract table data, columns, relationships and views"""
        for filename in file_list:
            file_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                module_name = filename.split('.')[0]  # Extract module name from filename
                self._parse_file(file_path, module_name)
            except Exception as e:
                logger.error(f"Error parsing file {filename}: {str(e)}")
                
        # Process relationships after all tables are loaded
        self._process_relationships()
        
        return self.tables, self.columns, self.relationships, self.views
    
    def _parse_file(self, file_path: str, module_name: str) -> None:
        """Parse a single JSON file and extract table data"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Fix potential JSON issues
            content = self._fix_json(content)
            
            # Parse JSON
            data = json.loads(content)
            
            # Check if data is a list (typical format)
            if isinstance(data, list):
                self._process_list_format(data, module_name)
            else:
                logger.warning(f"Unexpected data format in {file_path}, skipping")
                
        except Exception as e:
            logger.error(f"Error in _parse_file for {file_path}: {str(e)}")
            raise
    
    def _fix_json(self, content: str) -> str:
        """Fix common JSON issues"""
        # Complete unfinished arrays
        if content.startswith('[') and not content.endswith(']'):
            content += ']'
            
        # Remove trailing commas
        content = re.sub(r',\s*(\}|\])(?=\s*(\}|\]|$))', r'\1', content)
        
        return content
    
    def _process_list_format(self, data: List[Dict[str, Any]], module_name: str) -> None:
        """Process JSON data in list format"""
        for item in data:
            # Skip items without tableview_title
            if 'tableview_title' not in item:
                continue
                
            # Clean up submodule name
            submodule_name = re.sub(r'^\d+\s+', '', item['tableview_title'])
            
            # Process table data if available
            if 'table_data' in item and isinstance(item['table_data'], list):
                for table_data in item['table_data']:
                    try:
                        self._extract_table(table_data, module_name, submodule_name)
                    except Exception as e:
                        table_name = table_data.get('table_title', 'unknown')
                        logger.error(f"Error extracting table {table_name}: {str(e)}")
    
    def _extract_table(self, table_data: Dict[str, Any], module_name: str, submodule_name: str) -> None:
        """Extract table information from table_data object"""
        if 'table_title' not in table_data or 'data' not in table_data:
            return
            
        table_name = table_data['table_title']
        data = table_data['data']
        
        # Generate table ID (lowercase name)
        table_id = table_name.lower()
        
        # Skip if this table was already processed
        if table_id in self.tables:
            return
            
        # Extract table description
        description = data.get('short_description', '')
        if description:
            # Clean up description: remove newlines and excess whitespace
            description = re.sub(r'\s+', ' ', description).strip()
        
        # Extract table details
        details = TableDetails(
            schema=data.get('details', {}).get('schema', 'FUSION'),
            object_owner=data.get('details', {}).get('object_owner'),
            object_type=data.get('details', {}).get('object_type', 'TABLE'),
            tablespace=data.get('details', {}).get('tablespace')
        )
        
        # Extract primary key
        primary_key = None
        primary_key_columns = []
        if 'primary_key' in data and isinstance(data['primary_key'], dict):
            pk_data = data['primary_key']
            primary_key = PrimaryKey(
                name=pk_data.get('name', ''),
                columns=pk_data.get('columns', '')
            )
            
            # Extract primary key column names
            if isinstance(pk_data.get('columns'), str):
                primary_key_columns = [col.strip() for col in pk_data['columns'].split(',')]
            elif isinstance(pk_data.get('columns'), list):
                primary_key_columns = pk_data['columns']
        
        # Extract columns
        columns = []
        if 'columns' in data and isinstance(data['columns'], list):
            for col_data in data['columns']:
                try:
                    column = Column(
                        name=col_data.get('name', ''),
                        datatype=col_data.get('datatype', ''),
                        length=col_data.get('length'),
                        precision=col_data.get('precision'),
                        not_null=col_data.get('not_null'),
                        comments=col_data.get('comments'),
                        flexfield_mapping=col_data.get('flexfield_mapping')
                    )
                    columns.append(column)
                    
                    # Create column node for the knowledge graph
                    column_name = col_data.get('name', '')
                    if column_name:
                        column_id = f"{table_id}_{column_name.lower()}"
                        
                        # Check if column is primary key
                        is_primary_key = column_name in primary_key_columns
                        
                        # Create column node
                        column_node = ColumnNode(
                            id=column_id,
                            name=column_name,
                            datatype=col_data.get('datatype', ''),
                            table_id=table_id,
                            description=col_data.get('comments', ''),
                            length=col_data.get('length'),
                            precision=col_data.get('precision'),
                            is_nullable=not col_data.get('not_null', False),
                            is_primary_key=is_primary_key,
                            is_foreign_key=False  # Will be updated later when processing foreign keys
                        )
                        
                        self.columns[column_id] = column_node
                    
                except Exception as e:
                    logger.warning(f"Error processing column in table {table_name}: {str(e)}")
        
        # Extract indexes
        indexes = []
        if 'indexes' in data and isinstance(data['indexes'], list):
            for idx_data in data['indexes']:
                try:
                    index = Index(
                        name=idx_data.get('index', ''),
                        columns=idx_data.get('columns', ''),
                        tablespace=idx_data.get('tablespace'),
                        uniqueness=idx_data.get('uniqueness', 'Non Unique')
                    )
                    indexes.append(index)
                except Exception as e:
                    logger.warning(f"Error processing index in table {table_name}: {str(e)}")
        
        # Create table node
        table_node = TableNode(
            id=table_id,
            name=table_name,
            module=module_name,
            submodule=submodule_name,
            description=description,
            details=details,
            primary_key=primary_key,
            columns=columns,
            indexes=indexes
        )
        
        # Store the table
        self.tables[table_id] = table_node
        
        # Store foreign keys for later relationship processing
        if 'foreign_keys' in data and isinstance(data['foreign_keys'], list):
            for fk_data in data['foreign_keys']:
                try:
                    foreign_key = ForeignKey(
                        table=fk_data.get('table', '').lower(),
                        foreign_table=fk_data.get('foreign_table', '').lower(),
                        foreign_key_column=fk_data.get('foreign_key_column', '')
                    )
                    
                    # Update column node to mark as foreign key
                    fk_column_name = fk_data.get('foreign_key_column', '')
                    if fk_column_name:
                        fk_column_id = f"{table_id}_{fk_column_name.lower()}"
                        if fk_column_id in self.columns:
                            self.columns[fk_column_id].is_foreign_key = True
                            
                            # Set reference to target column
                            target_table = fk_data.get('foreign_table', '').lower()
                            target_column = fk_data.get('foreign_key_column', '')
                            if target_table and target_column:
                                target_column_id = f"{target_table}_{target_column.lower()}"
                                self.columns[fk_column_id].references_column = target_column_id
                    
                    # Store for later processing
                    self._add_foreign_key(table_id, foreign_key)
                except Exception as e:
                    logger.warning(f"Error processing foreign key in table {table_name}: {str(e)}")
    
    def _add_foreign_key(self, table_id: str, foreign_key: ForeignKey) -> None:
        """Store foreign key for later relationship processing"""
        # We only process the relationship if both tables will be present in our graph
        source_id = foreign_key.table
        target_id = foreign_key.foreign_table
        
        # Create a temporary relationship
        relationship = {
            'source_id': source_id,
            'target_id': target_id,
            'foreign_key_column': foreign_key.foreign_key_column
        }
        
        # Save for later processing
        self._temp_relationships.append(relationship)
    
    def parse_view(self, view_data: Dict[str, Any], module_name: str, submodule_name: str) -> Optional[ViewNode]:
        """Parse a single view definition
        
        Args:
            view_data: Dictionary containing view information
            module_name: Module name
            submodule_name: Submodule name
            
        Returns:
            ViewNode if successfully parsed, None otherwise
        """
        try:
            # Extract required fields
            view_id = view_data.get('id', '').lower()
            view_name = view_data.get('name', '')
            description = view_data.get('description', '')
            sql_query = view_data.get('sql_query', '')
            tables_used = view_data.get('tables_used', [])
            
            # Validate required fields
            if not all([view_id, view_name, sql_query, tables_used]):
                logger.warning(f"Missing required fields for view: {view_data}")
                return None
            
            # Create view node
            view_node = ViewNode(
                id=view_id,
                name=view_name,
                module=module_name,
                submodule=submodule_name,
                description=description,
                sql_query=sql_query,
                tables_used=[t.lower() for t in tables_used]  # Ensure lowercase for consistency
            )
            
            return view_node
            
        except Exception as e:
            logger.error(f"Error parsing view {view_data.get('name', 'unknown')}: {str(e)}")
            return None

    def add_view(self, view_data: Dict[str, Any], module_name: str = "Unknown", submodule_name: str = "Unknown") -> bool:
        """Add a view to the parser
        
        Args:
            view_data: Dictionary containing view information
            module_name: Module name
            submodule_name: Submodule name
            
        Returns:
            True if successfully added, False otherwise
        """
        view = self.parse_view(view_data, module_name, submodule_name)
        if view:
            self.views[view.id] = view
            return True
        return False

    def _process_relationships(self) -> None:
        """Process foreign keys to create relationships between tables"""
        # Set to track processed relationships to avoid duplicates
        processed = set()
        
        for rel in self._temp_relationships:
            source_id = rel['source_id']
            target_id = rel['target_id']
            
            # Skip if source or target table doesn't exist in our graph
            if source_id not in self.tables or target_id not in self.tables:
                continue
                
            # Create a unique key for this relationship
            rel_key = f"{source_id}|{target_id}|{rel['foreign_key_column']}"
            
            # Skip if already processed
            if rel_key in processed:
                continue
                
            # Create the relationship
            relationship = Relationship(
                source_id=source_id,
                target_id=target_id,
                type="REFERENCES",
                properties={
                    'foreign_key_column': rel['foreign_key_column']
                }
            )
            
            self.relationships.append(relationship)
            processed.add(rel_key)
    
    @property
    def _temp_relationships(self) -> List[Dict[str, str]]:
        """Get or create temporary relationships list"""
        if not hasattr(self, '_temp_rels'):
            self._temp_rels = []
        return self._temp_rels

# Example usage
if __name__ == "__main__":
    parser = OracleTableParser()
    tables, columns, relationships = parser.parse_all_files(
        ["Financials.json", "HCM.json", "SCM.json", "Project Management.json", "Sales and Fusion Service.json"]
    )
    print(f"Parsed {len(tables)} tables, {len(columns)} columns, and {len(relationships)} relationships")