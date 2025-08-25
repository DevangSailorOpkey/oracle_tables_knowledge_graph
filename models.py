from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
import json

class Index(BaseModel):
    """Model for table index definition"""
    name: str
    columns: List[str]
    tablespace: Optional[str] = None
    uniqueness: str = "Non Unique"
    
    @validator('columns', pre=True)
    def validate_columns(cls, v):
        """Convert index columns to list format"""
        if isinstance(v, str):
            return [col.strip() for col in v.split(',')]
        elif isinstance(v, list):
            return v
        else:
            raise ValueError(f"Invalid index columns format: {v}")

class Column(BaseModel):
    """Model for table column definition"""
    name: str
    datatype: str
    length: Optional[str] = None
    precision: Optional[str] = None
    not_null: Optional[str] = None
    comments: Optional[str] = None
    flexfield_mapping: Optional[str] = None

class ColumnNode(BaseModel):
    """Model for column node in knowledge graph"""
    id: str  # Column ID (format: tablename_columnname)
    name: str  # Column name
    type: Literal["COLUMN"] = "COLUMN"
    datatype: str  # Data type
    table_id: str  # ID of the parent table
    description: Optional[str] = None  # Column description/comments
    length: Optional[str] = None  # Column length
    precision: Optional[str] = None  # Column precision
    is_nullable: bool = True  # Whether column can be null
    is_primary_key: bool = False  # Whether column is part of primary key
    is_foreign_key: bool = False  # Whether column is a foreign key
    references_column: Optional[str] = None  # ID of referenced column if foreign key
    embedding: Optional[List[float]] = None  # Vector embedding
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Convert to Neo4j-compatible dictionary"""
        return super().dict(*args, **kwargs)

class ForeignKey(BaseModel):
    """Model for foreign key relationship"""
    table: str  # Source table
    foreign_table: str  # Target table
    foreign_key_column: str  # Column name

class PrimaryKey(BaseModel):
    """Model for primary key definition"""
    name: str
    columns: str  # Can be comma-separated string
    
    @validator('columns', pre=True)
    def validate_columns(cls, v):
        """Ensure columns is a string"""
        if isinstance(v, list):
            return ", ".join(v)
        return v

class TableDetails(BaseModel):
    """Model for table details"""
    schema: Optional[str] = "FUSION"
    object_owner: Optional[str] = None
    object_type: Optional[str] = "TABLE"
    tablespace: Optional[str] = None

class TableNode(BaseModel):
    """Model for table node in knowledge graph"""
    id: str  # Table ID (usually table name in lowercase)
    name: str  # Table name
    type: Literal["TABLE"] = "TABLE"
    module: str  # Module name
    submodule: str  # Submodule name
    description: Optional[str] = None  # Table description
    details: Optional[TableDetails] = None  # Additional details
    primary_key: Optional[PrimaryKey] = None  # Primary key information
    columns: Optional[List[Column]] = None  # Column definitions
    indexes: Optional[List[Index]] = None  # Index information
    embedding: Optional[List[float]] = None  # Vector embedding
    tablespace: Optional[str] = None  # ADD THIS LINE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Convert to Neo4j-compatible dictionary"""
        d = super().dict(*args, **kwargs)
        
        # Convert complex objects to JSON strings
        if d.get('columns'):
            d['columns'] = json.dumps([col.dict() for col in self.columns])
        if d.get('indexes'):
            d['indexes'] = json.dumps([idx.dict() for idx in self.indexes])
        if d.get('primary_key'):
            d['primary_key'] = json.dumps(self.primary_key.dict())
        if d.get('details'):
            d['details'] = json.dumps(self.details.dict())
            
        return d

class Relationship(BaseModel):
    """Model for relationships between tables"""
    source_id: str  # Source table ID
    target_id: str  # Target table ID
    type: Literal["REFERENCES", "USES_TABLE"] = "REFERENCES"
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ViewNode(BaseModel):
    """Model for view node in knowledge graph"""
    id: str  # View ID (usually view name in lowercase)
    name: str  # View name
    type: Literal["VIEW"] = "VIEW"
    module: str  # Module name
    submodule: str  # Submodule name
    description: Optional[str] = None  # View description
    sql_query: str  # SQL query for the view
    tables_used: List[str]  # List of table names used in the view
    embedding: Optional[List[float]] = None  # Vector embedding
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Convert to Neo4j-compatible dictionary"""
        d = super().dict(*args, **kwargs)
        
        # Convert list to JSON string for Neo4j
        if d.get('tables_used'):
            d['tables_used'] = json.dumps(self.tables_used)
            
        return d