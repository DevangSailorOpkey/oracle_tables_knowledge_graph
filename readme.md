# Oracle Tables Knowledge Graph RAG

A semantic search and exploration system for Oracle Fusion tables using a knowledge graph approach with vector embeddings for natural language queries.

## Overview

This project creates a knowledge graph of Oracle tables with vector embeddings to enable natural language queries about tables, columns, and their relationships. It helps users explore and understand complex database schemas without needing to know exact table or column names.

## Features

- **Natural Language Search**: Find tables and columns using semantic similarity rather than exact name matching
- **Relationship Discovery**: Explore connections between tables through foreign key relationships
- **Column-Level Querying**: Search for specific data elements across all tables
- **SQL Generation**: Get suggested SQL queries based on your search intent
- **Hybrid Search**: Query both tables and columns simultaneously
- **Console Interface**: Easy-to-use CLI for all operations
- **Column Metadata Updates**: Update column descriptions and automatically regenerate embeddings
- **View Management**: Load and search database views with their table relationships

## Architecture

The system is built around these core components:

- **Knowledge Graph**: Neo4j database storing table, column, and view nodes with relationships
- **Vector Embeddings**: Table and column metadata embedded using Ollama's nomic-embed-text model
- **Graph Builder**: Creates and maintains the knowledge graph structure
- **RAG Engine**: Handles queries and retrieval from the knowledge graph
- **CLI Interface**: Command-line tools for interacting with the system

## Prerequisites

- Python 3.8+
- Neo4j 5.11+ (for vector search capabilities)
- Ollama with the nomic-embed-text model

## Installation

1. Clone this repository
2. Run the setup script to install dependencies and create necessary directories:

```bash
python setup.py
```

3. Ensure Neo4j is running and accessible
4. Ensure Ollama is running with the nomic-embed-text model available

## Usage

### Loading Data

Place your Oracle JSON files in the `Tables/` directory, then load them into the knowledge graph:

```bash
python cli.py load --files Financials.json HCM.json SCM.json
```

### Querying Tables

Search for tables using natural language:

```bash
python cli.py query "Find customer sales information"
```

Get information about the system:

```bash
python cli.py info
```

### Working with Columns

Search for specific columns across all tables:

```bash
python cli.py column search "customer identifier"
```

Search for specific columns inside a specific table:
```bash
python cli.py column search "identifier" --table-id customer_accounts
```

List all columns for a specific table:

```bash
python cli.py column list customer_accounts
```

Get detailed information about a specific column:

```bash
python cli.py column details customer_accounts_customer_id
```

### Combined Search

Search for both tables and columns matching your query:

```bash
python cli.py query "customer data" --node-type both
```

### Update a column's description

Update a column's description and automatically regenerate its embedding:

```bash
python cli.py column update payment_amounts_total_amount --description "Total payment amount including taxes and fees in transaction currency"
```

### Verify the update

```bash
python cli.py column details payment_amounts_total_amount
```

## Example Workflows

### Finding Tables Related to Customer Orders

```bash
# Find tables related to customer orders
python cli.py query "customer order processing"

# Look at columns for the top result
python cli.py column list order_headers_all

# Find columns related to order status across all tables
python cli.py column search "order status"
```

### Exploring Foreign Key Relationships

```bash
# Find a specific table
python cli.py query "payment methods"

# Get system info including relationship counts
python cli.py info

# View details about a specific column that might be a foreign key
python cli.py column details payment_methods_payment_type_id
```

## JSON File Format

The system expects Oracle table JSON files in the following format:

```json
[
  {
    "tableview_title": "Module Name",
    "table_data": [
      {
        "table_title": "TABLE_NAME",
        "data": {
          "short_description": "Description of the table",
          "columns": [
            {
              "name": "COLUMN_NAME",
              "datatype": "VARCHAR2",
              "length": "50",
              "not_null": "Y",
              "comments": "Description of the column"
            }
          ],
          "primary_key": {
            "name": "PK_NAME",
            "columns": "COLUMN_NAME"
          },
          "foreign_keys": [
            {
              "table": "TABLE_NAME",
              "foreign_table": "REFERENCED_TABLE",
              "foreign_key_column": "COLUMN_NAME"
            }
          ],
          "indexes": [
            {
              "index": "INDEX_NAME",
              "columns": "COLUMN_NAME",
              "uniqueness": "UNIQUE"
            }
          ],
          "details": {
            "schema": "FUSION",
            "object_owner": "OWNER",
            "object_type": "TABLE",
            "tablespace": "USERS"
          }
        }
      }
    ]
  }
]
```

## Environment Variables

- `NEO4J_URI`: Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password (default: `password`)
- `OLLAMA_BASE_URL`: Ollama API URL (default: `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL`: Embedding model to use (default: `nomic-embed-text`)

## Project Structure

```
├── cli.py                 # Command-line interface
├── embedder.py            # Vector embedding generation
├── graph_builder.py       # Neo4j knowledge graph operations
├── json_parser.py         # Oracle JSON file parser
├── models.py              # Data models for tables and columns
├── rag_engine.py          # Retrieval and query processing
├── requirements.txt       # Python dependencies
├── schema.yaml            # Neo4j schema definition
├── setup.py               # Setup and dependency installation
└── Tables/                # Directory for Oracle JSON files
```

## Advanced Configuration

### Neo4j Indexes

To improve performance, the system creates two vector indexes in Neo4j:
- `table_embedding`: For table nodes
- `column_embedding`: For column nodes

These require Neo4j 5.11+ with vector index capabilities.

### Embedding Model

The default embedding model is `nomic-embed-text`, but you can use any embedding model available in Ollama. To change the model:

```bash
export OLLAMA_EMBED_MODEL="your-preferred-model"
```

## Troubleshooting

### Connection Issues

If you encounter connection issues with Neo4j or Ollama:

1. Ensure both services are running:
   - Neo4j: `neo4j start`
   - Ollama: Typically runs as a background service

2. Check environment variables match your configuration

3. Run setup again to test connections:
   ```bash
   python setup.py
   ```

### Vector Search Limitations

If your Neo4j version doesn't support vector indexes, the system will fall back to manual similarity calculation, which is significantly slower.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Neo4j](https://neo4j.com/) for graph database capabilities
- [Ollama](https://ollama.ai/) for local embedding model support