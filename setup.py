#!/usr/bin/env python3
import subprocess
import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all prerequisites are installed"""
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8+ is required")
        return False
    
    # Check if pip is installed
    try:
        subprocess.run(["pip", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("pip is not installed or not in PATH")
        return False
    
    # Check if Neo4j is accessible
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    try:
        import neo4j
        driver = neo4j.GraphDatabase.driver(
            neo4j_uri,
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
        )
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        logger.info(f"Neo4j is accessible at {neo4j_uri}")
    except Exception as e:
        logger.warning(f"Neo4j connection test failed: {str(e)}")
        logger.warning("You'll need to start Neo4j before using the application")
    
    # Check if Ollama is accessible
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import requests
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            model_found = any(m.get("name") == embed_model for m in models)
            if model_found:
                logger.info(f"Ollama is accessible and {embed_model} model is available")
            else:
                logger.warning(f"Ollama is accessible but {embed_model} model is not found")
                logger.warning(f"Run 'ollama pull {embed_model}' to download the model")
        else:
            logger.warning(f"Ollama responded with status code {response.status_code}")
    except Exception as e:
        logger.warning(f"Ollama connection test failed: {str(e)}")
        logger.warning("You'll need to start Ollama before using the application")
    
    # Check Neo4j version for vector index support
    try:
        import neo4j
        driver = neo4j.GraphDatabase.driver(
            neo4j_uri,
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
        )
        with driver.session() as session:
            result = session.run("RETURN apoc.version()")
            neo4j_version = result.single()[0]
            if neo4j_version.startswith('5.') and float(neo4j_version.split('.')[1]) >= 11:
                logger.info(f"Neo4j version {neo4j_version} supports vector indexes")
            else:
                logger.warning(f"Neo4j version {neo4j_version} may not support vector indexes")
                logger.warning("Vector search functionality may be limited")
        driver.close()
    except Exception as e:
        logger.warning(f"Neo4j version check failed: {str(e)}")
    
    return True

def install_dependencies():
    """Install required Python dependencies"""
    logger.info("Installing required Python packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    logger.info("All dependencies installed successfully")

def setup_data_directory():
    """Create data directory if it doesn't exist"""
    tables_dir = "Tables"
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)
        logger.info(f"Created '{tables_dir}' directory for JSON files")
    else:
        logger.info(f"'{tables_dir}' directory already exists")

def print_usage_guide():
    """Print usage guide for the application"""
    logger.info("\n=== Oracle Tables Knowledge Graph RAG Usage Guide ===")
    logger.info("\nBasic Commands:")
    logger.info("1. Load data: python cli.py load --files Financials.json HCM.json")
    logger.info("2. Query tables: python cli.py query \"Find customer sales tables\"")
    logger.info("3. Get system info: python cli.py info")
    
    logger.info("\nColumn-Specific Commands:")
    logger.info("1. Search for columns: python cli.py column search \"customer identifier\"")
    logger.info("2. List columns for a table: python cli.py column list customer_accounts")
    logger.info("3. Get column details: python cli.py column details customer_accounts_customer_id")
    
    logger.info("\nCombined Search:")
    logger.info("- Search both tables and columns: python cli.py query \"customer data\" --node-type both")
    
    logger.info("\nFor more details, use: python cli.py --help")

def main():
    """Main setup function"""
    logger.info("Starting setup for Oracle Tables Knowledge Graph RAG...")
    
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Please address the issues above.")
        return 1
    
    try:
        install_dependencies()
        setup_data_directory()
        
        logger.info("Setup completed successfully!")
        print_usage_guide()
        
        return 0
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())