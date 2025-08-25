from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

VLLM_CONFIG_BIG = {
'model_id': os.environ.get('BIG_LLM_MODEL_ID', None), # Default model
'auth_token': os.environ.get('BIG_LLM_AUTH_TOKEN', None), # This should be set in environment
'api_base': os.environ.get('BIG_LLM_API_BASE', None), # Base URL for API
}

VLLM_CONFIG_REASONING = {
'model_id': os.environ.get('REASONING_LLM_MODEL_ID', None), # Default model
}

# Add check for model configuration
if not VLLM_CONFIG_BIG.get('model_id'):
    raise ValueError("VLLM_CONFIG_BIG['model_id'] is not set")

llm = ChatOpenAI(
    model=VLLM_CONFIG_BIG['model_id'],
    openai_api_key=VLLM_CONFIG_BIG['auth_token'],
    openai_api_base=VLLM_CONFIG_BIG['api_base'],
    max_tokens=8000,
    temperature=0,
    streaming=False
)

reasoning_llm = ChatOpenAI(
    model=VLLM_CONFIG_REASONING['model_id'],
    openai_api_key=VLLM_CONFIG_BIG['auth_token'],
    openai_api_base=VLLM_CONFIG_BIG['api_base'],
    max_tokens=8000,
    temperature=0,
    streaming=False
)