"""
A ready-to-run example that shows how to wire an Ollama LLM,
PGVector vector store, and a PostgreSQL devices table
into a ReAct-style agent using LangChain 1.0.3 + LangGraph.

Author:  Prakash Bhatt
Date:    2024-06-10
Updated: 2025 (LangChain 1.0.3 compatibility)
"""

import os
import logging
from dotenv import load_dotenv

# ---------- VERSION CHECK ----------
import langchain
print(f"LangChain version: {langchain.__version__}")

# ---------- BASIC IMPORTS ----------
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

# Use langchain-ollama for proper tool support
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    print("✓ Using langchain-ollama (recommended)")
except ImportError:
    print("⚠️  langchain-ollama not found. Installing it now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-ollama"])
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    print("✓ langchain-ollama installed successfully")

from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.utilities import SQLDatabase

# ---------- LANGGRAPH AGENT IMPORTS ----------
try:
    from langgraph.prebuilt import create_react_agent
    print("✓ Using LangGraph for agent creation")
    USE_LANGGRAPH = True
except ImportError:
    print("⚠️  LangGraph not installed. Please run: pip install langgraph")
    print("   Falling back to manual tool execution...")
    USE_LANGGRAPH = False

# ---------- LOGGING ----------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------- ENV ----------
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
    raise RuntimeError(
        "Missing required DB credentials. "
        "Set DB_HOST, DB_USER, DB_PASSWORD, DB_NAME in a .env file."
    )

CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

logger.info(f"DB URL: {CONNECTION_STRING}")
logger.info(f"Ollama URL: {OLLAMA_URL}")

# ---------- EMBEDDINGS & LLM ----------
# Using langchain-ollama for proper tool binding support
embedding_model = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
)

# ChatOllama from langchain-ollama supports tool binding
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
)

# ---------- PGVector ----------
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name="comprice_docs",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------- SQL ----------
db = SQLDatabase.from_uri(CONNECTION_STRING, include_tables=["devices"])


def query_devices_db(sql: str) -> str:
    """
    Run a raw SQL query against the devices table.
    
    Args:
        sql: A SQL SELECT query string
        
    Returns:
        Formatted results or error message
    """
    logger.info(f"Executing SQL: {sql}")
    try:
        result = db.run(sql)
        
        if not result or result.strip() == "":
            return "(no rows found)"
        
        return result
    except Exception as exc:
        logger.error(f"SQL error: {exc}")
        return f"⚠️ Error executing query: {exc}"


def retrieve_knowledge(question: str) -> str:
    """
    Retrieve contextual knowledge from the vector store.
    
    Args:
        question: A natural language question or search query
        
    Returns:
        Concatenated relevant document content
    """
    logger.info(f"Retrieving knowledge for: {question}")
    try:
        docs = retriever.invoke(question)
        
        if not docs:
            return "No relevant knowledge found in the knowledge base."
        
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as exc:
        logger.error(f"Retrieval error: {exc}")
        return f"⚠️ Error retrieving knowledge: {exc}"


# ---------- TOOLS ----------
tools = [
    Tool(
        name="KnowledgeBaseRetriever",
        func=retrieve_knowledge,
        description=(
            "Use this tool to fetch contextual information about devices, "
            "technical terms, or concepts from the knowledge base. "
            "Input should be a clear natural language question. "
            "This tool is useful for understanding terminology, specifications, "
            "or getting background information about device features."
        ),
    ),
    Tool(
        name="DevicesSQLQuery",
        func=query_devices_db,
        description=(
            "Use this tool to query the PostgreSQL 'devices' table using SQL. "
            "Input MUST be a valid SQL SELECT statement. "
            "Available columns: device_type, brand, model, release_year, os, "
            "form_factor, cpu_brand, cpu_model, cpu_tier, cpu_cores, cpu_threads, "
            "cpu_base_ghz, cpu_boost_ghz, gpu_brand, gpu_model, gpu_tier, vram_gb, "
            "ram_gb, storage_type, storage_gb, storage_drive_count, display_type, "
            "display_size_in, resolution, refresh_hz, battery_wh, charger_watts, "
            "psu_watts, wifi, bluetooth, weight_kg, warranty_months, price. "
            "Example: SELECT brand, model, price FROM devices WHERE brand='Samsung' AND release_year > 2021 LIMIT 10;"
        ),
    ),
]

# ---------- CREATE AGENT ----------
if USE_LANGGRAPH:
    logger.info("Creating agent using LangGraph...")
    
    agent_executor = create_react_agent(
        llm,
        tools,
    )
    
    logger.info("✓ Agent created successfully\n")
else:
    # Fallback: Simple tool execution without agent
    logger.warning("Running in fallback mode without agent reasoning")
    agent_executor = None


# ---------- HELPER FUNCTION ----------
def run_query(query: str, query_name: str) -> None:
    """Execute a query with proper error handling."""
    logger.info("=" * 80)
    logger.info(f"{query_name}: {query}")
    logger.info("=" * 80)
    
    if agent_executor:
        try:
            # LangGraph returns a generator/stream, we need to consume it
            result = agent_executor.invoke({"messages": [("user", query)]})
            
            # Extract the final response from messages
            if "messages" in result:
                final_message = result["messages"][-1]
                output = final_message.content if hasattr(final_message, 'content') else str(final_message)
            else:
                output = str(result)
                
            logger.info(f"\n{query_name} Result:\n{output}\n")
            
        except Exception as e:
            logger.error(f"❌ Error in {query_name}: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Fallback: Direct tool calling
        logger.info("Using direct tool execution (no agent reasoning)...")
        
        # Try SQL query first for data queries
        if "list" in query.lower() or "find" in query.lower() or "show" in query.lower():
            logger.info("Attempting SQL query...")
            # This is a simplified fallback - not as smart as an agent
            sql = "SELECT * FROM devices LIMIT 10;"
            result = query_devices_db(sql)
            logger.info(f"\n{query_name} Result:\n{result}\n")
        
        # Try knowledge base for explanatory queries
        elif "explain" in query.lower() or "what" in query.lower():
            logger.info("Attempting knowledge retrieval...")
            result = retrieve_knowledge(query)
            logger.info(f"\n{query_name} Result:\n{result}\n")


# ---------- MAIN ----------
if __name__ == "__main__":
    
    if not USE_LANGGRAPH:
        logger.warning("\n" + "!" * 80)
        logger.warning("IMPORTANT: Install langgraph for full agent capabilities:")
        logger.warning("pip install langgraph")
        logger.warning("!" * 80 + "\n")
    
    # 1️⃣ Combined knowledge-base + SQL query
    run_query(
        "List all Samsung desktops released after 2021 with more than 8 CPU cores "
        "and a price under 1500 USD.",
        "🔍 Query 1"
    )
    
    # 2️⃣ Purely explanatory query
    run_query(
        "Explain what CPU tier means in this dataset and how it affects performance.",
        "📘 Query 2"
    )
    
    # 3️⃣ Complex multi-step query
    run_query(
        "What are the top 3 most affordable laptops with at least 16GB RAM "
        "and explain what makes a good CPU for laptops?",
        "💡 Query 3"
    )
    
    logger.info("=" * 80)
    logger.info("✓ All queries completed")
    logger.info("=" * 80)