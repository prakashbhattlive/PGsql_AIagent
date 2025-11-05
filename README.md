# LLM Agent with Ollama, PGVector, and PostgreSQL

A ready-to-run example demonstrating how to integrate an Ollama LLM, PGVector vector store, and a PostgreSQL devices table into a ReAct-style agent using LangChain 1.0.3 and LangGraph.

## 🧠 Overview

This project showcases a modular LangChain agent architecture that:
- Uses Ollama for local LLM inference
- Stores embeddings in PGVector
- Queries structured device data from PostgreSQL
- Implements ReAct-style reasoning via LangGraph

## 📦 Tech Stack

| Component     | Version        | Description                              |
|--------------|----------------|------------------------------------------|
| LangChain     | 1.0.3          | Framework for building LLM-powered apps  |
| LangGraph     | Latest         | Graph-based orchestration for agents     |
| Ollama        | Latest         | Local LLM runtime                        |
| PostgreSQL    | 18.1.3         | Relational database for device metadata  |
| PGVector      | Compatible     | Vector store extension for PostgreSQL    |

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL with PGVector extension
- Ollama installed and running locally or you can use OpenAI or any other public inference


Database view:
<img width="1093" height="420" alt="image" src="https://github.com/user-attachments/assets/cff59520-9244-4b90-b13b-5c00d5a712af" />
<img width="1333" height="631" alt="image" src="https://github.com/user-attachments/assets/614f923e-fa36-4c7a-ba77-129c18a3e1ad" />



OutCome with details logs:
<img width="906" height="604" alt="image" src="https://github.com/user-attachments/assets/4005f07e-e3f1-4216-9707-cf5914ece239" />
<img width="970" height="603" alt="image" src="https://github.com/user-attachments/assets/facb9fe9-0034-412c-b9b5-a61f173138a4" />
<img width="936" height="541" alt="image" src="https://github.com/user-attachments/assets/798021e0-5b39-4490-a1b7-fbfc8ee6ce11" />
