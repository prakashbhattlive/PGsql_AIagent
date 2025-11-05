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
- **DataSet** https://www.kaggle.com/datasets/paperxd/all-computer-prices/data


Database view:
<img width="1093" height="420" alt="image" src="https://github.com/user-attachments/assets/cff59520-9244-4b90-b13b-5c00d5a712af" />
<img width="1333" height="631" alt="image" src="https://github.com/user-attachments/assets/614f923e-fa36-4c7a-ba77-129c18a3e1ad" />



## OutCome with detailed logs:

### Case1:
- EMBED_MODEL: **mxbai-embed-large:latest**
- LLM_MODEL: **mistral:latest**

  <img width="947" height="500" alt="image" src="https://github.com/user-attachments/assets/c4af8b5b-f783-49f1-88a9-5bc7618d6527" />
  <img width="945" height="503" alt="image" src="https://github.com/user-attachments/assets/b00891ee-8bce-40bd-8636-48ae6aa3f882" />
  <img width="956" height="545" alt="image" src="https://github.com/user-attachments/assets/45d421d3-0363-40c3-b74c-172e505fcaa6" />


### Case2:
- EMBED_MODEL: **mxbai-embed-large:latest**
- LLM_MODEL: **gpt-oss:20b**

  <img width="940" height="599" alt="image" src="https://github.com/user-attachments/assets/da7ad8a4-dba7-410c-9ffa-0a3b59542f01" />
  <img width="939" height="587" alt="image" src="https://github.com/user-attachments/assets/60763725-2f5e-453d-b2d1-7ae279400bf7" />
  <img width="938" height="636" alt="image" src="https://github.com/user-attachments/assets/e7f7c526-6fbc-466e-867c-88bf24b6ea91" />
