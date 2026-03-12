# DocuQuery RAG

Production-ready **Retrieval-Augmented Generation (RAG) Chatbot** that allows users to ask questions over private documents using LLMs and semantic search.

This project demonstrates how to build an **enterprise-grade AI assistant** capable of retrieving contextual information from document knowledge bases and generating accurate responses.

## 🚀 Features

* Document ingestion pipeline
* Semantic search with vector embeddings
* Retrieval-Augmented Generation
* Conversational memory
* Streaming responses
* REST API with FastAPI
* Dockerized environment
* Scalable architecture
* Observability ready

## 🧠 Architecture

User Question
↓
API (FastAPI)
↓
Embedding Model
↓
Vector Database (Semantic Search)
↓
Top-K Document Retrieval
↓
LLM Prompt Construction
↓
LLM Response Generation

## 🏗️ Tech Stack

* Python
* FastAPI
* LangChain
* OpenAI / LLM
* Vector Database (Chroma / Pinecone / Weaviate)
* Docker
* PostgreSQL (optional for metadata)
* Redis (optional for caching)

## 📂 Project Structure

```
docuquery-rag/
│
├── app/
│   ├── api/
│   │   ├── routes.py
│   │   └── dependencies.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── vectorstore/
│   │   └── vectordb.py
│   │
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── prompt_builder.py
│   │   └── generator.py
│   │
│   ├── services/
│   │   └── chat_service.py
│   │
│   └── main.py
│
├── data/
│   └── documents/
│
├── scripts/
│   └── ingest_documents.py
│
├── tests/
│
├── docker/
│   └── Dockerfile
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 📥 Document Ingestion

Supports:

* PDF
* Markdown
* TXT
* Web pages

Documents are processed through:

1. Loader
2. Chunking
3. Embedding
4. Vector storage

## 💬 Example Query

```
POST /chat

{
  "question": "What is the company's refund policy?"
}
```

Response:

```
{
  "answer": "The company offers a 30-day refund period..."
}
```

## ⚙️ Setup

Clone the repository:

```
git clone https://github.com/yourusername/docuquery-rag
cd docuquery-rag
```

Install dependencies:

```
pip install -r requirements.txt
```

Run API:

```
uvicorn app.main:app --reload
```

API docs:

```
http://localhost:8000/docs
```

## 🐳 Docker

Run with Docker:

```
docker-compose up --build
```

## 📈 Future Improvements

* Hybrid search (BM25 + Vector)
* Streaming LLM responses
* User session memory
* Multi-tenant document spaces
* Authentication
* Feedback learning loop

## 🎯 Use Cases

* Internal knowledge assistants
* Customer support automation
* Legal document search
* Enterprise documentation search

## 📜 License

MIT
