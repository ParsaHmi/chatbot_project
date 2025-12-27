# Vector Search Comparison Project

This project demonstrates a **vector similarity search** pipeline implemented with **two vector databases**: **ChromaDB** and **Milvus**. It allows you to store document embeddings and query them using various **indexing algorithms** and **similarity metrics**, then compare performance and results.

---

## Features

- **Vector Databases:**
  - **ChromaDB**: Lightweight vector database with HNSW indexing.
  - **Milvus**: Production-grade vector database supporting multiple indexing algorithms.

- **Indexing Algorithms Supported in Milvus:**
  - HNSW (`HNSW`)
  - IVF_FLAT (`IVF_FLAT`)
  - IVF_PQ (`IVF_PQ`)

- **Similarity Metrics Supported in Milvus:**
  - Cosine similarity (`COSINE`)
  - Euclidean distance (`L2`)
  - Dot product / Inner product (`IP`)

- **Embedding Generation:**
  - Uses the **Ollama API** with the `"all-minilm:22m"` model (or another model of your choice).

- **Document Chunking:**
  - Splits large documents into overlapping sentence chunks for better embedding representation.

- **Performance Comparison:**
  - Measures query latency, memory usage, and optionally accuracy if ground truth is provided.
  - Supports automated comparison across multiple indexing algorithms and similarity metrics.

---

## Requirements

- Python >= 3.13
- Dependencies (can install via `requirements.txt`):
  ```bash
  pip install pymilvus chromadb transformers torch tqdm fitz spacy ollama


## How to Run
first you need to use following command for docker :
  ```bash
    docker-compose up -d

then you can run the app using python !