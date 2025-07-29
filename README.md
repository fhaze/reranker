
# BGE Reranker Service

## Overview
The BGE Reranker Service is an API-compatible reranker using the BAAI/bge-reranker-v2-m3 model. It provides a FastAPI service that accepts a query and a list of documents, then returns the documents ranked by their relevance to the query.

## Features
- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
- **Sentence Transformers**: Utilizes pre-trained transformer models for sentence embeddings and reranking.
- **Model Management**: Efficiently loads and unloads the model to free up memory when not in use.
- **Background Task**: Periodically checks and unloads the model if it hasn't been used for a specified time.

## Dependencies
- Python 3.12
- FastAPI==0.116.1
- Uvicorn==0.30.6
- Sentence Transformers==5.0.0
- Pydantic==2.9.2
- Numpy==2.1.1
- AP Scheduler==3.11.0
- Pytz==2024.2

## Setup

### Using Docker
1. **Build the Docker Image**
   ```bash
   docker build -t bge-reranker-service .
   ```

2. **Run the Docker Container**
   ```bash
   docker run -p 8000:8000 bge-reranker-service
   ```

### Without Docker
1. **Install Dependencies**
   Make sure you have Python 3.12 installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

### Example Request
To rerank documents, send a POST request to `/v1/rerank` with the following JSON body:
```json
{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "Your query here",
    "documents": ["Document 1", "Document 2", "Document 3"],
    "top_n": 2,
    "return_documents": true
}
```

### Example Response
```json
{
    "results": [
        {
            "index": 0,
            "relevance_score": 0.9876,
            "document": {
                "text": "Document 1"
            }
        },
        {
            "index": 2,
            "relevance_score": 0.9765,
            "document": {
                "text": "Document 3"
            }
        }
    ],
    "model": "BAAI/bge-reranker-v2-m3",
    "usage": {
        "prompt_tokens": 3,
        "completion_tokens": 0,
        "total_tokens": 3
    }
}
```

## License
This project is licensed under the MIT License.

