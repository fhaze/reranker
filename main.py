import gc
import logging
import os
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import List, Optional

import torch
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pytz import utc
from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger("apscheduler").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.lock = Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_used = 0

    def get_model(self):
        with self.lock:
            self.last_used = time.time()
            if not self.model:
                logging.info(f"Loading model: {self.model_name}")
                self.model = CrossEncoder(self.model_name, device=self.device)
            return self.model

    def unload_model(self, ttl_seconds=60 * 5):
        """Unload model to free memory"""
        with self.lock:
            if self.model and (time.time() - self.last_used > ttl_seconds):
                logging.info("Unloading model to free memory")
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()


# Initialize model manager
model_manager = ModelManager("BAAI/bge-reranker-v2-m3")


# Background task to periodically check model TTL
@asynccontextmanager
async def lifespan(_: FastAPI):
    scheduler = AsyncIOScheduler(timezone=utc)
    scheduler.add_job(func=model_manager.unload_model, trigger="interval", seconds=5)
    scheduler.start()
    yield


# Initialize FastAPI app
app = FastAPI(
    title="BGE Reranker Service",
    description="OpenAI API compatible reranker using BAAI/bge-reranker-v2-m3",
    version="1.0.0",
    lifespan=lifespan,
)


class RerankRequest(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True


class Document(BaseModel):
    text: str


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[Document] = None


class RerankResponse(BaseModel):
    results: List[RerankResult]
    model: str
    usage: dict


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    try:
        logger.info("Incoming /rerank request: %s", request.model_dump_json(indent=2))

        # Get model (loads if needed)
        model = model_manager.get_model()

        # Prepare pairs for reranking
        query_document_pairs = [[request.query, doc] for doc in request.documents]

        # Get relevance scores
        scores = model.predict(query_document_pairs, convert_to_tensor=True)
        scores = torch.sigmoid(scores).cpu().tolist()

        # Create results with indices and scores
        results = [
            RerankResult(
                index=i,
                relevance_score=float(score),
                document=Document(text=doc) if request.return_documents else None,
            )
            for i, (doc, score) in enumerate(zip(request.documents, scores))
        ]

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Apply top_n if specified
        if request.top_n:
            results = results[: request.top_n]

        response = RerankResponse(
            results=results,
            model=request.model,
            usage={
                "prompt_tokens": len(request.query.split()),
                "completion_tokens": 0,
                "total_tokens": len(request.query.split()),
            },
        )

        logger.info("Response for /rerank: %s", response.model_dump_json(indent=2))
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
