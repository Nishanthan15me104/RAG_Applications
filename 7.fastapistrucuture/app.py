# FINAL PRODUCTION FOLDER STRUCTURE
# project/
# │
# ├── app/
# │   ├── main.py                 # Entry point
# │
# │   ├── api/
# │   │   ├── routes.py           # All endpoints
# │
# │   ├── services/
# │   │   ├── rag_service.py      # Business logic
# │
# │   ├── core/
# │   │   ├── config.py           # ENV config
# │   │   ├── logging.py          # Logging setup
# │
# │   ├── models/
# │   │   ├── request.py          # Pydantic models
# │
# ├── src/
# │   ├── retriever.py            # Your existing logic
# │   ├── generator.py            # Your existing logic
# │
# ├── .env
# ├── requirements.txt
# 🚀 1. app/main.py (ENTRY POINT)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.logging import logger
from app.services.rag_service import init_services, cleanup_services
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix=settings.API_V1_PREFIX)

# Startup
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting API...")
    await init_services()
    logger.info("✅ Services initialized")

# Shutdown
@app.on_event("shutdown")
async def shutdown():
    logger.info("🛑 Shutting down API...")
    await cleanup_services()
    logger.info("Cleanup complete")

# 🚀 2. app/api/routes.py (API LAYER)
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.request import QueryRequest
from app.services.rag_service import service
from app.core.logging import logger

router = APIRouter()

@router.post("/ask")
async def ask_maritime(request: QueryRequest):
    try:
        logger.info(f"📨 Query: {request.prompt}")

        stream = await service.handle_query(request.prompt)

        return StreamingResponse(stream, media_type="text/plain")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Error")


@router.get("/health")
def health():
    return {"status": "healthy"}

# 🚀 3. app/services/rag_service.py (BUSINESS LOGIC)
import asyncio
from src.retriever import MaritimeHybridRetriever
from src.generator import MaritimeGenerator

service = None


class RAGService:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    async def handle_query(self, prompt: str):

        # Retrieval (runs in thread)
        docs, timings = await asyncio.to_thread(
            self.retriever.retrieve, prompt
        )

        if not docs:
            async def empty_stream():
                yield "No relevant documents found."
            return empty_stream()

        # Generation (stream)
        return self.generator.generate_stream(prompt, docs)


# Startup init
async def init_services():
    global service
    retriever = MaritimeHybridRetriever(use_images=False)
    generator = MaritimeGenerator()

    service = RAGService(retriever, generator)


# Cleanup
async def cleanup_services():
    global service
    service = None

# 🚀 4. app/models/request.py (INPUT VALIDATION)
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    prompt: str = Field(..., max_length=1000)

# 🚀 5. app/core/config.py (CONFIG)
from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Maritime RAG API"
    API_V1_PREFIX: str = "/api/v1"

    ALLOWED_ORIGINS: str = "*"

settings = Settings()

# 🚀 6. app/core/logging.py (LOGGING)
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger("API")

# 🚀 7. .env
# APP_NAME=Maritime RAG API
# API_V1_PREFIX=/api/v1
# ALLOWED_ORIGINS=*

# 🚀 8. requirements.txt
# fastapi
# uvicorn
# pydantic