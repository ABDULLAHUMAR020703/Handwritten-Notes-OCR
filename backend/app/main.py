from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from backend directory
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    logger_env = logging.getLogger(__name__)
    logger_env.info(f"✅ Loaded environment variables from: {env_path}")
except ImportError:
    # python-dotenv not installed - skip .env loading
    pass
except Exception as e:
    # .env file not found or error loading - continue without it
    pass

from app.api import convert, upload, download

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Handwritten Notes OCR to Word")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(convert.router, prefix="/api", tags=["convert"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(download.router, prefix="/api", tags=["download"])

@app.on_event("startup")
async def startup_event():
    """Log server startup information."""
    logger.info("=" * 60)
    logger.info("🚀 Handwritten Notes OCR API Server Starting...")
    logger.info("=" * 60)
    logger.info(f"📅 Server started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("🌐 Server will be available at: http://0.0.0.0:8000")
    logger.info("📚 API Documentation: http://0.0.0.0:8000/docs")
    logger.info("💚 Health Check: http://0.0.0.0:8000/")
    logger.info("=" * 60)

@app.get("/")
async def root():
    logger.info("✅ Health check endpoint accessed")
    return {"status": "ok", "message": "Handwritten Notes OCR API"}

if __name__ == "__main__":
    logger.info("Starting server with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
