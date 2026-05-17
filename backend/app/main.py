from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from datetime import datetime
import os
import uuid

# Import routers
from backend.app.routers import patients, predictions
from backend.app.config.settings import get_settings
from backend.app.config.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID for traceability"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Custom middleware for logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with processing time"""

    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time*1000:.2f}ms"
        )

        response.headers["X-Process-Time"] = str(process_time)
        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        process_time = time.time() - start_time

        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)},
            headers={"X-Process-Time": str(process_time)}
        )

# Global exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    
    logger.info("="*70)
    logger.info("STARTING CLINICAL LAB ANALYSIS API")
    logger.info("="*70)
    logger.info(f"Environment: {settings.DEBUG and 'DEBUG' or 'PRODUCTION'}")
    logger.info(f"Host: {settings.HOST}")
    logger.info(f"Port: {settings.PORT}")
    
    try:
        # Load models
        from backend.app.services.model_loader import get_model_loader
        model_loader = get_model_loader(
            model_path=settings.MODEL_PATH,
            use_cache=settings.ENABLE_MODEL_CACHE
        )
        
        if model_loader.is_ready():
            logger.info("ALL MODELS LOADED AND READY")
        else:
            logger.warning("Some models failed to load")
            logger.warning(f"Status: {model_loader.get_load_status()}")
        
        logger.info("="*70)
        logger.info("API READY FOR REQUESTS")
        logger.info("="*70)
    
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")

# Include routers
app.include_router(patients.router)
app.include_router(predictions.router)

# Error handler for 404
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    logger.warning(f"404 Not Found: {request.url.path}")
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": request.url.path}
    )

# Error handler for 500
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
