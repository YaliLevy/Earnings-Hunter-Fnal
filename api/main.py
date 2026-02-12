"""
FastAPI Backend for Earnings Hunter

This API wraps the existing Python analysis logic and exposes it
for the React frontend to consume.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routers import analysis, quote, historical


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("Earnings Hunter API starting up...")
    yield
    # Shutdown
    print("Earnings Hunter API shutting down...")


app = FastAPI(
    title="Earnings Hunter API",
    description="AI-powered earnings analysis with Golden Triangle methodology",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev
        "https://*.vercel.app",   # Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(quote.router, prefix="/api", tags=["Quote"])
app.include_router(historical.router, prefix="/api", tags=["Historical"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Earnings Hunter API",
        "version": "2.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "api": "running",
        "ml_models": "loaded",
        "fmp_client": "ready"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )
