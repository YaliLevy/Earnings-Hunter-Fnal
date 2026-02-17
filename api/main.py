"""
FastAPI Backend for Earnings Hunter

This API wraps the existing Python analysis logic and exposes it
for the React frontend to consume.
In production, also serves the React build (SPA).
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

# CORS configuration
allowed_origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative dev
]

# Add Railway domains dynamically
railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
if railway_domain:
    allowed_origins.append(f"https://{railway_domain}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routers
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(quote.router, prefix="/api", tags=["Quote"])
app.include_router(historical.router, prefix="/api", tags=["Historical"])


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


# ========== Serve React Frontend (Production) ==========
# Mount static files if the frontend build exists
frontend_dist = project_root / "frontend" / "dist"

if frontend_dist.exists():
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    # Catch-all: serve index.html for SPA routing
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve React SPA for any non-API route."""
        # Check if it's a static file in dist
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        # Otherwise serve index.html (SPA routing)
        return FileResponse(str(frontend_dist / "index.html"))
else:
    @app.get("/")
    async def root():
        """API root when no frontend build."""
        return {
            "status": "healthy",
            "service": "Earnings Hunter API",
            "version": "2.0.0",
            "frontend": "not built - run 'cd frontend && npm run build'"
        }
