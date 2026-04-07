# eyed_backend/app.py
# Main FastAPI application for the Eye-D biometric authentication system.
# Registers routers, sets up CORS, adds logging, global exception handlers,
# and provides a health check endpoint.

import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes import register, verify
import os

# -------------------------------
# 1. LOGGING CONFIGURATION
# -------------------------------
# Logging helps debug issues and monitor system health.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eyed_backend")

# -------------------------------
# 2. CREATE FASTAPI APP
# -------------------------------
app = FastAPI(
    title="Eye-D Biometric Authentication API",
    description="Authenticates users based on their unique eyelid squeeze sequence (kinetics, 3D path, wrinkles, etc.)",
    version="1.0.0",
    docs_url="/docs",          # Swagger UI at /docs
    redoc_url="/redoc",        # ReDoc at /redoc
)

# -------------------------------
# 3. CORS MIDDLEWARE (for web/mobile clients)
# -------------------------------
# CORS allows your frontend (e.g., React, Flutter) to call this API from a different origin.
# In production, restrict allowed_origins to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                     # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],                     # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],                     # Allow all headers
)

# -------------------------------
# 4. GLOBAL EXCEPTION HANDLERS
# -------------------------------
# These ensure that errors are returned in a consistent JSON format,
# and unexpected errors don't crash the server.

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles HTTP exceptions (400, 404, 500) with a JSON response."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catches any unhandled exception, logs it, and returns a 500 error."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# -------------------------------
# 5. HEALTH CHECK ENDPOINT
# -------------------------------
# Useful for load balancers and monitoring.
@app.get("/health", tags=["System"])
async def health_check():
    """Returns a simple status to confirm the API is running."""
    return {"status": "ok", "service": "Eye-D Backend"}

# -------------------------------
# 6. REGISTER ROUTERS (from routes/ folder)
# -------------------------------
# The register and verify modules each contain a FastAPI router.
app.include_router(register.router)
app.include_router(verify.router)

# Optionally, you can add a router for incremental updates later.
# from routes import update
# app.include_router(update.router)

# -------------------------------
# 7. STARTUP AND SHUTDOWN EVENTS (optional)
# -------------------------------
@app.on_event("startup")
async def startup_event():
    """Runs when the server starts. Use it to initialize resources (e.g., database pool)."""
    logger.info("Eye-D Backend starting up...")
    # Ensure necessary directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("temp_features", exist_ok=True)
    # Initialize database tables (if not already done)
    from db_handler import init_db
    init_db()
    logger.info("Database initialized (if not already present).")

@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the server shuts down. Clean up resources here."""
    logger.info("Eye-D Backend shutting down.")

# -------------------------------
# 8. RUN THE APP (only when executed directly)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    # uvicorn.run() starts the ASGI server.
    # host="0.0.0.0" makes the server accessible from other machines on the network.
    # port=8000 is the default FastAPI port.
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # Auto‑reload on code changes (disable in production)
        log_level="info"
    )
