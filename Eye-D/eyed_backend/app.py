import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes import register, verify, data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eyed_backend")

app = FastAPI(
    title="Eye-D Biometric Authentication API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Eye-D Backend"}

app.include_router(register.router)
app.include_router(verify.router)
app.include_router(data.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Eye-D Backend starting up...")
    os.makedirs("temp_features", exist_ok=True)
    from database.db_handler import DBHandler
    DBHandler()
    logger.info("Database ready.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
