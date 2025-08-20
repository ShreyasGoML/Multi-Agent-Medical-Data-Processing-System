from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import router
import uvicorn

# Create FastAPI app with metadata
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Entry point for running locally
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # assumes this file is named main.py
        host="0.0.0.0",
        port=8000,
        reload=True
    )
