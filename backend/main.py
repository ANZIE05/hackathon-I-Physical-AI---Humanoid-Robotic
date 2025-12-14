from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import uvicorn
import os

from api.rag import router as rag_router

app = FastAPI(
    title="Physical AI & Humanoid Robotics RAG API",
    description="Retrieval-Augmented Generation API for the Physical AI & Humanoid Robotics textbook",
    version="1.0.0",
    # Add security headers and documentation
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Add security headers
)

# Include RAG API routes
app.include_router(rag_router, prefix="/rag", tags=["rag"])

@app.get("/")
def read_root():
    return {"message": "Physical AI & Humanoid Robotics RAG API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)