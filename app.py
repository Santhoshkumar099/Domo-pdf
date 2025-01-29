from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import requests
import json
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging
from typing import Dict

# Load environment variables
load_dotenv()

app = FastAPI(title="PDF Text Extraction and Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMO_DEVELOPER_TOKEN = os.getenv("DOMO_DEVELOPER_TOKEN")
API_URL = "https://gwcteq-partner.domo.com/api/ai/v1/text/generation"

if not DOMO_DEVELOPER_TOKEN:
    raise RuntimeError("Missing Domo Developer Token. Set it in your environment variables.")

class Query(BaseModel):
    question: str

# In-memory storage for PDF content (consider a database for production)
pdf_storage: Dict[str, str] = {}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file given its byte content."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def query_domo_api(question: str, context: str) -> str:
    """Queries the Domo API with the extracted text as context."""
    payload = {
        "input": question,
        "model": "domo.domo_ai.domogpt-chat-small-v1:anthropic",
        "system": f"You are a chatbot that answers questions based on the following PDF text: {context}. Provide concise answers, limited to 3-4 lines, ensuring clarity and relevance."
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-DOMO-Developer-Token": DOMO_DEVELOPER_TOKEN
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        response_data = response.json()
        if "output" not in response_data:
            logger.error("Unexpected response from Domo API")
            raise HTTPException(status_code=500, detail="Unexpected response from Domo API")
        return response_data["output"]

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Domo API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling Domo API: {str(e)}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF and extracts text."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    contents = await file.read()  # Read file only once
    pdf_text = extract_text_from_pdf(contents)
    
    # Store PDF text in memory with a unique identifier (e.g., session ID)
    pdf_storage["current_pdf"] = pdf_text
    
    return {"message": "PDF processed successfully", "text_length": len(pdf_text)}

@app.post("/ask-question/")
async def ask_question(query: Query):
    """Handles question answering based on the uploaded PDF content."""
    if "current_pdf" not in pdf_storage:
        raise HTTPException(status_code=400, detail="Please upload a PDF first")
    
    response = query_domo_api(query.question, pdf_storage["current_pdf"])
    return {"answer": response}

@app.get("/")
async def root():
    return {"message": "Welcome to PDF Analysis API. Use /upload-pdf/ to upload a PDF and /ask-question/ to ask questions about it."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render's PORT environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)
