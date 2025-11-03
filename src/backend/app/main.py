from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import io
import pandas as pd
import re
import os

from .models import MatchRequest, MatchResponse, MatchResult, UploadResponse, HealthResponse, ExportRequest
from .tfidf_model import TFIDFModel
from .data_parser import DataParser

# Security constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIME_TYPES = [
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
    'application/vnd.ms-excel',  # .xls
]

# Global variables
search_model = None
dataset = None

def sanitize_excel_value(value):
    """
    Sanitize values to prevent Excel formula injection.
    Prefixes dangerous characters with a single quote to treat them as text.
    """
    if not isinstance(value, str):
        return value
    
    # Characters that could start Excel formulas or commands
    dangerous_chars = ['=', '+', '-', '@', '\t', '\r']
    
    # If the value starts with a dangerous character, prefix with single quote
    if value and value[0] in dangerous_chars:
        return "'" + value
    
    return value

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global search_model
    print("API starting up...")
    print("Waiting for dataset upload...")
    
    yield
    
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="TF-IDF Connection Matcher API",
    description="Upload dataset and match people based on professional background",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - Configured for production
# In production, this should be restricted to your specific domain
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # Disabled for security unless authentication is added
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "TF-IDF Connection Matcher API",
        "endpoints": {
            "POST /upload": "Upload Excel dataset",
            "POST /match": "Find matching connections",
            "GET /health": "Check API status"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API has a dataset loaded."""
    return HealthResponse(
        status="healthy",
        dataset_loaded=dataset is not None,
        dataset_size=len(dataset) if dataset is not None else 0
    )

@app.post("/upload", response_model=UploadResponse, tags=["Data"])
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload an Excel file containing the dataset.
    
    Expected columns:
    - Name
    - Professional Title/Employment & Career
    - Board Service
    
    Security: Max file size 10MB, validates MIME type and file content.
    """
    global dataset, search_model
    
    # Validate filename extension
    if not file.filename or not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only .xlsx and .xls files are allowed."
        )
    
    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type. Expected Excel file, got {file.content_type}"
        )
    
    try:
        # Read file contents with size limit
        contents = await file.read(MAX_FILE_SIZE + 1)
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Parse the dataset
        parser = DataParser()
        df = parser.parse_excel_bytes(io.BytesIO(contents))
        
        # Validate dataset isn't empty
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file contains no data"
            )
        
        # Validate required columns exist
        required_cols = ['name', 'employment', 'board_service', 'employment_norm', 'board_service_norm']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset missing required columns: {', '.join(missing)}"
            )
        
        # Store the dataset globally
        dataset = df
        
        # Initialize and fit the TF-IDF model
        search_model = TFIDFModel(max_features=10000, ngram_range=(1, 2))
        search_model.fit_corpus(df, ['employment_norm', 'board_service_norm'])
        
        return UploadResponse(
            status="success",
            message="Dataset uploaded and indexed successfully",
            rows_loaded=len(df),
            columns=df.columns.tolist()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the actual error server-side but return generic message to client
        print(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to process uploaded file. Please ensure it's a valid Excel file."
        )

@app.post("/match", response_model=MatchResponse, tags=["Search"])
async def match_connections(request: MatchRequest):
    """
    Find matching people based on professional background.
    
    - **query**: Description of person to match (employment, board service, etc.)
    - **top_k**: Number of matches to return (default: 10, max: 100)
    - **min_score**: Minimum normalized score threshold 0-1 (default: 0.8)
    """
    if search_model is None or dataset is None:
        raise HTTPException(
            status_code=400, 
            detail="No dataset loaded. Please upload a dataset first using /upload endpoint."
        )
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    print(f"Received match request: query='{request.query[:50]}...', top_k={request.top_k}, min_score={request.min_score}")
    
    # Perform TF-IDF search with normalized scores and filtering
    results_df = search_model.rank(request.query, top_k=request.top_k, min_score=request.min_score)
    
    # Handle case where no results meet threshold
    if results_df.empty:
        return MatchResponse(
            query=request.query,
            total_matches=0,
            matches=[]
        )
    
    # Convert to response format
    matches = []
    for idx, row in results_df.iterrows():
        matches.append(MatchResult(
            name=row['name'],
            employment=row['employment'],
            board_service=row['board_service'],
            score=float(row['tfidf_score']),
            rank=idx + 1
        ))
    
    return MatchResponse(
        query=request.query,
        total_matches=len(matches),
        matches=matches
    )

@app.post("/export", tags=["Export"])
async def export_matches(request: ExportRequest):
    """
    Export match results to Excel file.
    Security: Sanitizes values to prevent Excel formula injection.
    """
    if not request.matches:
        raise HTTPException(status_code=400, detail="No matches to export")
    
    # Limit number of matches to export (prevent large file DoS)
    if len(request.matches) > 1000:
        raise HTTPException(
            status_code=400, 
            detail="Too many matches to export. Maximum is 1000 records."
        )
    
    try:
        # Convert matches to DataFrame with sanitization
        data = []
        for match in request.matches:
            data.append({
                'Name': sanitize_excel_value(match.name),
                'Employment': sanitize_excel_value(match.employment),
                'Board Service': sanitize_excel_value(match.board_service),
                'Match Score': match.score,
                'Rank': match.rank
            })
        
        df = pd.DataFrame(data)
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Matches')
        
        output.seek(0)
        
        # Return as downloadable file
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=board_matches.xlsx"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log actual error but return generic message
        print(f"Export error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to export results. Please try again."
        )