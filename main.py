from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from modules.icd_extractor import extract_icd10_with_validation
from modules.report_generator import process_consultation
from modules.audio import transcribe_audio

# Initialize FastAPI app
app = FastAPI(
    title="ICD-10 Code Generator API",
    description="API for generating ICD-10 codes from medical text queries using LLM extraction",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8001",
        "https://aim-doc-assist.vercel.app",
        "http://91.98.81.85:8000",
        "https://91.98.81.85:8000"  # Add HTTPS version
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Request model
class ICD10Request(BaseModel):
    lang: str = Field(
        ..., 
        description="Language of the medical text query",
        example="en"
    )
    text: str = Field(
        ..., 
        description="Medical text query to convert to ICD-10 codes",
        example="chest pain"
    )
    top_k: Optional[int] = Field(
        default=3, 
        description="Number of top ICD-10 codes to return per diagnosis",
        ge=1,
        le=500
    )

# Response models
class ICD10Response(BaseModel):
    codes: List[str]
    scores: List[float]
    descriptions: List[str]

@app.get("/")
async def root():
    """Health check endpoint with debug info"""
    import os
    # Debug endpoint to check file structure
    debug_info = {
        "message": "ICD-10 Code Generator API is running (LLM-powered)",
        "current_dir": os.getcwd(),
        "app_dir_exists": os.path.exists("/app"),
        "archive_dir_exists": os.path.exists("archive"),
        "archive_app_dir_exists": os.path.exists("/app/archive") if os.path.exists("/app") else False,
        "descriptions_file_exists": os.path.exists("icd10_descriptions.json"),
        "vectors_file_exists": os.path.exists("icd10_vectors.npz")
    }
    return debug_info

@app.post("/generate_icd10_codes", response_model=ICD10Response)
async def generate_icd10_codes(request: ICD10Request):
    """
    Generate ICD-10 codes for a medical text query using LLM extraction.
    
    This endpoint takes a medical consultation text and:
    1. Uses GPT to extract diagnoses with reasoning
    2. Validates each diagnosis using semantic search
    3. Returns the most similar ICD-10 codes with scores
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Empty text is not allowed"
            )
        
        # Extract and validate ICD-10 codes using LLM + embedding validation
        # This returns: [{"diagnosis": str, "matches": [{"code": str, "description": str, "score": float}]}]
        validated_results = extract_icd10_with_validation(
            consultation=request.text,
            top_k=1,
            threshold=0.6
        )
        
        # Flatten results to maintain API compatibility
        codes = []
        scores = []
        descriptions = []
        
        for result in validated_results:
            if result["matches"]:
                # Add all matches for this diagnosis
                for match in result["matches"]:
                    codes.append(match["code"])
                    scores.append(match["score"])
                    descriptions.append(match["description"])
            else:
                # No valid matches found - could optionally include with score 0.0
                # For now, we skip diagnoses without matches to maintain quality
                pass
        
        # If no codes were found at all, return empty lists
        if not codes:
            codes = []
            scores = []
            descriptions = []
        
        return ICD10Response(
            codes=codes,
            scores=scores,
            descriptions=descriptions
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/process_audio")
async def process_audio_endpoint(audio_file: UploadFile = File(...)):
    """
    Process audio consultation file.
    
    This endpoint accepts audio files of any type and sends the content
    to the process_consultation() function for processing.
    
    Args:
        audio_file: The uploaded audio file
        
    Returns:
        JSON response with processing results
    """
    try:
        # Validate file type (basic check for audio files)
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            # Allow common audio extensions even if content-type is not set
            allowed_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.aiff'}
            file_extension = audio_file.filename.lower().split('.')[-1] if audio_file.filename else ''
            if f'.{file_extension}' not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail="File must be an audio file. Supported formats: mp3, wav, m4a, aac, ogg, flac, wma, aiff"
                )
        
        audio_content = await audio_file.read()
        
        if len(audio_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty audio file not allowed"
            )
        
        transcript = transcribe_audio(audio_content, audio_file.filename)
        
        result = process_consultation(transcript)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Test if embeddings are loaded
        from modules.embedding_en import load_embeddings
        import os
        
        embedding_dict, icd10_vocab, pro_vectors = load_embeddings()
        
        # Check if OpenAI API key is set
        openai_key_set = bool(os.getenv("OPENAI_API_KEY"))
        
        return {
            "status": "healthy",
            "embeddings_loaded": True,
            "vocabulary_size": len(icd10_vocab),
            "embedding_dimension": pro_vectors.shape[1] if len(pro_vectors) > 0 else 0,
            "openai_configured": openai_key_set,
            "extraction_method": "LLM + Semantic Validation"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "embeddings_loaded": False
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)