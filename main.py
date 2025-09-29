from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from icd_extractor import extract_icd10_with_validation

# Initialize FastAPI app
app = FastAPI(
    title="ICD-10 Code Generator API",
    description="API for generating ICD-10 codes from medical text queries using LLM extraction",
    version="2.0.0"
)

# Request model
class ICD10Request(BaseModel):
    query: str = Field(
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
    include_descriptions: Optional[bool] = Field(
        default=True,
        description="Whether to include ICD-10 descriptions in the response"
    )

# Response models
class ICD10Response(BaseModel):
    query: str
    codes: List[str]
    scores: List[float]
    descriptions: Optional[List[str]] = None

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

@app.post("/generate-icd10-codes", response_model=ICD10Response)
async def generate_icd10_codes(request: ICD10Request):
    """
    Generate ICD-10 codes for a medical text query using LLM extraction.
    
    This endpoint takes a medical consultation text and:
    1. Uses GPT to extract diagnoses with reasoning
    2. Validates each diagnosis using semantic search
    3. Returns the most similar ICD-10 codes with scores
    """
    try:
        if not request.query.strip():
            raise HTTPException(
                status_code=400, 
                detail="Empty query is not allowed"
            )
        
        # Extract and validate ICD-10 codes using LLM + embedding validation
        # This returns: [{"diagnosis": str, "matches": [{"code": str, "description": str, "score": float}]}]
        validated_results = extract_icd10_with_validation(
            consultation=request.query,
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
            descriptions = [] if request.include_descriptions else None
        
        return ICD10Response(
            query=request.query,
            codes=codes,
            scores=scores,
            descriptions=descriptions if request.include_descriptions else None
        )
        
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
        from embedding_en import load_embeddings
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