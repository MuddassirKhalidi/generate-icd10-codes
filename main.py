from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from faiss_embedding import faiss_search_icd10

# Initialize FastAPI app
app = FastAPI(
    title="ICD-10 Code Generator API",
    description="API for generating ICD-10 codes from medical text queries",
    version="1.0.0"
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
        description="Number of top ICD-10 codes to return",
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
        "message": "ICD-10 Code Generator API is running",
        "current_dir": os.getcwd(),
        "app_dir_exists": os.path.exists("/app"),
        "archive_dir_exists": os.path.exists("archive"),
        "archive_app_dir_exists": os.path.exists("/app/archive") if os.path.exists("/app") else False,
        "descriptions_file_exists": os.path.exists("archive/icd10data/icd10_descriptions.json"),
        "descriptions_file_app_exists": os.path.exists("/app/archive/icd10data/icd10_descriptions.json") if os.path.exists("/app") else False
    }
    return debug_info

@app.post("/generate-icd10-codes", response_model=ICD10Response)
async def generate_icd10_codes(request: ICD10Request):
    """
    Generate ICD-10 codes for a medical text query.
    
    This endpoint takes a medical text query and returns the most similar
    ICD-10 codes along with their similarity scores and descriptions.
    """
    try:
        if not request.query.strip():
            raise HTTPException(
                status_code=400, 
                detail="Empty query is not allowed"
            )
        
        # Use the existing search_icd10 function
        if request.include_descriptions:
            codes, scores, descriptions = faiss_search_icd10(
                query=request.query, 
                top_k=request.top_k, 
                verbose=True
            )
        else:
            codes, scores = faiss_search_icd10(
                query=request.query, 
                top_k=request.top_k, 
                verbose=False
            )
            descriptions = None
        
        return ICD10Response(
            query=request.query,
            codes=codes,
            scores=scores,
            descriptions=descriptions
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
        from faiss_embedding import faiss_load_embeddings
        embedding_dict, icd10_vocab, pro_vectors, faiss_index = faiss_load_embeddings()
        
        return {
            "status": "healthy",
            "embeddings_loaded": True,
            "vocabulary_size": len(icd10_vocab),
            "embedding_dimension": pro_vectors.shape[1] if len(pro_vectors) > 0 else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "embeddings_loaded": False
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
