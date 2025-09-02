from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from embedding import search_icd10

# Initialize FastAPI app
app = FastAPI(
    title="ICD-10 Code Generator API",
    description="API for generating ICD-10 codes from medical text queries",
    version="1.0.0"
)

# Request model
class ICD10Request(BaseModel):
    queries: List[str] = Field(
        ..., 
        description="List of medical text queries to convert to ICD-10 codes",
        min_items=1,
        example=["chest pain", "diabetes mellitus", "hypertension"]
    )
    top_k: Optional[int] = Field(
        default=3, 
        description="Number of top ICD-10 codes to return for each query",
        ge=1,
        le=10
    )
    include_descriptions: Optional[bool] = Field(
        default=False,
        description="Whether to include ICD-10 descriptions in the response"
    )

# Response models
class ICD10Result(BaseModel):
    query: str
    codes: List[str]
    scores: List[float]
    descriptions: Optional[List[str]] = None

class ICD10Response(BaseModel):
    results: List[ICD10Result]
    total_queries: int

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "ICD-10 Code Generator API is running"}

@app.post("/generate-icd10-codes", response_model=ICD10Response)
async def generate_icd10_codes(request: ICD10Request):
    """
    Generate ICD-10 codes for a list of medical text queries.
    
    This endpoint takes a list of medical text queries and returns the most similar
    ICD-10 codes along with their similarity scores.
    """
    try:
        results = []
        
        for query in request.queries:
            if not query.strip():
                raise HTTPException(
                    status_code=400, 
                    detail="Empty queries are not allowed"
                )
            
            # Use the existing search_icd10 function
            if request.include_descriptions:
                codes, scores, descriptions = search_icd10(
                    query, 
                    top_k=request.top_k, 
                    verbose=True
                )
                result = ICD10Result(
                    query=query,
                    codes=codes,
                    scores=scores,
                    descriptions=descriptions
                )
            else:
                codes, scores = search_icd10(
                    query, 
                    top_k=request.top_k, 
                    verbose=False
                )
                result = ICD10Result(
                    query=query,
                    codes=codes,
                    scores=scores
                )
            
            results.append(result)
        
        return ICD10Response(
            results=results,
            total_queries=len(request.queries)
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
        from embedding import load_embeddings
        embedding_dict, icd10_vocab, pro_vectors = load_embeddings()
        
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
