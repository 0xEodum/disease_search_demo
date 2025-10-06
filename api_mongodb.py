"""
FastAPI СЃРµСЂРІРµСЂ СЃ РёРЅС‚РµРіСЂР°С†РёРµР№ MongoDB
РџРѕРґРґРµСЂР¶РёРІР°РµС‚ РєР°Рє JSON С„Р°Р№Р»С‹, С‚Р°Рє Рё MongoDB РІ РєР°С‡РµСЃС‚РІРµ РёСЃС‚РѕС‡РЅРёРєР° РґР°РЅРЅС‹С…
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from pymongo import MongoClient
from disease_search_engine import MedicalLabAnalyzer
from sync_manager import SyncManager
import time
import os
import asyncio

app = FastAPI(
    title="Medical Lab Disease Search API with MongoDB",
    description="(JSON + MongoDB)",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


analyzer: Optional[MedicalLabAnalyzer] = None
sync_manager: Optional[SyncManager] = None
mongodb_client: Optional[MongoClient] = None
startup_time = time.time()

USE_MONGODB = "true"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "medical_lab")
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "3600"))


class TestInput(BaseModel):
    name: str = Field(..., description="РќР°Р·РІР°РЅРёРµ С‚РµСЃС‚Р°")
    value: str = Field(..., description="Р—РЅР°С‡РµРЅРёРµ")
    units: str = Field(..., description="Р•РґРёРЅРёС†С‹")


class AnalysisRequest(BaseModel):
    tests: List[TestInput]
    gender: str = Field("unisex", description="male/female/unisex")
    top_k: int = Field(10, ge=1, le=50)
    categories: Optional[List[str]] = None


class DiseaseResult(BaseModel):
    disease_id: str
    canonical_name: str
    matched_patterns: int
    total_patterns: int
    matched_score: float
    contradiction_penalty: float
    total_score: float
    max_possible_score: float
    normalized_score: float
    matched_details: List[dict] = []
    contradictions: List[dict] = []
    missing_data: List[dict] = []
    redundant_data: List[dict] = []
    expected_patterns: List[dict] = []


class AnalysisResponse(BaseModel):
    success: bool
    processing_time_ms: float
    results: List[DiseaseResult]
    total_found: int
    data_source: str

class TestExplanationRequest(BaseModel):
    tests: List[TestInput]
    gender: str = Field("unisex", description="male/female/unisex")


class ValueWithUnits(BaseModel):
    value: Optional[Any] = None
    units: Optional[str] = None


class StatusValue(BaseModel):
    value: Optional[Any] = None


class TestExplanationItem(BaseModel):
    test_name: ValueWithUnits
    user_value: ValueWithUnits
    reference_value: ValueWithUnits
    status: StatusValue


class TestExplanationResponse(BaseModel):
    success: bool
    data_source: str
    total: int
    items: List[TestExplanationItem]


class HealthResponse(BaseModel):
    status: str
    data_source: str
    diseases_loaded: int
    references_loaded: int
    uptime_seconds: float
    sync_status: Optional[dict] = None


class SyncStatusResponse(BaseModel):
    enabled: bool
    status: Optional[dict] = None


# ============================================================
# Lifecycle
# ============================================================

@app.on_event("startup")
async def startup():
    global analyzer, sync_manager, mongodb_client
    
    print("=" * 60)
    print("Starting Medical Lab API v2.0")
    print("=" * 60)
    print(f"  Data source: {'MongoDB' if USE_MONGODB else 'JSON files'}")
    
    if USE_MONGODB:
        print(f"  MongoDB URI: {MONGODB_URI}")
        print(f"  Database: {MONGODB_DB}")
        print(f"  Sync interval: {SYNC_INTERVAL}s")
    
    print("=" * 60)
    
    try:
        if USE_MONGODB:
            mongodb_client = MongoClient(MONGODB_URI)

            mongodb_client.admin.command('ping')
            print("вњ“ Connected to MongoDB")

            analyzer = MedicalLabAnalyzer(mongodb_client=mongodb_client)

            sync_manager = SyncManager(
                analyzer=analyzer,
                mongodb_client=mongodb_client,
                db_name=MONGODB_DB,
                check_interval=SYNC_INTERVAL
            )

            asyncio.create_task(sync_manager.start())
            
        else:
            analyzer = MedicalLabAnalyzer()
            analyzer.load_references('ref_blood.json')
            analyzer.load_diseases('diseases.json')
            print("вњ“ Loaded data from JSON files")
        
        print("вњ… Search engine initialized successfully")
        
    except Exception as e:
        print(f"вќЊ Initialization failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    print("\nрџ‘‹ Shutting down...")
    
    if sync_manager:
        sync_manager.stop()
    
    if mongodb_client:
        mongodb_client.close()


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Medical Lab Disease Search API v2.0",
        "data_source": "mongodb" if USE_MONGODB else "json",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    uptime = time.time() - startup_time
    
    # Sync status
    sync_status_data = None
    if sync_manager:
        sync_status_data = sync_manager.get_status()
    
    return HealthResponse(
        status="healthy",
        data_source="mongodb" if USE_MONGODB else "json",
        diseases_loaded=len(analyzer.search_engine.diseases) if analyzer.search_engine else 0,
        references_loaded=len(analyzer.reference_manager.references) if analyzer.reference_manager else 0,
        uptime_seconds=uptime,
        sync_status=sync_status_data
    )


@app.post("/api/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_tests(request: AnalysisRequest):
    if not analyzer or not analyzer.search_engine:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized"
        )
    
    try:
        start_time = time.time()
        
        results = analyzer.analyze_patient(
            tests=[t.dict() for t in request.tests],
            gender=request.gender,
            top_k=request.top_k,
            categories=request.categories
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        disease_results = [
            DiseaseResult(
                disease_id=r.disease_id,
                canonical_name=r.canonical_name,
                matched_patterns=r.matched_patterns,
                total_patterns=r.total_patterns,
                matched_score=r.matched_score,
                contradiction_penalty=r.contradiction_penalty,
                total_score=r.total_score,
                max_possible_score=r.max_possible_score,
                normalized_score=r.normalized_score,
                matched_details=r.matched_details,
                contradictions=r.contradictions,
                missing_data=r.missing_data,
                redundant_data=r.redundant_data,
                expected_patterns=r.expected_patterns
            )
            for r in results
        ]
        
        return AnalysisResponse(
            success=True,
            processing_time_ms=processing_time,
            results=disease_results,
            total_found=len(disease_results),
            data_source="mongodb" if USE_MONGODB else "json"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Analysis failed",
                "message": str(e)
            }
        )

@app.post("/api/tests/explanations", response_model=TestExplanationResponse, tags=["Analysis"])
async def explain_user_data(request: TestExplanationRequest):
    """Return explanation details for each provided laboratory test."""
    if not analyzer or not analyzer.reference_manager:
        raise HTTPException(
            status_code=503,
            detail="Reference manager not initialized"
        )

    try:
        explanation_dicts = analyzer.explain_tests(
            tests=[t.dict() for t in request.tests],
            gender=request.gender
        )

        items = [
            TestExplanationItem(
                test_name=ValueWithUnits(**entry['test_name']),
                user_value=ValueWithUnits(**entry['user_value']),
                reference_value=ValueWithUnits(**entry['reference_value']),
                status=StatusValue(**entry['status'])
            )
            for entry in explanation_dicts
        ]

        return TestExplanationResponse(
            success=True,
            data_source="mongodb" if USE_MONGODB else "json",
            total=len(items),
            items=items
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to build explanations",
                "message": str(e)
            }
        )


@app.get("/api/sync/status", response_model=SyncStatusResponse, tags=["Sync"])
async def get_sync_status():
    if not USE_MONGODB or not sync_manager:
        return SyncStatusResponse(
            enabled=False,
            status={"message": "Sync not enabled (using JSON files)"}
        )
    
    status = sync_manager.get_status()
    
    return SyncStatusResponse(
        enabled=True,
        status=status
    )


@app.post("/api/sync/force", tags=["Sync"])
async def force_sync():
    if not USE_MONGODB or not sync_manager:
        raise HTTPException(
            status_code=400,
            detail="Sync not available in JSON mode"
        )
    
    try:
        result = sync_manager.force_sync()
        return {
            "success": True,
            **result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Sync failed",
                "message": str(e)
            }
        )


@app.post("/api/reload", tags=["Admin"])
async def reload_data():
    if USE_MONGODB:
        raise HTTPException(
            status_code=400,
            detail="Use /api/sync/force for MongoDB mode"
        )
    
    try:
        analyzer.load_references('ref_blood.json')
        analyzer.load_diseases('diseases.json')
        
        return {
            "success": True,
            "message": "Data reloaded from JSON files",
            "diseases_loaded": len(analyzer.search_engine.diseases),
            "references_loaded": len(analyzer.reference_manager.references)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Reload failed",
                "message": str(e)
            }
        )


@app.get("/api/diseases", tags=["Info"])
async def list_diseases(
    category: Optional[str] = Query(None, description="")
):
    if not analyzer or not analyzer.search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    diseases = []
    
    for disease in analyzer.search_engine.diseases.values():
        if category:
            categories = set(p.category for p in disease.patterns)
            if category not in categories:
                continue
        
        diseases.append({
            "disease_id": disease.disease_id,
            "canonical_name": disease.canonical_name,
            "total_patterns": len(disease.patterns),
            "max_idf_score": disease.max_idf_score,
            "categories": list(set(p.category for p in disease.patterns))
        })
    
    return {
        "total": len(diseases),
        "diseases": diseases
    }


@app.get("/api/tests", tags=["Info"])
async def list_tests(category: Optional[str] = Query(None, description="")):
    if not analyzer or not analyzer.reference_manager:
        raise HTTPException(status_code=503, detail="Reference manager not initialized")
    
    tests = []
    
    for cat, tests_dict in analyzer.reference_manager.references.items():
        if category and cat != category:
            continue
        
        for test_name, test_data in tests_dict.items():
            tests.append({
                "test_name": test_name,
                "category": cat,
                "alt_names": test_data.get('alt_names', []),
                "units": test_data.get('units', '')
            })
    
    return {
        "total": len(tests),
        "tests": tests
    }



if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_mongodb:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
