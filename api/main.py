"""FastAPI serving layer for the hybrid recommendation engine."""
import hashlib, logging, time
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)
app = FastAPI(title="Ecommerce Recommendation Engine", version="2.1.0")
Instrumentator().instrument(app).expose(app)

_recommender = None
_redis = None
AB_BUCKETS = ["control", "treatment_v1", "treatment_v2"]

def assign_ab_bucket(user_id: str) -> str:
    return AB_BUCKETS[int(hashlib.md5(user_id.encode()).hexdigest(), 16) % len(AB_BUCKETS)]

class RecommendRequest(BaseModel):
    user_id: str = Field(..., example="u_1234")
    top_k: int = Field(10, ge=1, le=50)
    context: str = Field("homepage")
    exclude_skus: list[str] = Field(default_factory=list)
    candidate_pool: list[str] = Field(default_factory=list)

class RecommendedItem(BaseModel):
    sku: str
    score: float
    source: str

class RecommendResponse(BaseModel):
    user_id: str
    bucket: str
    recommendations: list[RecommendedItem]
    latency_ms: float
    cache_hit: bool

class FeedbackRequest(BaseModel):
    user_id: str
    sku: str
    event: str = Field(..., example="click")
    context: str = "homepage"

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    t0 = time.perf_counter()
    if _redis is None: raise HTTPException(503, "Cache unavailable")
    if _recommender is None: raise HTTPException(503, "Model not loaded")

    cache_key = f"rec:{req.user_id}:{req.context}:{req.top_k}"
    cached = await _redis.get(cache_key)
    if cached:
        return RecommendResponse(user_id=req.user_id, bucket=assign_ab_bucket(req.user_id),
            recommendations=[RecommendedItem(**i) for i in cached],
            latency_ms=round((time.perf_counter()-t0)*1000, 2), cache_hit=True)

    candidates = req.candidate_pool or await _redis.get_candidate_pool(req.context)
    if not candidates: raise HTTPException(422, "No candidate SKUs available")

    results = _recommender.recommend(req.user_id, candidates, req.top_k, req.exclude_skus)
    serialized = [{"sku": r.sku, "score": r.score, "source": r.source} for r in results]
    await _redis.set(cache_key, serialized, ttl=300)

    return RecommendResponse(user_id=req.user_id, bucket=assign_ab_bucket(req.user_id),
        recommendations=[RecommendedItem(**i) for i in serialized],
        latency_ms=round((time.perf_counter()-t0)*1000, 2), cache_hit=False)

@app.post("/feedback", status_code=204)
async def feedback(req: FeedbackRequest):
    if _redis: await _redis.xadd("feedback_stream", {**req.dict(), "ts": int(time.time())})

@app.get("/ab-assign/{user_id}")
async def ab_assign(user_id: str):
    return {"user_id": user_id, "bucket": assign_ab_bucket(user_id)}

@app.get("/health")
async def health():
    return {"status": "ok"}
