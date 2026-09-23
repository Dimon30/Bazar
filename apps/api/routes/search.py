from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api.dependencies import get_runtime
from rank_system.serving import RetrievalRuntime


router = APIRouter(tags=["search"])


class ProductResult(BaseModel):
    product_id: str
    title: str
    description: str
    brand: str
    color: str
    locale: str
    score: float
    position: int


class SearchResult(BaseModel):
    request_id: str
    query: str
    model_version: str
    model_type: str
    latency_ms: float
    products: list[ProductResult]


@router.get("/search", response_model=SearchResult)
def search(
    q: str = Query(min_length=1, max_length=300),
    limit: int = Query(default=10, ge=1, le=100),
    runtime: RetrievalRuntime = Depends(get_runtime),
) -> SearchResult:
    try:
        response = runtime.search(q, str(uuid4()), limit)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return SearchResult(query=q, **response.__dict__)
