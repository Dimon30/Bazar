from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_runtime
from rank_system.serving import RetrievalRuntime


router = APIRouter(prefix="/models", tags=["models"])


@router.get("/active")
def get_active_model(runtime: RetrievalRuntime = Depends(get_runtime)) -> dict[str, object]:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="No active retrieval model is loaded.")
    return {"model_version": runtime.version, "model_type": runtime.model_type, "catalog_size": len(runtime.catalog)}


@router.post("/reload")
def reload_active_model(runtime: RetrievalRuntime = Depends(get_runtime)) -> dict[str, object]:
    runtime.reload()
    return {"ready": runtime.ready, "model_version": runtime.version, "model_type": runtime.model_type}
