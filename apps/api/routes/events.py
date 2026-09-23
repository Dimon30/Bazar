from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from apps.api.dependencies import get_runtime
from bazar_data.marketplace_schema import search_events
from rank_system.serving import RetrievalRuntime


router = APIRouter(tags=["events"])


class EventRequest(BaseModel):
    request_id: str
    query_text: str = Field(min_length=1, max_length=300)
    product_id: str
    position: int = Field(ge=1)
    event_type: Literal["view", "click", "add_to_cart", "purchase"]
    event_metadata: dict[str, object] = Field(default_factory=dict)


class EventResponse(BaseModel):
    event_id: str
    accepted: bool


@router.post("/events", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
def create_event(event: EventRequest, runtime: RetrievalRuntime = Depends(get_runtime)) -> EventResponse:
    if not runtime.ready or runtime.version is None:
        raise HTTPException(status_code=503, detail="No active retrieval model is loaded.")
    if event.product_id not in runtime.catalog:
        raise HTTPException(status_code=404, detail="Unknown product_id.")
    event_id = str(uuid4())
    runtime.database.upsert(search_events, [{
        "event_id": event_id, "request_id": event.request_id, "event_type": event.event_type,
        "query_text": event.query_text, "product_id": event.product_id, "position": event.position,
        "model_version": runtime.version, "is_simulated": True, "event_metadata": event.event_metadata,
    }], key_columns=("event_id",))
    return EventResponse(event_id=event_id, accepted=True)
