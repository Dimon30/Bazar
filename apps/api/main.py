from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

from apps.api.routes.events import router as events_router
from apps.api.routes.health import router as health_router
from apps.api.routes.models import router as models_router
from apps.api.routes.search import router as search_router
from bazar_data.database import Database
from rank_system.serving import RetrievalRuntime


def create_app(database_url: str | None = None) -> FastAPI:
    database = Database(database_url or os.environ.get("DATABASE_URL", "data/prod/bazar.db"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database.create_schema()
        runtime = RetrievalRuntime(database)
        runtime.reload()
        app.state.retrieval_runtime = runtime
        yield

    app = FastAPI(
        title="Bazar API",
        version="0.2.0",
        description="Marketplace search and retrieval API",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(search_router)
    app.include_router(events_router)
    app.include_router(models_router)
    return app


app = create_app()
