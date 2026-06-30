from fastapi import FastAPI

from apps.api.routes.health import router as health_router


app = FastAPI(
    title="Bazar API",
    version="0.1.0",
    description="Marketplace ML system API",
)

app.include_router(health_router)
