from fastapi import APIRouter, Request


router = APIRouter()


def health_response() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "bazar-api",
    }


@router.get("/")
def root() -> dict[str, str]:
    return health_response()


@router.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    return {**health_response(), "retrieval_ready": request.app.state.retrieval_runtime.ready}
