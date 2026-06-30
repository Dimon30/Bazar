from fastapi import APIRouter


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
def health() -> dict[str, str]:
    return health_response()
