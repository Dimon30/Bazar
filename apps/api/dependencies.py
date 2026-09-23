from fastapi import Request

from rank_system.serving import RetrievalRuntime


def get_runtime(request: Request) -> RetrievalRuntime:
    return request.app.state.retrieval_runtime
