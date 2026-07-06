from fastapi import Request

def get_assistant(request: Request):
    """Dependency to retrieve the assistant instance from app state."""
    return request.app.state.assistant
