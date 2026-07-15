from fastapi import Header, HTTPException, Request


def get_session_id(x_session_id: str = Header(..., alias="X-Session-ID")) -> str:
    value = x_session_id.strip()
    if not value or len(value) > 128:
        raise HTTPException(status_code=400, detail="Invalid X-Session-ID")
    return value


def get_assistant(request: Request, session_id: str = Header(..., alias="X-Session-ID")):
    """Return only the assistant belonging to the requesting client."""
    return request.app.state.sessions.get(session_id)