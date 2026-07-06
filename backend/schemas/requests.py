from pydantic import BaseModel

class ConfigRequest(BaseModel):
    provider: str
    # API keys are read from the .env file — no user input required

class RemoveDocumentRequest(BaseModel):
    filename: str
