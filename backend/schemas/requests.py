from typing import Literal

from pydantic import BaseModel


class ConfigRequest(BaseModel):
    provider: Literal["deepseek", "openai", "anthropic", "google", "local"]


class RemoveDocumentRequest(BaseModel):
    filename: str