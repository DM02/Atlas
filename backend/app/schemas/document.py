import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    mime_type: str
    status: str
    created_at: datetime


class DocumentStatusOut(BaseModel):
    id: uuid.UUID
    status: str
    latest_version_number: int | None = None
    error_message: str | None = None


class DocumentVersionOut(BaseModel):
    id: uuid.UUID
    version_number: int
    created_at: datetime
    is_current: bool


class ChunkOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    version_number: int
    chunk_index: int
    content: str
    page_number: int | None
    section_title: str | None
