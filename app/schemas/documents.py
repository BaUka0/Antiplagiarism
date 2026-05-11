from typing import Optional

from pydantic import BaseModel, ConfigDict
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str
    source: Optional[str] = None
    year: Optional[int] = None

class DocumentCreate(DocumentBase):
    pass

class DocumentRead(DocumentBase):
    id: int
    status: str
    total_pages: int
    text_len: int
    clean_word_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class DocumentDetailed(DocumentRead):
    raw_text: Optional[str] = None
    body_text: Optional[str] = None
    clean_text: Optional[str] = None
    n_chunks: int = 0
