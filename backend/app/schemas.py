from typing import Literal, Optional

from pydantic import BaseModel


class Chapter(BaseModel):
    chapter_id: str
    title: str
    page_start: Optional[int]
    page_end: Optional[int]
    content: str
    char_count: int


class Textbook(BaseModel):
    textbook_id: str
    filename: str
    title: str
    total_pages: Optional[int]
    total_chars: int
    chapters: list[Chapter]


class UploadResult(BaseModel):
    filename: str
    format: str
    size: int
    status: Literal["completed", "failed"]
    error: Optional[str] = None
    textbook: Optional[Textbook] = None


class UploadResponse(BaseModel):
    results: list[UploadResult]
