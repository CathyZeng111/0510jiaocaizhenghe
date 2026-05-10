from typing import Literal, Optional

from pydantic import BaseModel, Field


class RagChapterInput(BaseModel):
    chapter_id: str
    title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    content: str
    char_count: int


class RagTextbookInput(BaseModel):
    textbook_id: str
    filename: str
    title: str
    total_pages: Optional[int] = None
    total_chars: int
    chapters: list[RagChapterInput]


class RagChunkMetadata(BaseModel):
    textbook_id: str
    textbook_title: str
    filename: str
    chapter_id: str
    chapter_title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_index: int
    char_start: int
    char_end: int
    retrieval_model: str = "keyword_char_ngram_v1"
    embedding_model: Optional[str] = None
    chunk_size: int = 800
    chunk_overlap: int = 80


class RagIndexRequest(BaseModel):
    textbooks: list[RagTextbookInput]
    index_id: str = "default"
    chunk_size: int = Field(default=800, ge=500, le=800)
    chunk_overlap: int = Field(default=80, ge=50, le=100)
    embedding_model: Optional[str] = None


class RagIndexResponse(BaseModel):
    index_id: str
    textbook_count: int
    chapter_count: int
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: Optional[str] = None
    embedding_count: int = 0
    vector_store_type: str = "none"
    chunking_strategy: str = "sliding_window_800_overlap_80"


class RagIndexJobStartResponse(BaseModel):
    job_id: str
    status: str
    index_id: str
    estimated_chunk_count: int = 0


class RagIndexStatusResponse(BaseModel):
    job_id: str
    status: str
    index_id: str
    textbook_count: int = 0
    chapter_count: int = 0
    chunk_count: int = 0
    embedding_count: int = 0
    estimated_chunk_count: int = 0
    vector_store_type: str = "none"
    error: Optional[str] = None
    result: Optional[RagIndexResponse] = None


class RagQueryRequest(BaseModel):
    query: str
    index_id: str = "default"
    textbooks: Optional[list[RagTextbookInput]] = None
    top_k: int = Field(default=5, ge=1, le=10)
    min_score: float = Field(default=0.08, ge=0, le=1)
    retrieval_mode: Literal["hybrid", "keyword", "char_ngram", "vector"] = "hybrid"
    answer_mode: Literal["auto", "extractive", "llm"] = "auto"
    query_embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None


class RagCitation(BaseModel):
    citation_id: str
    chunk_id: str
    textbook_id: str
    textbook_title: str
    filename: str
    chapter_id: str
    chapter_title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    score: float


class RagSourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: RagChunkMetadata


class RagRetrievalInfo(BaseModel):
    index_id: str
    retrieval_mode: Literal["hybrid", "keyword", "char_ngram", "vector"]
    retrieval_model: str
    embedding_model: Optional[str] = None
    embedding_count: int = 0
    vector_store_type: str = "none"
    chunking_strategy: str = "sliding_window_800_overlap_80"
    top_k: int
    min_score: float
    matched_chunks: int


class RagQueryResponse(BaseModel):
    answer: str
    citations: list[RagCitation]
    source_chunks: list[RagSourceChunk]
    retrieval: RagRetrievalInfo
