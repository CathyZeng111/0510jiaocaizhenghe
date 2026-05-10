from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from .parser_quality import ParserQualityReport, assess_textbook, assess_upload_result
from .schemas import Chapter, Textbook, UploadResult


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 160


@dataclass(frozen=True)
class TextbookChunk:
    chunk_id: str
    textbook_id: str
    chapter_id: str
    chapter_title: str
    text: str
    char_start: int
    char_end: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "textbook_id": self.textbook_id,
            "chapter_id": self.chapter_id,
            "chapter_title": self.chapter_title,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


@dataclass(frozen=True)
class TextbookRecord:
    textbook: Textbook
    source_signature: str
    parser_version: str
    stored_at: datetime
    quality: ParserQualityReport
    chunks: list[TextbookChunk] = field(default_factory=list)

    @property
    def textbook_id(self) -> str:
        return self.textbook.textbook_id


class InMemoryTextbookStore:
    """Process-local cache for parsed textbooks and retrieval-ready chunks."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._lock = threading.RLock()
        self._records: dict[str, TextbookRecord] = {}
        self._upload_results: dict[str, UploadResult] = {}
        self._signature_to_textbook_id: dict[str, str] = {}

    def cache_upload_result(
        self,
        result: UploadResult,
        *,
        parser_version: str = "p0-parser-v1",
    ) -> UploadResult:
        signature = upload_result_signature(result)
        with self._lock:
            self._upload_results[signature] = result

        if result.status != "completed" or result.textbook is None:
            assess_upload_result(result)
            return result

        textbook = self.put_textbook(
            result.textbook,
            source_signature=signature,
            parser_version=parser_version,
        )
        cached_result = copy_upload_result(result, textbook=textbook)
        with self._lock:
            self._upload_results[signature] = cached_result
        return cached_result

    def put_textbook(
        self,
        textbook: Textbook,
        *,
        source_signature: Optional[str] = None,
        parser_version: str = "p0-parser-v1",
    ) -> Textbook:
        signature = source_signature or textbook_signature(textbook)

        with self._lock:
            existing_id = self._signature_to_textbook_id.get(signature)
            if existing_id and existing_id in self._records:
                return self._records[existing_id].textbook

            textbook_id = self._unique_textbook_id(textbook.textbook_id, signature)
            stored_textbook = (
                textbook
                if textbook_id == textbook.textbook_id
                else copy_textbook(textbook, textbook_id=textbook_id)
            )
            record = TextbookRecord(
                textbook=stored_textbook,
                source_signature=signature,
                parser_version=parser_version,
                stored_at=datetime.now(timezone.utc),
                quality=assess_textbook(stored_textbook),
                chunks=build_chunks(
                    stored_textbook,
                    chunk_size=self._chunk_size,
                    overlap=self._chunk_overlap,
                ),
            )
            self._records[textbook_id] = record
            self._signature_to_textbook_id[signature] = textbook_id
            return stored_textbook

    def get_textbook(self, textbook_id: str) -> Optional[Textbook]:
        record = self.get_record(textbook_id)
        return record.textbook if record else None

    def get_record(self, textbook_id: str) -> Optional[TextbookRecord]:
        with self._lock:
            return self._records.get(textbook_id)

    def get_upload_result(self, signature: str) -> Optional[UploadResult]:
        with self._lock:
            return self._upload_results.get(signature)

    def list_textbooks(self) -> list[Textbook]:
        with self._lock:
            return [record.textbook for record in self._records.values()]

    def list_records(self) -> list[TextbookRecord]:
        with self._lock:
            return list(self._records.values())

    def get_chapters(self, textbook_id: str) -> list[Chapter]:
        record = self.get_record(textbook_id)
        return list(record.textbook.chapters) if record else []

    def get_chunks(self, textbook_id: Optional[str] = None) -> list[TextbookChunk]:
        with self._lock:
            if textbook_id is not None:
                record = self._records.get(textbook_id)
                return list(record.chunks) if record else []

            chunks: list[TextbookChunk] = []
            for record in self._records.values():
                chunks.extend(record.chunks)
            return chunks

    def get_rag_documents(self, textbook_id: Optional[str] = None) -> list[dict[str, Any]]:
        return [
            {"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}
            for chunk in self.get_chunks(textbook_id)
        ]

    def get_graph_seed(self, textbook_id: str) -> Optional[dict[str, Any]]:
        record = self.get_record(textbook_id)
        if record is None:
            return None

        return {
            "textbook": {
                "id": record.textbook.textbook_id,
                "title": record.textbook.title,
                "filename": record.textbook.filename,
                "total_pages": record.textbook.total_pages,
                "total_chars": record.textbook.total_chars,
            },
            "chapters": [
                {
                    "id": chapter.chapter_id,
                    "title": chapter.title,
                    "page_start": chapter.page_start,
                    "page_end": chapter.page_end,
                    "char_count": chapter.char_count,
                }
                for chapter in record.textbook.chapters
            ],
            "quality": record.quality.to_dict(),
        }

    def delete_textbook(self, textbook_id: str) -> bool:
        with self._lock:
            record = self._records.pop(textbook_id, None)
            if record is None:
                return False
            self._signature_to_textbook_id.pop(record.source_signature, None)
            return True

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._upload_results.clear()
            self._signature_to_textbook_id.clear()

    def _unique_textbook_id(self, requested_id: str, signature: str) -> str:
        if requested_id not in self._records:
            return requested_id

        suffix = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
        candidate = f"{requested_id}_{suffix}"
        counter = 2
        while candidate in self._records:
            candidate = f"{requested_id}_{suffix}_{counter}"
            counter += 1
        return candidate


textbook_store = InMemoryTextbookStore()


def cache_upload_result(
    result: UploadResult,
    *,
    parser_version: str = "p0-parser-v1",
) -> UploadResult:
    return textbook_store.cache_upload_result(result, parser_version=parser_version)


def build_chunks(
    textbook: Textbook,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextbookChunk]:
    chunks: list[TextbookChunk] = []
    stride = max(1, chunk_size - overlap)

    for chapter in textbook.chapters:
        content = chapter.content.strip()
        if not content:
            continue

        offset = 0
        chunk_index = 1
        while offset < len(content):
            end = min(len(content), offset + chunk_size)
            text = content[offset:end].strip()
            if text:
                chunks.append(
                    TextbookChunk(
                        chunk_id=(
                            f"{textbook.textbook_id}:{chapter.chapter_id}:"
                            f"chunk_{chunk_index:03d}"
                        ),
                        textbook_id=textbook.textbook_id,
                        chapter_id=chapter.chapter_id,
                        chapter_title=chapter.title,
                        text=text,
                        char_start=offset,
                        char_end=end,
                        page_start=chapter.page_start,
                        page_end=chapter.page_end,
                    )
                )
                chunk_index += 1
            offset += stride

    return chunks


def upload_result_signature(result: UploadResult) -> str:
    payload = [
        result.filename,
        result.format,
        str(result.size),
        result.status,
        result.error or "",
    ]
    if result.textbook is not None:
        payload.append(textbook_signature(result.textbook))
    return stable_hash(payload)


def textbook_signature(textbook: Textbook) -> str:
    chapter_payload = [
        (
            chapter.title,
            str(chapter.page_start),
            str(chapter.page_end),
            str(chapter.char_count),
            hashlib.sha1(chapter.content.encode("utf-8")).hexdigest(),
        )
        for chapter in textbook.chapters
    ]
    return stable_hash(
        [
            textbook.filename,
            textbook.title,
            str(textbook.total_pages),
            str(textbook.total_chars),
            repr(chapter_payload),
        ]
    )


def stable_hash(parts: Iterable[str]) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def copy_upload_result(result: UploadResult, **updates: Any) -> UploadResult:
    if hasattr(result, "model_copy"):
        return result.model_copy(update=updates)
    return result.copy(update=updates)


def copy_textbook(textbook: Textbook, **updates: Any) -> Textbook:
    if hasattr(textbook, "model_copy"):
        return textbook.model_copy(update=updates)
    return textbook.copy(update=updates)
