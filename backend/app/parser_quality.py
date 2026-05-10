from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .schemas import Textbook, UploadResult


@dataclass(frozen=True)
class ParserQualityReport:
    score: float
    level: str
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_upload_result(result: UploadResult) -> ParserQualityReport:
    if result.status != "completed" or result.textbook is None:
        return ParserQualityReport(
            score=0.0,
            level="failed",
            warnings=[result.error or "解析失败，未生成可复用教材内容"],
            metrics={
                "filename": result.filename,
                "format": result.format,
                "size": result.size,
            },
        )

    return assess_textbook(result.textbook)


def assess_textbook(textbook: Textbook) -> ParserQualityReport:
    warnings: list[str] = []
    metrics: dict[str, Any] = {
        "textbook_id": textbook.textbook_id,
        "filename": textbook.filename,
        "chapter_count": len(textbook.chapters),
        "total_pages": textbook.total_pages,
        "total_chars": textbook.total_chars,
    }

    chapter_count = len(textbook.chapters)
    empty_chapter_count = sum(1 for chapter in textbook.chapters if chapter.char_count == 0)
    short_chapter_count = sum(
        1 for chapter in textbook.chapters if 0 < chapter.char_count < 80
    )
    page_span_inversions = sum(
        1
        for chapter in textbook.chapters
        if chapter.page_start is not None
        and chapter.page_end is not None
        and chapter.page_start > chapter.page_end
    )
    duplicate_title_count = _duplicate_count(
        chapter.title for chapter in textbook.chapters if chapter.title
    )

    metrics.update(
        {
            "empty_chapter_count": empty_chapter_count,
            "short_chapter_count": short_chapter_count,
            "duplicate_title_count": duplicate_title_count,
            "page_span_inversions": page_span_inversions,
            "avg_chars_per_chapter": (
                round(textbook.total_chars / chapter_count, 2) if chapter_count else 0
            ),
        }
    )

    score = 1.0
    if chapter_count == 0:
        score -= 0.45
        warnings.append("未识别出章节，图谱构建只能按整书内容兜底")
    if textbook.total_chars == 0:
        score -= 0.45
        warnings.append("教材正文为空，无法用于 RAG 检索")
    if empty_chapter_count:
        score -= min(0.2, empty_chapter_count * 0.04)
        warnings.append(f"存在 {empty_chapter_count} 个空章节")
    if chapter_count and short_chapter_count / chapter_count > 0.4:
        score -= 0.12
        warnings.append("短章节占比较高，可能存在目录或页眉误识别")
    if duplicate_title_count:
        score -= min(0.12, duplicate_title_count * 0.03)
        warnings.append(f"存在 {duplicate_title_count} 个重复章节标题")
    if page_span_inversions:
        score -= 0.18
        warnings.append("存在章节页码范围倒置")
    if textbook.total_pages is not None and textbook.total_pages <= 0:
        score -= 0.08
        warnings.append("页数信息异常")

    score = max(0.0, round(score, 2))
    return ParserQualityReport(
        score=score,
        level=_quality_level(score),
        warnings=warnings,
        metrics=metrics,
    )


def _duplicate_count(values: Any) -> int:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split()).lower()
        if not normalized:
            continue
        if normalized in seen:
            duplicates.add(normalized)
        seen.add(normalized)
    return len(duplicates)


def _quality_level(score: float) -> str:
    if score >= 0.85:
        return "good"
    if score >= 0.6:
        return "warning"
    return "poor"
