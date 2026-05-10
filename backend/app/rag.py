from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import chromadb
from fastapi import APIRouter, BackgroundTasks, HTTPException

from .modelscope_client import ModelScopeError, chat_completion, create_embeddings, default_embedding_model
from .rag_schemas import (
    RagChapterInput,
    RagChunkMetadata,
    RagCitation,
    RagIndexJobStartResponse,
    RagIndexRequest,
    RagIndexResponse,
    RagIndexStatusResponse,
    RagQueryRequest,
    RagQueryResponse,
    RagRetrievalInfo,
    RagSourceChunk,
    RagTextbookInput,
)

NO_ANSWER = "当前知识库中未找到相关信息"
CHUNK_SIZE = 650
CHUNK_OVERLAP = 80
TOP_K = 5
CHUNKING_STRATEGY = "sliding_window_500_800_overlap_50_100"
RETRIEVAL_MODEL = "chroma_vector_rerank_v2"
CHROMA_DIR = Path.cwd() / "data" / "chroma"
SENTENCE_BOUNDARY = "。！？!?；;\n"
TERM_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")
LOW_SIGNAL_CHAPTER_RE = re.compile(
    r"(前言|序|目录|版权|编委|主编|副主编|简介|封面|扉页|致谢|附录|索引|参考文献|数字版)",
    re.IGNORECASE,
)

router = APIRouter(prefix="/api/rag", tags=["rag"])
INDEX_JOBS: dict[str, dict[str, object]] = {}
INDEX_COLLECTIONS: dict[str, str] = {}


@router.post("/index", response_model=RagIndexResponse)
def create_rag_index(request: RagIndexRequest) -> RagIndexResponse:
    return index_textbooks(request)


@router.post("/index/start", response_model=RagIndexJobStartResponse)
def start_rag_index(request: RagIndexRequest, background_tasks: BackgroundTasks) -> RagIndexJobStartResponse:
    job_id = f"rag_job_{uuid.uuid4().hex[:12]}"
    estimated = estimate_chunk_count(request.textbooks, request.chunk_size, request.chunk_overlap)
    INDEX_JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "index_id": request.index_id,
        "textbook_count": len(request.textbooks),
        "chapter_count": sum(len(textbook.chapters) for textbook in request.textbooks),
        "chunk_count": 0,
        "embedding_count": 0,
        "estimated_chunk_count": estimated,
        "vector_store_type": "chroma",
        "error": None,
        "result": None,
    }
    background_tasks.add_task(run_index_job, job_id, request)
    return RagIndexJobStartResponse(
        job_id=job_id,
        status="queued",
        index_id=request.index_id,
        estimated_chunk_count=estimated,
    )


@router.get("/index/status/{job_id}", response_model=RagIndexStatusResponse)
def get_rag_index_status(job_id: str) -> RagIndexStatusResponse:
    job = INDEX_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"RAG 索引任务不存在：{job_id}")
    return RagIndexStatusResponse(**job)


@router.post("/query", response_model=RagQueryResponse)
def query_rag(request: RagQueryRequest) -> RagQueryResponse:
    if request.textbooks is not None:
        index_textbooks(
            RagIndexRequest(
                textbooks=request.textbooks,
                index_id=request.index_id,
                embedding_model=request.embedding_model,
            )
        )

    collection_name = INDEX_COLLECTIONS.get(request.index_id) or collection_name_for_index(request.index_id)
    collection = get_chroma_client().get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail=f"RAG 索引不存在或为空：{request.index_id}")

    embedding_model = request.embedding_model or default_embedding_model()
    try:
        query_embedding = create_embeddings([request.query], model=embedding_model)[0]
    except (ModelScopeError, IndexError) as exc:
        raise HTTPException(status_code=502, detail=f"问题向量化失败：{exc}") from exc

    top_k = min(request.top_k or TOP_K, TOP_K)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    matches = chroma_results_to_matches(results)

    if not matches:
        return RagQueryResponse(
            answer=NO_ANSWER,
            citations=[],
            source_chunks=[],
            retrieval=retrieval_info(request, embedding_model, collection.count(), 0),
        )

    citations = [
        RagCitation(
            citation_id=citation_label(match["metadata"]),
            chunk_id=str(match["id"]),
            textbook_id=str(match["metadata"].get("textbook_id", "")),
            textbook_title=str(match["metadata"].get("textbook_title", "")),
            filename=str(match["metadata"].get("filename", "")),
            chapter_id=str(match["metadata"].get("chapter_id", "")),
            chapter_title=str(match["metadata"].get("chapter_title", "")),
            page_start=optional_int(match["metadata"].get("page_start")),
            page_end=optional_int(match["metadata"].get("page_end")),
            score=float(match["score"]),
        )
        for match in matches
    ]
    source_chunks = [
        RagSourceChunk(
            chunk_id=str(match["id"]),
            text=str(match["document"]),
            score=round(float(match["score"]), 4),
            metadata=metadata_to_schema(match["metadata"], embedding_model),
        )
        for match in matches
    ]

    answer = generate_answer(request.query, matches, citations, request.answer_mode)
    return RagQueryResponse(
        answer=answer,
        citations=citations,
        source_chunks=source_chunks,
        retrieval=retrieval_info(request, embedding_model, collection.count(), len(matches)),
    )


def run_index_job(job_id: str, request: RagIndexRequest) -> None:
    update_job(job_id, status="running")

    def progress(**updates: object) -> None:
        update_job(job_id, **updates)

    try:
        result = index_textbooks(request, progress)
        update_job(
            job_id,
            status="completed",
            chunk_count=result.chunk_count,
            embedding_count=result.embedding_count,
            vector_store_type=result.vector_store_type,
            result=result,
        )
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc))


def update_job(job_id: str, **updates: object) -> None:
    job = INDEX_JOBS.get(job_id)
    if job:
        job.update(updates)


def index_textbooks(request: RagIndexRequest, progress: Optional[Callable[..., None]] = None) -> RagIndexResponse:
    embedding_model = request.embedding_model or default_embedding_model()
    chunk_size = request.chunk_size or CHUNK_SIZE
    chunk_overlap = request.chunk_overlap or CHUNK_OVERLAP
    chunks = build_chunks(request.textbooks, chunk_size, chunk_overlap, embedding_model)
    if progress:
        progress(chunk_count=len(chunks), vector_store_type="chroma")

    collection_name = collection_name_for_index(request.index_id)
    collection = recreate_collection(collection_name)

    embedding_count = 0
    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = create_embeddings([chunk["text"] for chunk in batch], model=embedding_model)
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=embeddings,
            metadatas=[chunk["metadata"] for chunk in batch],
        )
        embedding_count += len(embeddings)
        if progress:
            progress(embedding_count=embedding_count)

    INDEX_COLLECTIONS[request.index_id] = collection_name
    return RagIndexResponse(
        index_id=request.index_id,
        textbook_count=len(request.textbooks),
        chapter_count=sum(len(textbook.chapters) for textbook in request.textbooks),
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        embedding_count=embedding_count,
        vector_store_type="chroma",
        chunking_strategy=CHUNKING_STRATEGY,
    )


def build_chunks(
    textbooks: list[RagTextbookInput],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for textbook in textbooks:
        for chapter in textbook.chapters:
            content = normalize_content(chapter.content)
            if not content:
                continue
            for chunk_index, (text, char_start, char_end) in enumerate(
                split_text(content, chunk_size, chunk_overlap),
                start=1,
            ):
                chunks.append(
                    {
                        "id": f"{textbook.textbook_id}:{chapter.chapter_id}:chunk_{chunk_index:04d}",
                        "text": text,
                        "metadata": sanitize_metadata(
                            {
                                "textbook_id": textbook.textbook_id,
                                "textbook_title": textbook.title,
                                "filename": textbook.filename,
                                "chapter_id": chapter.chapter_id,
                                "chapter_title": chapter.title,
                                "page_start": chapter.page_start,
                                "page_end": chapter.page_end,
                                "chunk_index": chunk_index,
                                "char_start": char_start,
                                "char_end": char_end,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "embedding_model": embedding_model,
                                "retrieval_model": RETRIEVAL_MODEL,
                            }
                        ),
                    }
                )
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    chunks: list[tuple[str, int, int]] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        if text_len - start <= chunk_size:
            chunk = text[start:text_len].strip()
            if chunk:
                chunks.append((chunk, start, text_len))
            break
        preferred_end = min(start + chunk_size, text_len)
        end = nearest_boundary(text, start, preferred_end, 500)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        start = max(end - chunk_overlap, start + 1)
    return chunks


def nearest_boundary(text: str, start: int, preferred_end: int, min_size: int) -> int:
    lower = min(start + min_size, preferred_end)
    window = text[lower:preferred_end]
    for index in range(len(window) - 1, -1, -1):
        if window[index] in SENTENCE_BOUNDARY:
            return lower + index + 1
    return preferred_end


def estimate_chunk_count(textbooks: list[RagTextbookInput], chunk_size: int, chunk_overlap: int) -> int:
    step = max(1, chunk_size - chunk_overlap)
    count = 0
    for textbook in textbooks:
        for chapter in textbook.chapters:
            text_len = len(normalize_content(chapter.content))
            if text_len <= 0:
                continue
            count += max(1, math.ceil((text_len - chunk_overlap) / step))
    return count


def generate_answer(
    question: str,
    matches: list[dict[str, object]],
    citations: list[RagCitation],
    answer_mode: str = "auto",
) -> str:
    if answer_mode == "extractive":
        return extractive_answer(matches, citations)
    context = "\n\n".join(
        f"{citations[index].citation_id}\n{match['document']}"
        for index, match in enumerate(matches)
    )
    prompt = f"""
你是教材 RAG 问答助手。请严格遵守：
1. 只基于提供的 top5 文本切片回答，不要使用切片外知识。
2. 每个关键结论后必须附来源引用，引用格式使用上下文中的方括号。
3. 如果上下文中找不到答案，只回答：{NO_ANSWER}
4. 回答要直接、准确、适合医学教材学习。

用户原始问题：
{question}

top5 文本切片：
{context}
""".strip()
    try:
        answer = chat_completion(
            [
                {"role": "system", "content": "你只能基于给定教材切片回答，并必须保留来源引用。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        return answer.strip() or NO_ANSWER
    except ModelScopeError:
        return extractive_answer(matches, citations)


def extractive_answer(matches: list[dict[str, object]], citations: list[RagCitation]) -> str:
    if not matches:
        return NO_ANSWER
    first = str(matches[0]["document"])
    sentence = re.split(r"(?<=[。！？!?；;])", first)[0].strip() or first[:220]
    return f"{sentence} {citations[0].citation_id}".strip()


def retrieve_matches(
    collection,
    request: RagQueryRequest,
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, object]]:
    candidate_limit = max(top_k * 4, 20)

    if request.retrieval_mode in {"hybrid", "vector"}:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_limit,
            include=["documents", "metadatas", "distances"],
        )
        matches = chroma_results_to_matches(results)
        reranked = rerank_matches(request.query, matches, request.retrieval_mode)
        return [match for match in reranked if float(match["score"]) >= request.min_score][:top_k]

    records = collection.get(include=["documents", "metadatas"])
    ids = records.get("ids") or []
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    matches: list[dict[str, object]] = []
    for index, chunk_id in enumerate(ids):
        if index >= len(documents) or index >= len(metadatas):
            continue
        score = lexical_match_score(request.query, documents[index], metadatas[index], request.retrieval_mode)
        if score <= 0:
            continue
        matches.append(
            {
                "id": chunk_id,
                "document": documents[index],
                "metadata": metadatas[index],
                "score": score,
            }
        )
    matches.sort(key=lambda item: float(item["score"]), reverse=True)
    return [match for match in matches if float(match["score"]) >= request.min_score][:top_k]


def rerank_matches(query: str, matches: list[dict[str, object]], retrieval_mode: str) -> list[dict[str, object]]:
    reranked: list[dict[str, object]] = []
    for match in matches:
        reranked.append(
            {
                **match,
                "score": lexical_match_score(
                    query,
                    str(match["document"]),
                    dict(match["metadata"]),
                    retrieval_mode,
                    vector_score=float(match["score"]),
                ),
            }
        )
    reranked.sort(key=lambda item: float(item["score"]), reverse=True)
    return reranked


def lexical_match_score(
    query: str,
    document: str,
    metadata: dict[str, object],
    retrieval_mode: str,
    *,
    vector_score: float = 0.0,
) -> float:
    query_terms = term_counter(query)
    query_ngrams = char_ngram_counter(query)
    doc_terms = term_counter(document)
    doc_ngrams = char_ngram_counter(document)
    title_terms = term_counter(str(metadata.get("chapter_title", "")))
    keyword_score = weighted_jaccard(query_terms, doc_terms)
    ngram_score = cosine_similarity(query_ngrams, doc_ngrams)
    title_score = weighted_jaccard(query_terms, title_terms)
    substring_score = 0.15 if normalize_query(query) and normalize_query(query) in normalize_query(document) else 0.0
    low_signal_penalty = 0.12 if is_low_signal_metadata(metadata, document) else 0.0

    if retrieval_mode == "keyword":
        score = keyword_score + title_score * 0.18 + substring_score - low_signal_penalty
    elif retrieval_mode == "char_ngram":
        score = ngram_score + title_score * 0.12 + substring_score - low_signal_penalty
    elif retrieval_mode == "vector":
        score = vector_score + title_score * 0.06 + substring_score * 0.3 - low_signal_penalty
    else:
        score = (
            vector_score * 0.45
            + keyword_score * 0.22
            + ngram_score * 0.18
            + title_score * 0.15
            + substring_score
            - low_signal_penalty
        )
    return clamp_score(score)


def chroma_results_to_matches(results: dict[str, object]) -> list[dict[str, object]]:
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    matches: list[dict[str, object]] = []
    for index, chunk_id in enumerate(ids):
        distance = float(distances[index]) if index < len(distances) else 1.0
        score = max(0.0, 1.0 - distance)
        matches.append(
            {
                "id": chunk_id,
                "document": documents[index],
                "metadata": metadatas[index],
                "score": score,
            }
        )
    return sorted(matches, key=lambda item: float(item["score"]), reverse=True)


def retrieval_info(
    request: RagQueryRequest,
    embedding_model: str,
    embedding_count: int,
    matched_chunks: int,
) -> RagRetrievalInfo:
    return RagRetrievalInfo(
        index_id=request.index_id,
        retrieval_mode=request.retrieval_mode,
        retrieval_model=RETRIEVAL_MODEL,
        embedding_model=embedding_model,
        embedding_count=embedding_count,
        vector_store_type="chroma",
        chunking_strategy=CHUNKING_STRATEGY,
        top_k=min(request.top_k or TOP_K, TOP_K),
        min_score=request.min_score,
        matched_chunks=matched_chunks,
    )


def metadata_to_schema(metadata: dict[str, object], embedding_model: str) -> RagChunkMetadata:
    return RagChunkMetadata(
        textbook_id=str(metadata.get("textbook_id", "")),
        textbook_title=str(metadata.get("textbook_title", "")),
        filename=str(metadata.get("filename", "")),
        chapter_id=str(metadata.get("chapter_id", "")),
        chapter_title=str(metadata.get("chapter_title", "")),
        page_start=optional_int(metadata.get("page_start")),
        page_end=optional_int(metadata.get("page_end")),
        chunk_index=int(metadata.get("chunk_index") or 0),
        char_start=int(metadata.get("char_start") or 0),
        char_end=int(metadata.get("char_end") or 0),
        retrieval_model=RETRIEVAL_MODEL,
        embedding_model=embedding_model,
        chunk_size=int(metadata.get("chunk_size") or CHUNK_SIZE),
        chunk_overlap=int(metadata.get("chunk_overlap") or CHUNK_OVERLAP),
    )


def citation_label(metadata: dict[str, object]) -> str:
    page_start = optional_int(metadata.get("page_start"))
    page_end = optional_int(metadata.get("page_end"))
    if page_start and page_end and page_end != page_start:
        page = f"第 {page_start}-{page_end} 页"
    elif page_start:
        page = f"第 {page_start} 页"
    else:
        page = "页码未知"
    return f"[{metadata.get('textbook_title', '')}, {metadata.get('chapter_title', '')}, {page}]"


def recreate_collection(name: str):
    client = get_chroma_client()
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def get_chroma_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def collection_name_for_index(index_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", index_id).strip("_") or "default"
    return f"rag_{safe}"[:60]


def sanitize_metadata(metadata: dict[str, object]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def optional_int(value: object) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_low_signal_metadata(metadata: dict[str, object], document: str) -> bool:
    title = str(metadata.get("chapter_title", "")).strip()
    if LOW_SIGNAL_CHAPTER_RE.search(title):
        return True
    page_start = optional_int(metadata.get("page_start")) or 0
    return page_start <= 15 and len(document.strip()) < 320


def term_counter(text: str) -> Counter[str]:
    terms: list[str] = []
    for token in TERM_RE.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            terms.extend(chinese_terms(token))
        elif len(token) > 1:
            terms.append(token)
    return Counter(terms)


def chinese_terms(token: str) -> list[str]:
    if len(token) <= 2:
        return [token]
    terms = [token]
    for size in (2, 3, 4):
        terms.extend(token[index : index + size] for index in range(0, len(token) - size + 1))
    return terms


def char_ngram_counter(text: str) -> Counter[str]:
    compact = normalize_query(text)
    grams: list[str] = []
    for size in (2, 3):
        if len(compact) >= size:
            grams.extend(compact[index : index + size] for index in range(0, len(compact) - size + 1))
    return Counter(grams)


def weighted_jaccard(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    overlap = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return overlap / union if union else 0.0


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", "", text).lower().strip()


def normalize_content(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def clear_chroma_store() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
