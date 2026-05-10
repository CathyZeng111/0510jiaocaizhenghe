from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from .modelscope_client import (
    ModelScopeError,
    chat_completion,
    create_embeddings,
    extract_json_object,
    is_modelscope_configured,
)
from .parsers import SUPPORTED_EXTENSIONS, parse_file_path
from .rag import NO_ANSWER, index_textbooks, normalize_content, query_rag, split_text, term_counter
from .rag_eval_schemas import (
    EvalSourceReference,
    RagEvalAggregateMetrics,
    RagEvalCoverage,
    RagEvalDataset,
    RagEvalGenerateRequest,
    RagEvalItem,
    RagEvalOptimizeRequest,
    RagEvalOptimizeResponse,
    RagEvalQueryConfig,
    RagEvalQuestionResult,
    RagEvalRun,
    RagEvalRunRequest,
    RagEvalRunResponse,
    RagEvalTokenUsage,
)
from .rag_schemas import RagChapterInput, RagIndexRequest, RagQueryRequest, RagTextbookInput
from .schemas import Chapter, Textbook
from .textbook_store import textbook_store

router = APIRouter(prefix="/api/rag-eval", tags=["rag-eval"])

DATASETS: dict[str, RagEvalDataset] = {}
LOW_SIGNAL_TITLE_RE = re.compile(
    r"(前言|序|目录|版权|编委|主编|副主编|简介|封面|扉页|致谢|附录|索引|参考文献|英文版|数字版)",
    re.IGNORECASE,
)
ASCII_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


class CandidateSource:
    def __init__(
        self,
        *,
        source_id: str,
        textbook: Textbook,
        chapter: Chapter,
        excerpt: str,
    ) -> None:
        self.source_id = source_id
        self.textbook = textbook
        self.chapter = chapter
        self.excerpt = excerpt
        self.embedding: Optional[list[float]] = None

    @property
    def title_text(self) -> str:
        return f"{self.textbook.title} {self.chapter.title}".strip()

    def to_reference(self) -> EvalSourceReference:
        return EvalSourceReference(
            source_id=self.source_id,
            textbook_id=self.textbook.textbook_id,
            textbook_title=self.textbook.title,
            filename=self.textbook.filename,
            chapter_id=self.chapter.chapter_id,
            chapter_title=self.chapter.title,
            page_start=self.chapter.page_start,
            page_end=self.chapter.page_end,
            excerpt=self.excerpt[:320],
        )


@router.post("/generate", response_model=RagEvalDataset)
def generate_dataset(request: RagEvalGenerateRequest) -> RagEvalDataset:
    textbooks = ensure_textbooks(request.textbook_ids)
    candidate_sources = collect_candidate_sources(textbooks, request.question_count)
    if len(candidate_sources) < request.question_count:
        raise HTTPException(status_code=400, detail="可用于生成评测题的有效教材片段不足")

    dataset = build_eval_dataset(
        textbooks=textbooks,
        candidate_sources=candidate_sources,
        question_count=request.question_count,
        use_llm=request.use_llm,
        dataset_id=request.dataset_id,
    )
    DATASETS[dataset.dataset_id] = dataset
    persist_json(dataset, request.persist_path)
    return dataset


@router.post("/run", response_model=RagEvalRunResponse)
def run_evaluation(request: RagEvalRunRequest) -> RagEvalRunResponse:
    dataset = resolve_dataset(request.dataset_id, request.dataset)
    textbooks = ensure_textbooks(dataset.textbook_ids)
    runs = [evaluate_config(dataset, textbooks, config, judge_mode=request.judge_mode) for config in request.configs]
    best_run = max(runs, key=run_rank_key)
    response = RagEvalRunResponse(dataset_id=dataset.dataset_id, runs=runs, best_run_id=best_run.run_id)
    persist_json(response, request.persist_path)
    return response


@router.post("/optimize", response_model=RagEvalOptimizeResponse)
def optimize_evaluation(request: RagEvalOptimizeRequest) -> RagEvalOptimizeResponse:
    dataset = resolve_dataset(request.dataset_id, request.dataset)
    textbooks = ensure_textbooks(dataset.textbook_ids)
    configs = request.configs or default_search_configs()
    runs = [evaluate_config(dataset, textbooks, config, judge_mode=request.judge_mode) for config in configs]
    ordered = sorted(runs, key=run_rank_key, reverse=True)
    best_run = ordered[0]
    response = RagEvalOptimizeResponse(
        dataset_id=dataset.dataset_id,
        best_run_id=best_run.run_id,
        best_config=best_run.config,
        leaderboard=[run.metrics for run in ordered],
        runs=ordered,
    )
    persist_json(response, request.persist_path)
    return response


def ensure_textbooks(requested_ids: Optional[list[str]] = None) -> list[Textbook]:
    textbooks = textbook_store.list_textbooks()
    if not textbooks:
        load_local_textbooks()
        textbooks = textbook_store.list_textbooks()
    if not textbooks:
        raise HTTPException(status_code=404, detail="未找到可用教材，请先上传教材或准备 textbooks 目录")

    if not requested_ids:
        return textbooks

    mapping = {textbook.textbook_id: textbook for textbook in textbooks}
    selected = [mapping[textbook_id] for textbook_id in requested_ids if textbook_id in mapping]
    if len(selected) != len(requested_ids):
        missing = [textbook_id for textbook_id in requested_ids if textbook_id not in mapping]
        raise HTTPException(status_code=404, detail=f"未找到教材：{', '.join(missing)}")
    return selected


def load_local_textbooks() -> None:
    textbook_dir = Path.cwd() / "textbooks"
    if not textbook_dir.exists():
        return
    textbook_store.clear()
    for index, path in enumerate(sorted(textbook_dir.iterdir()), start=1):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        textbook = parse_file_path(path, path.name, f"book_{index:02d}")
        textbook_store.put_textbook(textbook)


def collect_candidate_sources(textbooks: list[Textbook], question_count: int) -> list[CandidateSource]:
    sources: list[CandidateSource] = []
    per_book_limit = max(6, math.ceil(question_count * 1.6 / max(len(textbooks), 1)))

    for textbook in textbooks:
        usable = [chapter for chapter in textbook.chapters if is_usable_chapter(chapter)]
        usable.sort(key=lambda chapter: ((chapter.page_start or 9999), -chapter.char_count))
        added = 0
        for chapter in usable:
            segments = split_text(normalize_content(chapter.content), chunk_size=560, chunk_overlap=80)
            for excerpt_index, (segment, _, _) in enumerate(segments[:2], start=1):
                if len(segment) < 220:
                    continue
                sources.append(
                    CandidateSource(
                        source_id=f"{textbook.textbook_id}:{chapter.chapter_id}:src_{excerpt_index:02d}",
                        textbook=textbook,
                        chapter=chapter,
                        excerpt=segment[:720],
                    )
                )
                added += 1
                if added >= per_book_limit:
                    break
            if added >= per_book_limit:
                break
    attach_source_embeddings(sources)
    return sources


def is_usable_chapter(chapter: Chapter) -> bool:
    page_start = chapter.page_start or 0
    title = (chapter.title or "").strip()
    if page_start < 20:
        return False
    if chapter.char_count < 1200:
        return False
    if not title or LOW_SIGNAL_TITLE_RE.search(title):
        return False
    return True


def attach_source_embeddings(sources: list[CandidateSource]) -> None:
    if not sources or not is_modelscope_configured():
        return
    texts = [f"{source.chapter.title}\n{source.excerpt[:300]}" for source in sources]
    batch_size = 24
    for start in range(0, len(sources), batch_size):
        batch = sources[start : start + batch_size]
        try:
            embeddings = create_embeddings(texts[start : start + batch_size])
        except ModelScopeError:
            return
        for source, embedding in zip(batch, embeddings):
            source.embedding = embedding


def build_eval_dataset(
    *,
    textbooks: list[Textbook],
    candidate_sources: list[CandidateSource],
    question_count: int,
    use_llm: bool,
    dataset_id: Optional[str],
) -> RagEvalDataset:
    dataset_id = dataset_id or f"rag_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    source_lookup = {source.source_id: source for source in candidate_sources}
    plans = build_question_plans(candidate_sources, question_count)
    items = generate_eval_items(plans, source_lookup, use_llm=use_llm)
    return RagEvalDataset(
        dataset_id=dataset_id,
        generated_at=datetime.now(timezone.utc),
        textbook_ids=[textbook.textbook_id for textbook in textbooks],
        llm_generated=bool(use_llm and is_modelscope_configured()),
        coverage=build_coverage(items),
        items=items,
    )


def build_question_plans(sources: list[CandidateSource], question_count: int) -> list[dict[str, object]]:
    targets = allocate_type_targets(question_count)
    plans: list[dict[str, object]] = []
    plan_index = 1

    for source in rotate_sources(sources, targets["factual"]):
        plans.append(make_single_plan(plan_index, "factual", source))
        plan_index += 1
    for source in rotate_sources(sources[1:] + sources[:1], targets["reasoning"]):
        plans.append(make_single_plan(plan_index, "reasoning", source))
        plan_index += 1

    pairs = build_cross_book_pairs(sources)
    for left, right in pairs[: targets["comparative"]]:
        plans.append(make_pair_plan(plan_index, "comparative", left, right))
        plan_index += 1
    for left, right in pairs[targets["comparative"] : targets["comparative"] + targets["cross_textbook"]]:
        plans.append(make_pair_plan(plan_index, "cross_textbook", left, right))
        plan_index += 1

    while len(plans) < question_count and sources:
        plans.append(make_single_plan(plan_index, "factual", sources[len(plans) % len(sources)]))
        plan_index += 1
    return plans[:question_count]


def allocate_type_targets(question_count: int) -> dict[str, int]:
    targets = {
        "factual": max(6, round(question_count * 0.35)),
        "reasoning": max(5, round(question_count * 0.25)),
        "comparative": max(4, round(question_count * 0.2)),
        "cross_textbook": max(4, question_count - max(6, round(question_count * 0.35)) - max(5, round(question_count * 0.25)) - max(4, round(question_count * 0.2))),
    }
    total = sum(targets.values())
    while total > question_count:
        for key in ("factual", "reasoning", "comparative", "cross_textbook"):
            if total <= question_count:
                break
            if targets[key] > 4:
                targets[key] -= 1
                total -= 1
    while total < question_count:
        targets["factual"] += 1
        total += 1
    return targets


def rotate_sources(sources: list[CandidateSource], limit: int) -> list[CandidateSource]:
    if not sources or limit <= 0:
        return []
    return [sources[index % len(sources)] for index in range(limit)]


def make_single_plan(index: int, question_type: str, source: CandidateSource) -> dict[str, object]:
    return {
        "plan_id": f"plan_{index:03d}",
        "question_type": question_type,
        "difficulty": "easy" if question_type == "factual" else "medium",
        "source_ids": [source.source_id],
    }


def make_pair_plan(index: int, question_type: str, left: CandidateSource, right: CandidateSource) -> dict[str, object]:
    return {
        "plan_id": f"plan_{index:03d}",
        "question_type": question_type,
        "difficulty": "hard" if question_type == "cross_textbook" else "medium",
        "source_ids": [left.source_id, right.source_id],
    }


def build_cross_book_pairs(sources: list[CandidateSource]) -> list[tuple[CandidateSource, CandidateSource]]:
    pairs: list[tuple[float, CandidateSource, CandidateSource]] = []
    for left_index, left in enumerate(sources):
        for right in sources[left_index + 1 :]:
            if left.textbook.textbook_id == right.textbook.textbook_id:
                continue
            score = pair_similarity(left, right)
            if score >= 0.16:
                pairs.append((score, left, right))
    pairs.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[CandidateSource, CandidateSource]] = []
    used_keys: set[tuple[str, str]] = set()
    for _, left, right in pairs:
        key = tuple(sorted([left.source_id, right.source_id]))
        if key in used_keys:
            continue
        used_keys.add(key)
        selected.append((left, right))
    return selected


def pair_similarity(left: CandidateSource, right: CandidateSource) -> float:
    title_overlap = jaccard(set(term_counter(left.title_text)), set(term_counter(right.title_text)))
    excerpt_overlap = jaccard(set(term_counter(left.excerpt[:220])), set(term_counter(right.excerpt[:220])))
    embedding_overlap = cosine(left.embedding, right.embedding)
    return round(max(title_overlap * 0.7 + excerpt_overlap * 0.3, embedding_overlap), 4)


def generate_eval_items(
    plans: list[dict[str, object]],
    source_lookup: dict[str, CandidateSource],
    *,
    use_llm: bool,
) -> list[RagEvalItem]:
    if use_llm and is_modelscope_configured():
        generated = generate_items_with_llm(plans, source_lookup)
        if generated:
            return generated
    return generate_items_with_templates(plans, source_lookup)


def generate_items_with_llm(
    plans: list[dict[str, object]],
    source_lookup: dict[str, CandidateSource],
) -> list[RagEvalItem]:
    items: list[RagEvalItem] = []
    batch_size = 4
    for start in range(0, len(plans), batch_size):
        batch = plans[start : start + batch_size]
        payload = []
        for plan in batch:
            current_sources = [source_lookup[source_id] for source_id in plan["source_ids"] if source_id in source_lookup]
            payload.append(
                {
                    "plan_id": plan["plan_id"],
                    "question_type": plan["question_type"],
                    "difficulty": plan["difficulty"],
                    "sources": [
                        {
                            "source_id": source.source_id,
                            "textbook_title": source.textbook.title,
                            "chapter_title": source.chapter.title,
                            "page_start": source.chapter.page_start,
                            "page_end": source.chapter.page_end,
                            "excerpt": source.excerpt[:520],
                        }
                        for source in current_sources
                    ],
                }
            )

        prompt = (
            "请基于给定教材片段生成 RAG 自动评测题，输出严格 JSON。"
            "每个题目必须只依赖给定片段，不得引入外部医学知识。"
            "问题类型必须与 plan 中 question_type 一致。"
            "expected_answer 用 1 到 3 句中文写出标准答案。"
            "source_ids 必须从给定片段里选择，并覆盖支撑答案所需的全部片段。"
            "comparative 或 cross_textbook 题目必须体现比较、联系或综合。"
            "输出格式：{\"items\": [{\"plan_id\": \"...\", \"question\": \"...\", \"difficulty\": \"easy|medium|hard\", \"expected_answer\": \"...\", \"source_ids\": [\"...\"], \"notes\": \"...\"}]}"
            f"\n\n计划数据：\n{json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            result = extract_json_object(
                chat_completion(
                    [
                        {"role": "system", "content": "你是严谨的医学教材 RAG 评测集构建助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1800,
                    response_format={"type": "json_object"},
                )
            )
        except (ModelScopeError, ValueError, TypeError, json.JSONDecodeError):
            return []

        for current in result.get("items", []):
            plan = next((item for item in batch if item["plan_id"] == current.get("plan_id")), None)
            if not plan:
                continue
            source_ids = [source_id for source_id in current.get("source_ids", []) if source_id in source_lookup]
            if not source_ids:
                source_ids = list(plan["source_ids"])
            items.append(
                RagEvalItem(
                    question_id=str(plan["plan_id"]).replace("plan", "q"),
                    question=str(current.get("question", "")).strip(),
                    question_type=plan["question_type"],
                    difficulty=current.get("difficulty", plan["difficulty"]),
                    expected_answer=str(current.get("expected_answer", "")).strip(),
                    expected_citations=[source_lookup[source_id].to_reference() for source_id in source_ids],
                    notes=current.get("notes"),
                )
            )
    return [item for item in items if item.question and item.expected_answer]


def generate_items_with_templates(
    plans: list[dict[str, object]],
    source_lookup: dict[str, CandidateSource],
) -> list[RagEvalItem]:
    items: list[RagEvalItem] = []
    for plan in plans:
        sources = [source_lookup[source_id] for source_id in plan["source_ids"] if source_id in source_lookup]
        if not sources:
            continue
        answer = " ".join(first_sentences(source.excerpt, limit=1) for source in sources).strip()
        if plan["question_type"] == "factual":
            question = f"根据《{sources[0].textbook.title}》中“{sources[0].chapter.title}”的内容，请概括该知识点的核心定义或要点。"
        elif plan["question_type"] == "reasoning":
            question = f"根据《{sources[0].textbook.title}》中“{sources[0].chapter.title}”的片段，概括其中描述的机制、原因或后果。"
        else:
            question = f"结合《{sources[0].textbook.title}》和《{sources[-1].textbook.title}》相关片段，比较两段内容的联系与差异。"
        items.append(
            RagEvalItem(
                question_id=str(plan["plan_id"]).replace("plan", "q"),
                question=question,
                question_type=plan["question_type"],
                difficulty=plan["difficulty"],
                expected_answer=answer[:220] or sources[0].excerpt[:180],
                expected_citations=[source.to_reference() for source in sources],
                notes="fallback-template",
            )
        )
    return items


def build_coverage(items: list[RagEvalItem]) -> RagEvalCoverage:
    by_type = Counter(item.question_type for item in items)
    by_difficulty = Counter(item.difficulty for item in items)
    textbook_coverage: defaultdict[str, int] = defaultdict(int)
    for item in items:
        for citation in item.expected_citations:
            textbook_coverage[citation.textbook_title] += 1
    return RagEvalCoverage(
        question_count=len(items),
        by_type=dict(by_type),
        by_difficulty=dict(by_difficulty),
        textbook_coverage=dict(textbook_coverage),
    )


def resolve_dataset(dataset_id: Optional[str], dataset: Optional[RagEvalDataset]) -> RagEvalDataset:
    if dataset is not None:
        return dataset
    if dataset_id and dataset_id in DATASETS:
        return DATASETS[dataset_id]
    raise HTTPException(status_code=404, detail="未找到评测集，请先生成评测集或直接传入 dataset")


def evaluate_config(
    dataset: RagEvalDataset,
    textbooks: list[Textbook],
    config: RagEvalQueryConfig,
    *,
    judge_mode: str,
) -> RagEvalRun:
    index_id = f"{config.name}_{uuid4().hex[:8]}"
    embedding_model = config.embedding_model if config.use_embeddings else ""
    textbook_inputs = to_rag_inputs(textbooks)
    index_response = index_textbooks(
        RagIndexRequest(
            textbooks=textbook_inputs,
            index_id=index_id,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_model=embedding_model or None,
        )
    )

    results: list[RagEvalQuestionResult] = []
    total_input_tokens = estimate_index_embedding_tokens(textbooks, config.chunk_size, config.chunk_overlap) if index_response.embedding_count else 0
    total_output_tokens = 0
    answer_uses_llm = config.answer_mode != "extractive"

    for item in dataset.items:
        start = perf_counter()
        response = query_rag(
            RagQueryRequest(
                query=item.question,
                index_id=index_id,
                top_k=config.top_k,
                min_score=config.min_score,
                retrieval_mode=config.retrieval_mode,
                answer_mode=config.answer_mode,
                embedding_model=embedding_model or None,
            )
        )
        response_time_ms = round((perf_counter() - start) * 1000, 2)
        answer_score, answer_reason = score_answer(item.expected_answer, response.answer, judge_mode)
        precision, recall, hit = score_citations(item.expected_citations, response.citations)
        token_usage = estimate_query_tokens(item.question, response.answer, response.source_chunks, answer_uses_llm)
        total_input_tokens += token_usage.estimated_input_tokens
        total_output_tokens += token_usage.estimated_output_tokens
        results.append(
            RagEvalQuestionResult(
                question_id=item.question_id,
                question=item.question,
                question_type=item.question_type,
                difficulty=item.difficulty,
                expected_answer=item.expected_answer,
                predicted_answer=response.answer,
                expected_source_ids=[citation.source_id for citation in item.expected_citations],
                predicted_source_ids=[citation.chunk_id for citation in response.citations],
                answer_score=round(answer_score, 4),
                answer_reason=answer_reason,
                citation_precision=round(precision, 4),
                citation_recall=round(recall, 4),
                citation_hit=hit,
                response_time_ms=response_time_ms,
                token_usage=token_usage,
            )
        )

    metrics = build_aggregate_metrics(results, total_input_tokens, total_output_tokens)
    return RagEvalRun(
        run_id=f"run_{config.name}_{uuid4().hex[:8]}",
        config=config,
        judge_mode=judge_mode,
        metrics=metrics,
        results=results,
    )


def to_rag_inputs(textbooks: list[Textbook]) -> list[RagTextbookInput]:
    return [
        RagTextbookInput(
            textbook_id=textbook.textbook_id,
            filename=textbook.filename,
            title=textbook.title,
            total_pages=textbook.total_pages,
            total_chars=textbook.total_chars,
            chapters=[
                RagChapterInput(
                    chapter_id=chapter.chapter_id,
                    title=chapter.title,
                    page_start=chapter.page_start,
                    page_end=chapter.page_end,
                    content=chapter.content,
                    char_count=chapter.char_count,
                )
                for chapter in textbook.chapters
            ],
        )
        for textbook in textbooks
    ]


def build_aggregate_metrics(
    results: list[RagEvalQuestionResult],
    total_input_tokens: int,
    total_output_tokens: int,
) -> RagEvalAggregateMetrics:
    question_count = len(results) or 1
    return RagEvalAggregateMetrics(
        question_count=len(results),
        answer_accuracy=round(sum(result.answer_score for result in results) / question_count, 4),
        citation_accuracy=round(sum(result.citation_precision for result in results) / question_count, 4),
        citation_recall=round(sum(result.citation_recall for result in results) / question_count, 4),
        exact_citation_hit_rate=round(sum(1 for result in results if result.citation_hit) / question_count, 4),
        avg_response_time_ms=round(sum(result.response_time_ms for result in results) / question_count, 2),
        token_usage=RagEvalTokenUsage(
            estimated_input_tokens=total_input_tokens,
            estimated_output_tokens=total_output_tokens,
            estimated_total_tokens=total_input_tokens + total_output_tokens,
        ),
    )


def score_answer(expected: str, predicted: str, judge_mode: str) -> tuple[float, str]:
    lexical_score = lexical_answer_score(expected, predicted)
    if judge_mode == "lexical" or not is_modelscope_configured():
        return lexical_score, "lexical"

    llm_score, llm_reason = llm_judge_answer(expected, predicted)
    if judge_mode == "llm":
        return llm_score, llm_reason
    return round((lexical_score + llm_score) / 2, 4), f"hybrid: lexical={lexical_score}, llm={llm_reason}"


def lexical_answer_score(expected: str, predicted: str) -> float:
    if predicted.strip() == NO_ANSWER:
        return 0.0
    expected_terms = term_counter(expected)
    predicted_terms = term_counter(predicted)
    if not expected_terms:
        return 0.0
    overlap = sum(min(expected_terms[key], predicted_terms[key]) for key in expected_terms)
    total_expected = sum(expected_terms.values()) or 1
    key_recall = overlap / total_expected
    return round(min(1.0, key_recall), 4)


def llm_judge_answer(expected: str, predicted: str) -> tuple[float, str]:
    prompt = (
        "请比较标准答案与模型答案的一致性，输出 JSON。"
        "score 为 0 到 1 之间的小数，只看事实是否被正确回答，不要求字面一致。"
        "输出格式：{\"score\": 0.0, \"reason\": \"...\"}"
        f"\n\n标准答案：{expected}\n模型答案：{predicted}"
    )
    try:
        result = extract_json_object(
            chat_completion(
                [
                    {"role": "system", "content": "你是严格的 RAG 评测裁判。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=220,
                response_format={"type": "json_object"},
            )
        )
        score = float(result.get("score", 0))
        return max(0.0, min(1.0, round(score, 4))), str(result.get("reason", "llm"))
    except (ModelScopeError, ValueError, TypeError, json.JSONDecodeError):
        return lexical_answer_score(expected, predicted), "llm-fallback-to-lexical"


def score_citations(expected: list[EvalSourceReference], predicted: list[object]) -> tuple[float, float, bool]:
    expected_keys = {citation_key(citation.textbook_id, citation.chapter_id, citation.page_start, citation.page_end) for citation in expected}
    predicted_keys = {
        citation_key(citation.textbook_id, citation.chapter_id, citation.page_start, citation.page_end) for citation in predicted
    }
    if not expected_keys or not predicted_keys:
        return 0.0, 0.0, False
    overlap = len(expected_keys & predicted_keys)
    precision = overlap / len(predicted_keys)
    recall = overlap / len(expected_keys)
    return precision, recall, overlap == len(expected_keys)


def citation_key(textbook_id: str, chapter_id: str, page_start: Optional[int], page_end: Optional[int]) -> str:
    return f"{textbook_id}:{chapter_id}:{page_start or 0}:{page_end or 0}"


def estimate_query_tokens(question: str, answer: str, source_chunks: list[object], used_llm_context: bool) -> RagEvalTokenUsage:
    context_text = ""
    if used_llm_context:
        context_text = "\n".join(getattr(chunk, "text", "")[:1200] for chunk in source_chunks)
    input_tokens = estimate_tokens(question) + estimate_tokens(context_text)
    output_tokens = estimate_tokens(answer)
    return RagEvalTokenUsage(
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_total_tokens=input_tokens + output_tokens,
    )


def estimate_index_embedding_tokens(textbooks: list[Textbook], chunk_size: int, chunk_overlap: int) -> int:
    total = 0
    for textbook in textbooks:
        for chapter in textbook.chapters:
            content = normalize_content(chapter.content)
            if not content:
                continue
            for chunk_text, _, _ in split_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                total += estimate_tokens(chunk_text)
    return total


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_words = ASCII_WORD_RE.findall(text)
    cjk_chars = len(CHINESE_CHAR_RE.findall(text))
    punctuation = max(0, len(text) - cjk_chars - sum(len(word) for word in ascii_words))
    return max(1, round(cjk_chars * 0.7 + len(ascii_words) * 1.1 + punctuation * 0.15))


def run_rank_key(run: RagEvalRun) -> tuple[float, float, float, float]:
    metrics = run.metrics
    composite = (
        metrics.answer_accuracy * 0.5
        + metrics.citation_accuracy * 0.2
        + metrics.citation_recall * 0.2
        + metrics.exact_citation_hit_rate * 0.1
    )
    return (
        round(composite, 6),
        metrics.answer_accuracy,
        metrics.citation_recall,
        -metrics.avg_response_time_ms,
    )


def default_search_configs() -> list[RagEvalQueryConfig]:
    return [
        RagEvalQueryConfig(name="hybrid_default", retrieval_mode="hybrid", chunk_size=650, chunk_overlap=80, top_k=5, min_score=0.08, use_embeddings=True),
        RagEvalQueryConfig(name="hybrid_tighter", retrieval_mode="hybrid", chunk_size=560, chunk_overlap=80, top_k=4, min_score=0.1, use_embeddings=True),
        RagEvalQueryConfig(name="hybrid_wider", retrieval_mode="hybrid", chunk_size=760, chunk_overlap=100, top_k=6, min_score=0.06, use_embeddings=True),
        RagEvalQueryConfig(name="keyword_only", retrieval_mode="keyword", chunk_size=650, chunk_overlap=80, top_k=5, min_score=0.08, use_embeddings=False, answer_mode="extractive"),
        RagEvalQueryConfig(name="char_ngram_only", retrieval_mode="char_ngram", chunk_size=650, chunk_overlap=80, top_k=5, min_score=0.08, use_embeddings=False, answer_mode="extractive"),
        RagEvalQueryConfig(name="vector_focus", retrieval_mode="vector", chunk_size=650, chunk_overlap=80, top_k=5, min_score=0.12, use_embeddings=True),
    ]


def persist_json(payload: object, output_path: Optional[str]) -> None:
    if not output_path:
        return
    path = Path(output_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump_json"):
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def first_sentences(text: str, limit: int = 2) -> str:
    sentences = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text)
    return " ".join(sentence.strip() for sentence in sentences[:limit]).strip()


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def cosine(left: Optional[list[float]], right: Optional[list[float]]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))
