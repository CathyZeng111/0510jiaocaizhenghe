from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


QuestionType = Literal["factual", "comparative", "reasoning", "cross_textbook"]
DifficultyLevel = Literal["easy", "medium", "hard"]
JudgeMode = Literal["lexical", "llm", "hybrid"]


class EvalSourceReference(BaseModel):
    source_id: str
    textbook_id: str
    textbook_title: str
    filename: str
    chapter_id: str
    chapter_title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    excerpt: str


class RagEvalItem(BaseModel):
    question_id: str
    question: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    expected_answer: str
    expected_citations: list[EvalSourceReference]
    notes: Optional[str] = None


class RagEvalCoverage(BaseModel):
    question_count: int
    by_type: dict[str, int]
    by_difficulty: dict[str, int]
    textbook_coverage: dict[str, int]


class RagEvalDataset(BaseModel):
    dataset_id: str
    generated_at: datetime
    textbook_ids: list[str]
    llm_generated: bool = False
    coverage: RagEvalCoverage
    items: list[RagEvalItem]


class RagEvalGenerateRequest(BaseModel):
    textbook_ids: Optional[list[str]] = None
    question_count: int = Field(default=24, ge=20, le=50)
    use_llm: bool = True
    dataset_id: Optional[str] = None
    persist_path: Optional[str] = None


class RagEvalQueryConfig(BaseModel):
    name: str
    chunk_size: int = Field(default=650, ge=500, le=800)
    chunk_overlap: int = Field(default=80, ge=50, le=120)
    top_k: int = Field(default=5, ge=1, le=10)
    min_score: float = Field(default=0.08, ge=0, le=1)
    retrieval_mode: Literal["hybrid", "keyword", "char_ngram", "vector"] = "hybrid"
    answer_mode: Literal["auto", "extractive", "llm"] = "auto"
    use_embeddings: bool = True
    embedding_model: Optional[str] = None


class RagEvalTokenUsage(BaseModel):
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_total_tokens: int = 0


class RagEvalQuestionResult(BaseModel):
    question_id: str
    question: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    expected_answer: str
    predicted_answer: str
    expected_source_ids: list[str]
    predicted_source_ids: list[str]
    answer_score: float
    answer_reason: str
    citation_precision: float
    citation_recall: float
    citation_hit: bool
    response_time_ms: float
    token_usage: RagEvalTokenUsage


class RagEvalAggregateMetrics(BaseModel):
    question_count: int
    answer_accuracy: float
    citation_accuracy: float
    citation_recall: float
    exact_citation_hit_rate: float
    avg_response_time_ms: float
    token_usage: RagEvalTokenUsage


class RagEvalRun(BaseModel):
    run_id: str
    config: RagEvalQueryConfig
    judge_mode: JudgeMode
    metrics: RagEvalAggregateMetrics
    results: list[RagEvalQuestionResult]


class RagEvalRunRequest(BaseModel):
    dataset_id: Optional[str] = None
    dataset: Optional[RagEvalDataset] = None
    configs: list[RagEvalQueryConfig] = Field(default_factory=lambda: [RagEvalQueryConfig(name="baseline")])
    judge_mode: JudgeMode = "hybrid"
    persist_path: Optional[str] = None


class RagEvalRunResponse(BaseModel):
    dataset_id: str
    runs: list[RagEvalRun]
    best_run_id: str


class RagEvalOptimizeRequest(BaseModel):
    dataset_id: Optional[str] = None
    dataset: Optional[RagEvalDataset] = None
    configs: Optional[list[RagEvalQueryConfig]] = None
    judge_mode: JudgeMode = "lexical"
    persist_path: Optional[str] = None


class RagEvalOptimizeResponse(BaseModel):
    dataset_id: str
    best_run_id: str
    best_config: RagEvalQueryConfig
    leaderboard: list[RagEvalAggregateMetrics]
    runs: list[RagEvalRun]
