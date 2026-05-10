from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


DecisionType = Literal["merge", "keep", "remove"]
FeedbackAction = Literal["keep", "remove", "split", "merge", "unknown"]
ChatRole = Literal["teacher", "system"]


class KnowledgeNode(BaseModel):
    node_id: str
    name: str
    definition: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    source_textbook_id: Optional[str] = None
    chapter_id: Optional[str] = None
    node_type: str = "concept"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeEdge(BaseModel):
    edge_id: str
    source: str
    target: str
    relation: str = "related"
    weight: float = 1.0
    source_textbook_id: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextbookKnowledgeGraph(BaseModel):
    graph_id: str
    textbook_id: str
    textbook_title: str
    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)


class SimilarityBreakdown(BaseModel):
    normalized_name: float
    token_jaccard: float
    definition_jaccard: float
    embedding_similarity: float = 0.0
    final_score: float


class IntegrationDecision(BaseModel):
    decision_id: str
    action: DecisionType = "keep"
    affected_nodes: list[str] = Field(default_factory=list)
    result_node: Optional[str] = None
    decision: DecisionType
    node_ids: list[str]
    canonical_name: str
    score: float
    confidence: float = 0.0
    reason: str
    similarity: Optional[SimilarityBreakdown] = None


class ConflictDefinitionSource(BaseModel):
    node_id: str
    node_name: str
    textbook_id: str
    textbook_title: str
    definition: Optional[str] = None


class KnowledgeConflict(BaseModel):
    conflict_id: str
    normalized_name: str
    node_ids: list[str] = Field(default_factory=list)
    node_names: list[str] = Field(default_factory=list)
    textbook_titles: list[str] = Field(default_factory=list)
    token_jaccard: float
    definition_jaccard: float
    reason: str
    definitions: list[ConflictDefinitionSource] = Field(default_factory=list)


class CompressionStats(BaseModel):
    source_graph_count: int
    source_node_count: int
    source_edge_count: int
    merged_node_count: int
    merged_edge_count: int
    removed_node_count: int
    merge_group_count: int
    compression_ratio: float
    node_reduction_ratio: float
    source_char_count: int = 0
    merged_char_count: int = 0
    char_compression_ratio: float = 1.0
    char_budget: int = 0
    budget_met: bool = False


class IntegratedKnowledgeGraph(BaseModel):
    graph_id: str = "integrated_graph"
    textbook_ids: list[str] = Field(default_factory=list)
    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)


class GraphIntegrationResult(BaseModel):
    decisions: list[IntegrationDecision] = Field(default_factory=list)
    merged_graph: IntegratedKnowledgeGraph
    stats: CompressionStats
    conflicts: list[KnowledgeConflict] = Field(default_factory=list)


class TeacherFeedbackChange(BaseModel):
    action: FeedbackAction
    target: str
    affected_node_ids: list[str] = Field(default_factory=list)
    note: str


class TeacherFeedbackResult(BaseModel):
    result: GraphIntegrationResult
    changes: list[TeacherFeedbackChange] = Field(default_factory=list)
    unapplied: list[str] = Field(default_factory=list)


class IntegrationChatMessage(BaseModel):
    message_id: str
    role: ChatRole
    content: str
    created_at: str


class IntegrationChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class IntegrationChatResponse(BaseModel):
    session_id: str
    reply: str
    history: list[IntegrationChatMessage] = Field(default_factory=list)
    result: GraphIntegrationResult
    changes: list[TeacherFeedbackChange] = Field(default_factory=list)
    unapplied: list[str] = Field(default_factory=list)
