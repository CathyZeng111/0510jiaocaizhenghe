from typing import Literal, Optional

from pydantic import BaseModel, Field


RelationType = Literal["prerequisite", "contains", "parallel", "applies_to"]
NodeKind = Literal["book", "chapter", "concept", "application"]


class KnowledgeGraphBuildOptions(BaseModel):
    max_concepts_per_chapter: int = Field(default=12, ge=1, le=40)
    min_concept_score: float = Field(default=1.5, ge=0)
    include_chapter_nodes: bool = True


class GraphEvidence(BaseModel):
    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None
    snippet: str


class KnowledgeNode(BaseModel):
    id: str
    label: str
    kind: NodeKind
    weight: float = 1.0
    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None
    source_chapter_ids: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    evidence: list[GraphEvidence] = Field(default_factory=list)


class KnowledgeRelation(BaseModel):
    id: str
    type: RelationType
    source: str
    target: str
    weight: float = 1.0
    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None
    evidence: list[GraphEvidence] = Field(default_factory=list)


class KnowledgeGraphStats(BaseModel):
    node_count: int
    relation_count: int
    concept_count: int
    chapter_count: int
    relation_type_counts: dict[RelationType, int]


class KnowledgeGraph(BaseModel):
    textbook_id: str
    textbook_title: str
    extraction_version: str
    nodes: list[KnowledgeNode]
    relations: list[KnowledgeRelation]
    stats: KnowledgeGraphStats
    build_options: KnowledgeGraphBuildOptions
    llm_enrichment_note: str


class KnowledgeGraphResponse(BaseModel):
    graph: KnowledgeGraph
