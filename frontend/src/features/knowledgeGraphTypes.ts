export type KnowledgeRelationType = "prerequisite" | "contains" | "parallel" | "applies_to";
export type KnowledgeNodeKind = "book" | "chapter" | "concept" | "application";

export type GraphEvidence = {
  chapter_id: string | null;
  chapter_title: string | null;
  snippet: string;
};

export type KnowledgeNode = {
  id: string;
  label: string;
  kind: KnowledgeNodeKind;
  weight: number;
  chapter_id: string | null;
  chapter_title: string | null;
  source_chapter_ids: string[];
  aliases: string[];
  description: string | null;
  evidence: GraphEvidence[];
};

export type KnowledgeRelation = {
  id: string;
  type: KnowledgeRelationType;
  source: string;
  target: string;
  weight: number;
  chapter_id: string | null;
  chapter_title: string | null;
  evidence: GraphEvidence[];
};

export type KnowledgeGraphStats = {
  node_count: number;
  relation_count: number;
  concept_count: number;
  chapter_count: number;
  relation_type_counts: Record<KnowledgeRelationType, number>;
};

export type KnowledgeGraphBuildOptions = {
  max_concepts_per_chapter: number;
  min_concept_score: number;
  include_chapter_nodes: boolean;
};

export type KnowledgeGraph = {
  textbook_id: string;
  textbook_title: string;
  extraction_version: string;
  nodes: KnowledgeNode[];
  relations: KnowledgeRelation[];
  stats: KnowledgeGraphStats;
  build_options: KnowledgeGraphBuildOptions;
  llm_enrichment_note: string;
};

export type KnowledgeGraphResponse = {
  graph: KnowledgeGraph;
};
