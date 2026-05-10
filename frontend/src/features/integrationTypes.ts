export type DecisionType = "merge" | "keep" | "remove";
export type FeedbackAction = "keep" | "remove" | "split" | "merge" | "unknown";

export type KnowledgeNode = {
  node_id: string;
  name: string;
  definition: string | null;
  aliases: string[];
  source_textbook_id: string | null;
  chapter_id: string | null;
  node_type: string;
  metadata: Record<string, unknown>;
};

export type KnowledgeEdge = {
  edge_id: string;
  source: string;
  target: string;
  relation: string;
  weight: number;
  source_textbook_id: string | null;
  evidence: string[];
  metadata: Record<string, unknown>;
};

export type TextbookKnowledgeGraph = {
  graph_id: string;
  textbook_id: string;
  textbook_title: string;
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
};

export type SimilarityBreakdown = {
  normalized_name: number;
  token_jaccard: number;
  definition_jaccard: number;
  embedding_similarity: number;
  final_score: number;
};

export type IntegrationDecision = {
  decision_id: string;
  action: DecisionType;
  affected_nodes: string[];
  result_node: string | null;
  decision: DecisionType;
  node_ids: string[];
  canonical_name: string;
  score: number;
  confidence: number;
  reason: string;
  similarity: SimilarityBreakdown | null;
};

export type CompressionStats = {
  source_graph_count: number;
  source_node_count: number;
  source_edge_count: number;
  merged_node_count: number;
  merged_edge_count: number;
  removed_node_count: number;
  merge_group_count: number;
  compression_ratio: number;
  node_reduction_ratio: number;
  source_char_count: number;
  merged_char_count: number;
  char_compression_ratio: number;
  char_budget: number;
  budget_met: boolean;
};

export type IntegratedKnowledgeGraph = {
  graph_id: string;
  textbook_ids: string[];
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
};

export type GraphIntegrationResult = {
  decisions: IntegrationDecision[];
  merged_graph: IntegratedKnowledgeGraph;
  stats: CompressionStats;
};

export type TeacherFeedbackChange = {
  action: FeedbackAction;
  target: string;
  affected_node_ids: string[];
  note: string;
};

export type TeacherFeedbackResult = {
  result: GraphIntegrationResult;
  changes: TeacherFeedbackChange[];
  unapplied: string[];
};
