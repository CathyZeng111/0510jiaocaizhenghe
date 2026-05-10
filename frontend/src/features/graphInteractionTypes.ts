export type GraphNodeId = string;
export type TextbookSourceId = string;

export type TextbookSource = {
  id: TextbookSourceId;
  title: string;
  filename?: string;
  color?: string;
};

export type KnowledgeGraphNode = {
  id: GraphNodeId;
  label: string;
  frequency: number;
  sourceId?: TextbookSourceId;
  sourceTitle?: string;
  chapterTitle?: string;
  description?: string;
  sourceExcerpt?: string;
  keywords?: string[];
  x?: number;
  y?: number;
  metadata?: Record<string, unknown>;
};

export type KnowledgeGraphEdge = {
  id?: string;
  source: GraphNodeId;
  target: GraphNodeId;
  label?: string;
  weight?: number;
};

export type KnowledgeGraphData = {
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  sources?: TextbookSource[];
};

export type GraphViewport = {
  x: number;
  y: number;
  scale: number;
};
