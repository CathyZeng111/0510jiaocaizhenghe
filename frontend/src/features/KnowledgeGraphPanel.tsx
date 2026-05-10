import { useEffect, useMemo, useState } from "react";
import { Network, RefreshCw } from "lucide-react";
import { InteractiveGraph } from "./InteractiveGraph";
import type { KnowledgeGraphData } from "./graphInteractionTypes";
import type { Textbook } from "./ragTypes";
import type { TextbookKnowledgeGraph } from "./integrationTypes";

type Props = {
  textbooks: Textbook[];
  activeBookId: string | null;
  apiBase: string;
  onGraphReady?: (graphs: TextbookKnowledgeGraph[]) => void;
};

export function KnowledgeGraphPanel({ textbooks, activeBookId, apiBase, onGraphReady }: Props) {
  const [graphs, setGraphs] = useState<TextbookKnowledgeGraph[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [autoRequestedBookId, setAutoRequestedBookId] = useState<string | null>(null);
  const [expandedNodeId, setExpandedNodeId] = useState<string | null>(null);

  async function buildGraphs(textbookId: string) {
    const activeTextbook = textbooks.find((book) => book.textbook_id === textbookId) ?? textbooks[0];
    if (!activeTextbook) {
      setError("请先上传或加载一本教材。");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBase}/api/graphs/build`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ textbook_ids: [activeTextbook.textbook_id], use_llm: false }),
      });
      if (!response.ok) throw new Error(`图谱构建失败：HTTP ${response.status}`);
      const data = (await response.json()) as { graphs: TextbookKnowledgeGraph[] };
      setGraphs((current) => {
        const incomingIds = new Set(data.graphs.map((graph) => graph.textbook_id));
        const next = [...current.filter((graph) => !incomingIds.has(graph.textbook_id)), ...data.graphs];
        onGraphReady?.(next);
        return next;
      });
      setExpandedNodeId(null);
    } catch (currentError) {
      setError(currentError instanceof Error ? currentError.message : "图谱构建失败");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (textbooks.length === 0) {
      setGraphs([]);
      onGraphReady?.([]);
    }
  }, [onGraphReady, textbooks.length]);

  useEffect(() => {
    const activeTextbook = textbooks.find((book) => book.textbook_id === activeBookId) ?? textbooks[0];
    if (!activeTextbook || loading) return;
    if (graphs.some((graph) => graph.textbook_id === activeTextbook.textbook_id)) return;
    if (autoRequestedBookId === activeTextbook.textbook_id) return;
    setAutoRequestedBookId(activeTextbook.textbook_id);
    void buildGraphs(activeTextbook.textbook_id);
  }, [activeBookId, autoRequestedBookId, graphs, loading, textbooks]);

  const activeGraph = useMemo(() => {
    if (graphs.length === 0) return null;
    return graphs.find((graph) => graph.textbook_id === activeBookId) ?? graphs[0];
  }, [activeBookId, graphs]);

  const graphData = useMemo(
    () => (activeGraph ? toHierarchicalInteractiveGraph(activeGraph, expandedNodeId) : emptyGraph()),
    [activeGraph, expandedNodeId],
  );
  const primaryNodes = useMemo(
    () => activeGraph?.nodes.filter((node) => node.metadata?.level === "primary") ?? [],
    [activeGraph],
  );
  const expandedNode = activeGraph?.nodes.find((node) => node.node_id === expandedNodeId) ?? null;

  return (
    <section className="feature-panel">
      <div className="feature-header">
        <div>
          <p className="eyebrow">Knowledge Graph</p>
          <h2>单本教材知识图谱</h2>
        </div>
        {loading ? (
          <span className="auto-status">
            <RefreshCw size={16} className="spin-icon" />
            自动生成中
          </span>
        ) : null}
      </div>
      <div className="feature-metrics">
        <span>当前教材：{textbooks.find((book) => book.textbook_id === activeBookId)?.title ?? "未选择"}</span>
        <span>{graphs.length} 本已生成</span>
        <span>{primaryNodes.length} 个一级节点</span>
        <span>{expandedNode ? `已展开：${expandedNode.name}` : "未展开一级节点"}</span>
      </div>
      <p className="feature-note">
        进入本页后会自动基于已解析章节生成层级图谱；当前已关闭 ModelScope LLM 抽取，避免逐章调用导致等待过长。
      </p>
      {error ? <p className="error-banner">{error}</p> : null}
      {activeGraph ? (
        <>
          <div className="graph-level-list" aria-label="一级图谱节点">
            {primaryNodes.map((node) => (
              <button
                className={node.node_id === expandedNodeId ? "active" : ""}
                key={node.node_id}
                onClick={() => setExpandedNodeId(node.node_id === expandedNodeId ? null : node.node_id)}
                type="button"
              >
                {node.name}
              </button>
            ))}
          </div>
          <InteractiveGraph
            data={graphData}
            height="calc(100vh - 292px)"
            emptyMessage="当前没有可展示的层级图谱。"
            onNodeSelect={(node) => {
              if (node?.metadata?.level === "primary") {
                setExpandedNodeId(node.id === expandedNodeId ? null : node.id);
              }
            }}
          />
        </>
      ) : (
        <div className="empty-state compact">
          <Network size={36} />
          <h2>{loading ? "正在自动生成知识图谱" : "等待教材解析结果"}</h2>
          <p>系统会按一级章节优先展示，点击一级节点后展开二级内容块。</p>
        </div>
      )}
    </section>
  );
}

export function toHierarchicalInteractiveGraph(
  graph: TextbookKnowledgeGraph,
  expandedNodeId: string | null,
): KnowledgeGraphData {
  const primaryNodes = graph.nodes.filter((node) => node.metadata?.level === "primary");
  const visibleIds = new Set(primaryNodes.map((node) => node.node_id));

  if (expandedNodeId) {
    graph.nodes
      .filter((node) => node.metadata?.parent_id === expandedNodeId)
      .forEach((node) => visibleIds.add(node.node_id));
  }

  return toInteractiveGraph({
    ...graph,
    nodes: graph.nodes.filter((node) => visibleIds.has(node.node_id)),
    edges: graph.edges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)),
  });
}

export function toInteractiveGraph(graph: TextbookKnowledgeGraph): KnowledgeGraphData {
  return {
    sources: [{ id: graph.textbook_id, title: graph.textbook_title }],
    nodes: graph.nodes.map((node) => ({
      id: node.node_id,
      label: node.name,
      frequency: Number(node.metadata?.frequency ?? node.metadata?.source_count ?? 1),
      sourceId: node.source_textbook_id ?? graph.textbook_id,
      sourceTitle: graph.textbook_title,
      chapterTitle: String(node.metadata?.chapter_title ?? node.chapter_id ?? ""),
      description: node.definition ?? "",
      sourceExcerpt: String(node.metadata?.source_excerpt ?? ""),
      keywords: node.aliases,
      metadata: {
        type: node.node_type,
        page: node.metadata?.page_start as number | undefined,
        category: node.metadata?.category as string | undefined,
        level: node.metadata?.level as string | undefined,
        conflicts: node.metadata?.conflicts,
      },
    })),
    edges: graph.edges.map((edge) => ({
      id: edge.edge_id,
      source: edge.source,
      target: edge.target,
      label: edge.relation,
      weight: edge.weight,
    })),
  };
}

function emptyGraph(): KnowledgeGraphData {
  return { nodes: [], edges: [], sources: [] };
}
