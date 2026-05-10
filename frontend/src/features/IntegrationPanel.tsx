import { FormEvent, useEffect, useMemo, useState } from "react";
import { GitMerge, MessageSquare, RefreshCw, Send } from "lucide-react";
import { InteractiveGraph } from "./InteractiveGraph";
import { toInteractiveGraph } from "./KnowledgeGraphPanel";
import type {
  GraphIntegrationResult,
  IntegrationChatMessage,
  IntegrationChatResponse,
  TextbookKnowledgeGraph,
} from "./integrationTypes";
import type { Textbook } from "./ragTypes";

type Props = {
  graphs: TextbookKnowledgeGraph[];
  textbooks: Textbook[];
  apiBase: string;
};

export function IntegrationPanel({ graphs, textbooks, apiBase }: Props) {
  const [result, setResult] = useState<GraphIntegrationResult | null>(null);
  const [sourceGraphs, setSourceGraphs] = useState<TextbookKnowledgeGraph[]>(graphs);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState<IntegrationChatMessage[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const sessionId = "integration-default";

  useEffect(() => {
    if (graphs.length) setSourceGraphs(graphs);
  }, [graphs]);

  async function buildIntegration() {
    setLoading(true);
    setMessage(null);
    try {
      let graphsForIntegration = graphs;
      if (graphsForIntegration.length < 2 && textbooks.length >= 2) {
        const graphResponse = await fetch(`${apiBase}/api/graphs/build`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            textbook_ids: textbooks.map((textbook) => textbook.textbook_id),
            use_llm: false,
          }),
        });
        if (!graphResponse.ok) throw new Error(`自动生成单本图谱失败：HTTP ${graphResponse.status}`);
        const graphData = (await graphResponse.json()) as { graphs: TextbookKnowledgeGraph[] };
        graphsForIntegration = graphData.graphs;
      }
      setSourceGraphs(graphsForIntegration);
      const response = await fetch(`${apiBase}/api/integration/build`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ graphs: graphsForIntegration }),
      });
      if (!response.ok) throw new Error(`整合失败：HTTP ${response.status}`);
      setResult((await response.json()) as GraphIntegrationResult);
      setChatHistory([]);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "整合失败");
    } finally {
      setLoading(false);
    }
  }

  async function submitChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!chatInput.trim() || !result) return;
    const nextMessage = chatInput.trim();
    setLoading(true);
    setMessage(null);
    try {
      const response = await fetch(`${apiBase}/api/integration/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: nextMessage }),
      });
      if (!response.ok) throw new Error(`对话处理失败：HTTP ${response.status}`);
      const data = (await response.json()) as IntegrationChatResponse;
      setResult(data.result);
      setChatHistory(data.history);
      setMessage(data.changes.length ? `已更新 ${data.changes.length} 项整合结果` : null);
      setChatInput("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "对话处理失败");
    } finally {
      setLoading(false);
    }
  }

  const graphData = useMemo(() => {
    if (!result) return { nodes: [], edges: [], sources: [] };
    return toInteractiveGraph({
      graph_id: result.merged_graph.graph_id,
      textbook_id: "integrated",
      textbook_title: "整合图谱",
      nodes: result.merged_graph.nodes,
      edges: result.merged_graph.edges,
    });
  }, [result]);
  const sourceGraphData = useMemo(() => toSourceInteractiveGraph(sourceGraphs), [sourceGraphs]);
  const stats = result?.stats;

  return (
    <section className="feature-panel">
      <div className="feature-header">
        <div>
          <p className="eyebrow">Integration</p>
          <h2>跨教材知识整合</h2>
        </div>
        <button className="primary-action" type="button" onClick={() => void buildIntegration()} disabled={textbooks.length < 2 || loading}>
          <RefreshCw size={16} className={loading ? "spin-icon" : ""} />
          执行整合
        </button>
      </div>

      <div className="feature-metrics">
        <span>{Math.max(sourceGraphs.length, textbooks.length)} 个图谱</span>
        <span>{result?.stats.source_node_count ?? 0} 原节点</span>
        <span>{result?.stats.merged_node_count ?? 0} 整合后</span>
        <span>{result?.conflicts?.length ?? 0} 个定义冲突</span>
        <span>{formatNumber(stats?.source_char_count ?? 0)} → {formatNumber(stats?.merged_char_count ?? 0)} 字</span>
        <span className={stats?.budget_met ? "metric-good" : "metric-bad"}>
          字数压缩 {formatPercent(stats?.char_compression_ratio ?? 0)}
        </span>
      </div>
      {result ? (
        <div className="compression-banner">
          <strong>{stats?.budget_met ? "已满足 30% 压缩约束" : "未满足 30% 压缩约束"}</strong>
          <span>
            原始总字数 {formatNumber(stats?.source_char_count ?? 0)}，整合后 {formatNumber(stats?.merged_char_count ?? 0)}，
            预算上限 {formatNumber(stats?.char_budget ?? 0)}。
          </span>
        </div>
      ) : null}
      {message ? <p className="error-banner neutral">{message}</p> : null}

      {result ? (
        <div className="integration-layout">
          <div className="graph-compare">
            <div>
              <h3>整合前</h3>
              <InteractiveGraph data={sourceGraphData} height={360} />
            </div>
            <div>
              <h3>整合后</h3>
              <InteractiveGraph data={graphData} height={360} />
            </div>
          </div>
          <div className="decision-panel">
            <div className="chat-panel">
              <div className="chat-title">
                <MessageSquare size={16} />
                <strong>整合建议对话</strong>
              </div>
              <div className="chat-history">
                {chatHistory.length ? (
                  chatHistory.map((item) => (
                    <article key={item.message_id} className={`chat-message ${item.role}`}>
                      <span>{item.role === "teacher" ? "教师" : "系统"}</span>
                      <p>{item.content}</p>
                    </article>
                  ))
                ) : (
                  <div className="chat-empty">
                    可以追问原因，也可以直接修改方案，例如“为什么把炎症和炎症反应合并了？”或“把抗原和免疫原拆开”。
                  </div>
                )}
              </div>
              <form onSubmit={submitChat} className="chat-form">
                <textarea
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="输入整合追问或修改意见"
                />
                <button type="submit" disabled={loading || !chatInput.trim()}>
                  <Send size={16} />
                  发送
                </button>
              </form>
            </div>
            <h3>整合决策</h3>
            <div className="decision-list">
              {result.decisions.slice(0, 80).map((decision) => (
                <article key={decision.decision_id} className="decision-item">
                  <strong>{(decision.action ?? decision.decision).toUpperCase()} · {decision.canonical_name}</strong>
                  <span>{decision.reason}</span>
                  <small>
                    {decision.affected_nodes?.length ?? decision.node_ids.length} 节点 · 置信度{" "}
                    {(decision.confidence ?? decision.score).toFixed(2)}
                    {decision.similarity
                      ? ` · embedding ${decision.similarity.embedding_similarity.toFixed(2)}`
                      : ""}
                    {decision.result_node ? ` · 输出 ${decision.result_node}` : ""}
                  </small>
                </article>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="empty-state compact">
          <GitMerge size={36} />
          <h2>生成至少 2 本教材图谱后执行整合</h2>
          <p>系统会识别同义知识点，输出 merge / keep / remove 决策和压缩比。</p>
        </div>
      )}
    </section>
  );
}

function toSourceInteractiveGraph(graphs: TextbookKnowledgeGraph[]) {
  const nodes = graphs.flatMap((graph) =>
    graph.nodes.map((node) => ({
      ...node,
      node_id: `${graph.textbook_id}:${node.node_id}`,
      source_textbook_id: graph.textbook_id,
    })),
  );
  const edges = graphs.flatMap((graph) =>
    graph.edges.map((edge) => ({
      ...edge,
      edge_id: `${graph.textbook_id}:${edge.edge_id}`,
      source: `${graph.textbook_id}:${edge.source}`,
      target: `${graph.textbook_id}:${edge.target}`,
      source_textbook_id: graph.textbook_id,
    })),
  );
  return toInteractiveGraph({
    graph_id: "source_graphs",
    textbook_id: "source",
    textbook_title: "整合前图谱",
    nodes,
    edges,
  });
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value: number) {
  return value.toLocaleString();
}
