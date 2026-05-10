import { CSSProperties, FormEvent, useEffect, useMemo, useState } from "react";
import { BookOpenText, Database, Loader2, Search, Quote } from "lucide-react";
import {
  RagIndexJobStartResponse,
  RagIndexResponse,
  RagIndexStatusResponse,
  RagQueryResponse,
  Textbook,
} from "./ragTypes";

type RagPanelProps = {
  textbooks: Textbook[];
  indexId?: string;
  apiBase?: string;
};

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b";

export function RagPanel({ textbooks, indexId = "default", apiBase = DEFAULT_API_BASE }: RagPanelProps) {
  const [question, setQuestion] = useState("");
  const [indexSummary, setIndexSummary] = useState<RagIndexResponse | null>(null);
  const [indexProgress, setIndexProgress] = useState<RagIndexStatusResponse | null>(null);
  const [answer, setAnswer] = useState<RagQueryResponse | null>(null);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [indexedKey, setIndexedKey] = useState<string | null>(null);

  const availableChars = useMemo(
    () => textbooks.reduce((total, textbook) => total + textbook.total_chars, 0),
    [textbooks],
  );
  const canAsk = textbooks.length > 0 && question.trim().length > 0 && !isIndexing && !isAsking;
  const textbooksKey = useMemo(
    () => textbooks.map((textbook) => `${textbook.textbook_id}:${textbook.total_chars}:${textbook.chapters.length}`).join("|"),
    [textbooks],
  );

  useEffect(() => {
    if (!textbooks.length || isIndexing || indexedKey === textbooksKey) return;
    setIndexSummary(null);
    setAnswer(null);
    void buildIndex(textbooksKey);
  }, [indexedKey, isIndexing, textbooks.length, textbooksKey]);

  async function buildIndex(nextIndexedKey = textbooksKey): Promise<RagIndexResponse | null> {
    if (textbooks.length === 0) {
      setError("请先上传并解析教材。");
      return null;
    }

    setIsIndexing(true);
    setError(null);

    try {
      const response = await fetch(`${apiBase}/api/rag/index/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          index_id: indexId,
          textbooks,
          chunk_size: 800,
          chunk_overlap: 80,
          embedding_model: DEFAULT_EMBEDDING_MODEL,
        }),
      });

      if (!response.ok) {
        throw new Error(`启动索引失败：HTTP ${response.status}`);
      }

      const started = (await response.json()) as RagIndexJobStartResponse;
      setIndexProgress({
        job_id: started.job_id,
        status: "queued",
        index_id: started.index_id,
        textbook_count: textbooks.length,
        chapter_count: textbooks.reduce((total, textbook) => total + textbook.chapters.length, 0),
        chunk_count: 0,
        embedding_count: 0,
        estimated_chunk_count: started.estimated_chunk_count,
        vector_store_type: "none",
        error: null,
        result: null,
      });

      const data = await waitForIndexJob(started.job_id);
      setIndexSummary(data);
      setIndexedKey(nextIndexedKey);
      return data;
    } catch (currentError) {
      setError(currentError instanceof Error ? currentError.message : "索引失败");
      return null;
    } finally {
      setIsIndexing(false);
    }
  }

  async function waitForIndexJob(jobId: string): Promise<RagIndexResponse> {
    while (true) {
      await delay(1200);
      const response = await fetch(`${apiBase}/api/rag/index/status/${jobId}`);
      if (!response.ok) throw new Error(`索引状态获取失败：HTTP ${response.status}`);
      const status = (await response.json()) as RagIndexStatusResponse;
      setIndexProgress(status);
      if (status.status === "failed") throw new Error(status.error ?? "索引失败");
      if (status.status === "completed" && status.result) return status.result;
    }
  }

  async function askQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canAsk) return;

    setIsAsking(true);
    setError(null);

    try {
      const summary = indexSummary ?? (await buildIndex(textbooksKey));
      if (!summary) return;
      const response = await fetch(`${apiBase}/api/rag/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          index_id: indexId,
          query: question.trim(),
          top_k: 5,
          min_score: 0.08,
          retrieval_mode: "vector",
          embedding_model: summary.embedding_model ?? DEFAULT_EMBEDDING_MODEL,
        }),
      });

      if (!response.ok) {
        throw new Error(`问答失败：HTTP ${response.status}`);
      }

      const data = (await response.json()) as RagQueryResponse;
      setAnswer(data);
    } catch (currentError) {
      setError(currentError instanceof Error ? currentError.message : "问答失败");
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <section style={styles.panel} aria-label="RAG 精准问答">
      <header style={styles.header}>
        <div>
          <p style={styles.eyebrow}>RAG QA</p>
          <h2 style={styles.title}>精准问答</h2>
        </div>
        <div style={styles.indexButton}>
          {isIndexing ? <Loader2 size={16} style={styles.spinIcon} /> : <Database size={16} />}
          {isIndexing ? "自动索引中" : indexSummary ? "索引已完成" : "等待教材"}
        </div>
      </header>

      <div style={styles.summaryGrid}>
        <Metric label="教材" value={textbooks.length.toString()} />
        <Metric label="字符" value={availableChars.toLocaleString()} />
        <Metric label="分块" value={formatChunkMetric(indexSummary, indexProgress, isIndexing)} />
        <Metric label="向量块" value={formatEmbeddingMetric(indexSummary, indexProgress, isIndexing)} />
        <Metric label="向量库" value={indexSummary?.vector_store_type ?? (isIndexing ? "Chroma 中" : "待索引")} />
      </div>
      <p style={styles.note}>
        教材加载后会自动切片并写入 Chroma：每块 800 字，按句号/换行就近切分，保持 80 字重叠；同一批教材会复用已建索引。
      </p>

      <form onSubmit={askQuestion} style={styles.form}>
        <label htmlFor="rag-question" style={styles.label}>
          问题
        </label>
        <div style={styles.searchRow}>
          <input
            id="rag-question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="输入教材相关问题"
            style={styles.input}
          />
          <button type="submit" disabled={!canAsk} style={styles.askButton}>
            {isAsking ? <Loader2 size={16} style={styles.spinIcon} /> : <Search size={16} />}
            查询
          </button>
        </div>
      </form>

      {error ? <p style={styles.error}>{error}</p> : null}

      <div style={styles.answerBox}>
        {answer ? (
          <>
            <div style={styles.answerHeader}>
              <BookOpenText size={18} />
              <strong>回答</strong>
            </div>
            <p style={styles.answer}>{answer.answer}</p>
            {answer.citations.length > 0 ? (
              <div style={styles.citationList}>
                {answer.citations.map((citation) => (
                  <span key={citation.chunk_id} style={styles.citation}>
                    {citation.citation_id} {citation.textbook_title} / {citation.chapter_title}
                    {formatPages(citation.page_start, citation.page_end)} · {citation.score.toFixed(2)}
                  </span>
                ))}
              </div>
            ) : null}
            <p style={styles.retrievalMeta}>
              检索：{answer.retrieval.retrieval_model} · top-{answer.retrieval.top_k} ·{" "}
              {answer.retrieval.vector_store_type} · 命中 {answer.retrieval.matched_chunks} 块
            </p>
          </>
        ) : (
          <p style={styles.empty}>完成索引后即可对当前教材库提问。</p>
        )}
      </div>

      {answer?.source_chunks.length ? (
        <div style={styles.sources}>
          <div style={styles.answerHeader}>
            <Quote size={18} />
            <strong>来源片段</strong>
          </div>
          {answer.source_chunks.map((chunk) => (
            <article key={chunk.chunk_id} style={styles.sourceItem}>
              <div style={styles.sourceMeta}>
                <span>
                  {chunk.metadata.textbook_title} / {chunk.metadata.chapter_title}
                  {formatPages(chunk.metadata.page_start, chunk.metadata.page_end)}
                </span>
                <span>{chunk.score.toFixed(2)}</span>
              </div>
              <p style={styles.sourceText}>{chunk.text}</p>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function formatChunkMetric(
  summary: RagIndexResponse | null,
  progress: RagIndexStatusResponse | null,
  isIndexing: boolean,
) {
  if (summary) return summary.chunk_count.toLocaleString();
  if (isIndexing && progress) {
    if (progress.chunk_count > 0) return progress.chunk_count.toLocaleString();
    if (progress.estimated_chunk_count > 0) return `约 ${progress.estimated_chunk_count.toLocaleString()}`;
    return "计算中";
  }
  return "未建";
}

function formatEmbeddingMetric(
  summary: RagIndexResponse | null,
  progress: RagIndexStatusResponse | null,
  isIndexing: boolean,
) {
  if (summary) return summary.embedding_count.toLocaleString();
  if (isIndexing && progress) {
    const total = progress.chunk_count || progress.estimated_chunk_count;
    if (total > 0) return `${progress.embedding_count.toLocaleString()} / ${total.toLocaleString()}`;
    return "生成中";
  }
  return "未建";
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div style={styles.metric}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatPages(pageStart: number | null, pageEnd: number | null) {
  if (!pageStart) return "";
  if (!pageEnd || pageEnd === pageStart) return ` · 第 ${pageStart} 页`;
  return ` · 第 ${pageStart}-${pageEnd} 页`;
}

const styles: Record<string, CSSProperties> = {
  panel: {
    display: "flex",
    flexDirection: "column",
    gap: 16,
    border: "1px solid #d9e0ea",
    borderRadius: 8,
    background: "#ffffff",
    padding: 20,
    color: "#172033",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 14,
  },
  eyebrow: {
    margin: "0 0 6px",
    color: "#51617a",
    fontSize: 12,
    fontWeight: 700,
    textTransform: "uppercase",
  },
  title: {
    margin: 0,
    fontSize: 22,
    lineHeight: 1.2,
  },
  indexButton: {
    display: "inline-flex",
    minWidth: 98,
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    border: 0,
    borderRadius: 6,
    background: "#214e8a",
    color: "#ffffff",
    cursor: "pointer",
    fontWeight: 700,
    padding: "10px 12px",
  },
  summaryGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(5, minmax(0, 1fr))",
    gap: 8,
  },
  note: {
    margin: 0,
    color: "#607088",
    fontSize: 13,
    lineHeight: 1.55,
  },
  metric: {
    border: "1px solid #e1e7f0",
    borderRadius: 8,
    background: "#f7f9fc",
    padding: 10,
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  label: {
    color: "#51617a",
    fontSize: 13,
    fontWeight: 700,
  },
  searchRow: {
    display: "grid",
    gridTemplateColumns: "minmax(0, 1fr) auto",
    gap: 8,
  },
  input: {
    minWidth: 0,
    border: "1px solid #cfd8e6",
    borderRadius: 6,
    color: "#172033",
    font: "inherit",
    padding: "10px 12px",
  },
  askButton: {
    display: "inline-flex",
    minWidth: 86,
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    border: 0,
    borderRadius: 6,
    background: "#1c6dd0",
    color: "#ffffff",
    cursor: "pointer",
    fontWeight: 700,
    padding: "10px 12px",
  },
  error: {
    margin: 0,
    borderRadius: 6,
    background: "#ffeceb",
    color: "#b42318",
    padding: "10px 12px",
  },
  answerBox: {
    border: "1px solid #dce4ee",
    borderRadius: 8,
    background: "#fbfcfe",
    padding: 14,
  },
  answerHeader: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
  },
  answer: {
    margin: 0,
    color: "#223049",
    lineHeight: 1.7,
  },
  empty: {
    margin: 0,
    color: "#6a778c",
  },
  retrievalMeta: {
    margin: "12px 0 0",
    color: "#607088",
    fontSize: 12,
    fontWeight: 700,
  },
  citationList: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 12,
  },
  citation: {
    border: "1px solid #cfe0f5",
    borderRadius: 6,
    background: "#eef6ff",
    color: "#23476f",
    fontSize: 12,
    fontWeight: 700,
    padding: "6px 8px",
  },
  sources: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  sourceItem: {
    border: "1px solid #e1e7f0",
    borderRadius: 8,
    background: "#ffffff",
    padding: 12,
  },
  sourceMeta: {
    display: "flex",
    justifyContent: "space-between",
    gap: 10,
    color: "#607088",
    fontSize: 12,
    fontWeight: 700,
    marginBottom: 8,
  },
  sourceText: {
    display: "-webkit-box",
    margin: 0,
    overflow: "hidden",
    WebkitBoxOrient: "vertical",
    WebkitLineClamp: 4,
    color: "#263650",
    lineHeight: 1.65,
  },
  spinIcon: {
    animation: "spin 1s linear infinite",
  },
};
