import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AlertCircle, CheckCircle2, FileText, FolderOpen, Loader2, UploadCloud } from "lucide-react";
import { IntegrationPanel } from "./features/IntegrationPanel";
import { KnowledgeGraphPanel } from "./features/KnowledgeGraphPanel";
import { RagPanel } from "./features/RagPanel";
import type { TextbookKnowledgeGraph } from "./features/integrationTypes";
import "./styles.css";

type ParseStatus = "parsing" | "completed" | "failed";

type Chapter = {
  chapter_id: string;
  title: string;
  page_start: number | null;
  page_end: number | null;
  content: string;
  char_count: number;
};

type Textbook = {
  textbook_id: string;
  filename: string;
  title: string;
  total_pages: number | null;
  total_chars: number;
  chapters: Chapter[];
};

type UploadResult = {
  filename: string;
  format: string;
  size: number;
  status: "completed" | "failed";
  error: string | null;
  textbook: Textbook | null;
};

type FileItem = {
  id: string;
  filename: string;
  format: string;
  size: number;
  status: ParseStatus;
  error?: string | null;
  textbook?: Textbook | null;
};

type ActiveTab = "parse" | "graph" | "integration" | "rag";

type ChapterGroup = {
  id: string;
  title: string;
  overview: Chapter | null;
  children: Chapter[];
  page_start: number | null;
  page_end: number | null;
  char_count: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

function App() {
  const [items, setItems] = useState<FileItem[]>([]);
  const [activeBookId, setActiveBookId] = useState<string | null>(null);
  const [activeChapterId, setActiveChapterId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>("parse");
  const [graphs, setGraphs] = useState<TextbookKnowledgeGraph[]>([]);
  const [isLoadingLocal, setIsLoadingLocal] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const parsedBooks = useMemo(
    () => items.map((item) => item.textbook).filter((book): book is Textbook => Boolean(book)),
    [items],
  );
  const activeBook = parsedBooks.find((book) => book.textbook_id === activeBookId) ?? parsedBooks[0] ?? null;
  const activeChapter =
    activeBook?.chapters.find((chapter) => chapter.chapter_id === activeChapterId) ??
    activeBook?.chapters[0] ??
    null;

  async function uploadFiles(fileList: FileList | File[]) {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    const pendingItems = files.map((file) => ({
      id: `${file.name}-${file.size}-${crypto.randomUUID()}`,
      filename: file.name,
      format: file.name.split(".").pop()?.toLowerCase() ?? "unknown",
      size: file.size,
      status: "parsing" as const,
      textbook: null,
      error: null,
    }));

    setItems((current) => [...pendingItems, ...current]);

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      const response = await fetch(`${API_BASE}/api/textbooks/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`上传失败：HTTP ${response.status}`);
      }

      const data: { results: UploadResult[] } = await response.json();
      setItems((current) =>
        current.map((item) => {
          const pendingIndex = pendingItems.findIndex((pending) => pending.id === item.id);
          if (pendingIndex === -1) return item;

          const result = data.results[pendingIndex];
          if (!result) return { ...item, status: "failed", error: "未收到解析结果" };

          return {
            ...item,
            filename: result.filename,
            format: result.format,
            size: result.size,
            status: result.status,
            error: result.error,
            textbook: result.textbook,
          };
        }),
      );

      const firstBook = data.results.find((result) => result.textbook)?.textbook;
      if (firstBook) {
        setActiveBookId(firstBook.textbook_id);
        setActiveChapterId(firstBook.chapters[0]?.chapter_id ?? null);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "上传失败";
      setItems((current) =>
        current.map((item) =>
          pendingItems.some((pending) => pending.id === item.id) ? { ...item, status: "failed", error: message } : item,
        ),
      );
    }
  }

  async function loadLocalTextbooks() {
    setIsLoadingLocal(true);
    try {
      const response = await fetch(`${API_BASE}/api/textbooks/load-local`, { method: "POST" });
      if (!response.ok) throw new Error(`本地教材加载失败：HTTP ${response.status}`);
      const data: { results: UploadResult[] } = await response.json();
      const nextItems = data.results.map((result) => ({
        id: `${result.filename}-${result.size}-${crypto.randomUUID()}`,
        filename: result.filename,
        format: result.format,
        size: result.size,
        status: result.status,
        error: result.error,
        textbook: result.textbook,
      }));
      setItems(nextItems);
      const firstBook = data.results.find((result) => result.textbook)?.textbook;
      if (firstBook) {
        setActiveBookId(firstBook.textbook_id);
        setActiveChapterId(firstBook.chapters[0]?.chapter_id ?? null);
      }
      setActiveTab("parse");
    } catch (error) {
      const message = error instanceof Error ? error.message : "本地教材加载失败";
      setItems((current) => [
        {
          id: `local-error-${crypto.randomUUID()}`,
          filename: "textbooks/",
          format: "folder",
          size: 0,
          status: "failed",
          error: message,
          textbook: null,
        },
        ...current,
      ]);
    } finally {
      setIsLoadingLocal(false);
    }
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) {
      void uploadFiles(event.target.files);
      event.target.value = "";
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    void uploadFiles(event.dataTransfer.files);
  }

  return (
    <main className="app-shell">
      <section className="panel sidebar">
        <div>
          <p className="eyebrow">Textbook Parser</p>
          <h1>多格式教材加载与解析</h1>
        </div>

        <div
          className={`dropzone ${isDragging ? "dragging" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
        >
          <UploadCloud size={34} />
          <strong>拖拽教材到这里</strong>
          <span>支持 PDF、Markdown、TXT、DOCX，可批量上传</span>
          <button type="button">选择文件</button>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".pdf,.md,.markdown,.txt,.docx"
            onChange={handleInputChange}
          />
        </div>

        <button className="local-load-button" type="button" onClick={() => void loadLocalTextbooks()} disabled={isLoadingLocal}>
          {isLoadingLocal ? <Loader2 size={16} className="spin-icon" /> : <FolderOpen size={16} />}
          加载当前 textbooks 文件夹
        </button>

        <div className="file-list">
          <h2>文件列表</h2>
          {items.length === 0 ? <p className="muted">还没有上传教材。</p> : null}
          {items.map((item) => (
            <button
              className={`file-row ${item.textbook?.textbook_id === activeBook?.textbook_id ? "active" : ""}`}
              key={item.id}
              onClick={() => {
                if (item.textbook) {
                  setActiveBookId(item.textbook.textbook_id);
                  setActiveChapterId(item.textbook.chapters[0]?.chapter_id ?? null);
                }
              }}
              type="button"
            >
              <FileText size={20} />
              <span className="file-main">
                <strong>{item.filename}</strong>
                <small>
                  {item.format.toUpperCase()} · {formatBytes(item.size)}
                </small>
                {item.error ? <em>{item.error}</em> : null}
              </span>
              <StatusBadge status={item.status} />
            </button>
          ))}
        </div>
      </section>

      <section className="panel content">
        {activeBook ? (
          <>
            <div className="book-header">
              <div>
                <p className="eyebrow">{activeBook.filename}</p>
                <h2>{activeBook.title}</h2>
              </div>
              <div className="stats">
                <span>{activeBook.total_pages ?? "-"} 页</span>
                <span>{activeBook.total_chars.toLocaleString()} 字符</span>
                <span>{activeBook.chapters.length} 章</span>
              </div>
            </div>

            <nav className="tabs" aria-label="功能模块">
              <TabButton active={activeTab === "parse"} onClick={() => setActiveTab("parse")}>教材解析</TabButton>
              <TabButton active={activeTab === "graph"} onClick={() => setActiveTab("graph")}>知识图谱</TabButton>
              <TabButton active={activeTab === "integration"} onClick={() => setActiveTab("integration")}>跨教材整合</TabButton>
              <TabButton active={activeTab === "rag"} onClick={() => setActiveTab("rag")}>RAG 问答</TabButton>
            </nav>

            {activeTab === "parse" ? (
              <ChapterBrowser activeBook={activeBook} activeChapter={activeChapter} setActiveChapterId={setActiveChapterId} />
            ) : null}
            {activeTab === "graph" ? (
              <KnowledgeGraphPanel
                textbooks={parsedBooks}
                activeBookId={activeBook.textbook_id}
                apiBase={API_BASE}
                onGraphReady={setGraphs}
              />
            ) : null}
            {activeTab === "integration" ? <IntegrationPanel graphs={graphs} textbooks={parsedBooks} apiBase={API_BASE} /> : null}
            {activeTab === "rag" ? <RagPanel textbooks={parsedBooks} apiBase={API_BASE} /> : null}
          </>
        ) : (
          <div className="empty-state">
            <UploadCloud size={42} />
            <h2>上传教材后，这里会显示章节结构</h2>
            <p>PDF 会逐页解析并识别章节；Markdown、TXT、DOCX 会按标题或章节正则切分。</p>
          </div>
        )}
      </section>
    </main>
  );
}

function TabButton({ active, children, onClick }: { active: boolean; children: string; onClick: () => void }) {
  return (
    <button className={active ? "active" : ""} type="button" onClick={onClick}>
      {children}
    </button>
  );
}

function ChapterBrowser({
  activeBook,
  activeChapter,
  setActiveChapterId,
}: {
  activeBook: Textbook;
  activeChapter: Chapter | null;
  setActiveChapterId: (chapterId: string) => void;
}) {
  const groups = useMemo(() => buildChapterGroups(activeBook.chapters), [activeBook.chapters]);
  const [activeGroupId, setActiveGroupId] = useState<string | null>(groups[0]?.id ?? null);

  useEffect(() => {
    setActiveGroupId(groups[0]?.id ?? null);
  }, [activeBook.textbook_id, groups]);

  useEffect(() => {
    if (!activeChapter) return;
    const owner = groups.find(
      (group) => group.overview?.chapter_id === activeChapter.chapter_id || group.children.some((child) => child.chapter_id === activeChapter.chapter_id),
    );
    if (owner) setActiveGroupId(owner.id);
  }, [activeChapter, groups]);

  const activeGroup = groups.find((group) => group.id === activeGroupId) ?? groups[0] ?? null;
  const selectedChapter =
    activeGroup?.children.find((chapter) => chapter.chapter_id === activeChapter?.chapter_id) ??
    (activeGroup?.overview?.chapter_id === activeChapter?.chapter_id ? activeGroup.overview : null) ??
    activeGroup?.children[0] ??
    activeGroup?.overview ??
    null;

  function selectGroup(group: ChapterGroup) {
    setActiveGroupId(group.id);
    const nextChapter = group.children[0] ?? group.overview;
    if (nextChapter) setActiveChapterId(nextChapter.chapter_id);
  }

  return (
    <div className="chapter-layout">
      <div className="chapter-tree">
        <div className="chapter-column-title">章节目录</div>
        {groups.map((group) => (
          <section className={`chapter-tree-group ${group.id === activeGroup?.id ? "expanded" : ""}`} key={group.id}>
            <button
              className="chapter-tree-primary"
              onClick={() => selectGroup(group)}
              type="button"
            >
              <strong>{group.title}</strong>
              <span>
                {formatGroupPages(group)} · {group.children.length} 个二级标题
              </span>
            </button>

            {group.id === activeGroup?.id ? (
              <div className="chapter-tree-children">
                {group.overview ? (
                  <button
                    className={selectedChapter?.chapter_id === group.overview.chapter_id ? "selected" : ""}
                    onClick={() => setActiveChapterId(group.overview!.chapter_id)}
                    type="button"
                  >
                    <strong>章节概览</strong>
                    <span>{formatPages(group.overview)} · {group.overview.char_count.toLocaleString()} 字符</span>
                  </button>
                ) : null}
                {group.children.length ? (
                  group.children.map((chapter) => (
                    <button
                      className={chapter.chapter_id === selectedChapter?.chapter_id ? "selected" : ""}
                      key={chapter.chapter_id}
                      onClick={() => setActiveChapterId(chapter.chapter_id)}
                      type="button"
                    >
                      <strong>{chapter.title}</strong>
                      <span>
                        {formatPages(chapter)} · {chapter.char_count.toLocaleString()} 字符
                      </span>
                    </button>
                  ))
                ) : (
                  <p className="muted chapter-empty-note">该一级标题下暂未识别出二级标题。</p>
                )}
              </div>
            ) : null}
          </section>
        ))}
      </div>

      <article className="chapter-preview">
        {selectedChapter ? (
          <>
            <div className="preview-header">
              <h3>{selectedChapter.title}</h3>
              <span>{formatPages(selectedChapter)}</span>
            </div>
            <pre>{selectedChapter.content.slice(0, 4000) || "该章节没有可预览文本。"}</pre>
          </>
        ) : (
          <p className="muted">请选择章节查看正文预览。</p>
        )}
      </article>
    </div>
  );
}

function buildChapterGroups(chapters: Chapter[]): ChapterGroup[] {
  const groups: ChapterGroup[] = [];
  let current: ChapterGroup | null = null;

  for (let index = 0; index < chapters.length; index += 1) {
    const chapter = chapters[index];
    const kind = classifyChapterTitle(chapter.title);
    const next = chapters[index + 1];

    if (kind === "level2" && next && classifyChapterTitle(next.title) === "level1" && samePage(chapter, next)) {
      current = createGroup(next);
      groups.push(current);
      addChildToGroup(current, chapter);
      mergeChapterIntoGroup(current, next);
      index += 1;
      continue;
    }

    if (kind === "level1") {
      const existing = groups.find((group) => normalizeHeading(group.title) === normalizeHeading(chapter.title));
      if (existing) {
        current = existing;
        mergeChapterIntoGroup(existing, chapter);
      } else {
        current = createGroup(chapter);
        groups.push(current);
      }
      continue;
    }

    if (!current || kind === "front") continue;

    addChildToGroup(current, chapter);
    mergeGroupStats(current, chapter);
  }

  return groups.filter((group) => group.overview || group.children.length);
}

function createGroup(chapter: Chapter): ChapterGroup {
  return {
    id: `group_${chapter.chapter_id}`,
    title: chapter.title,
    overview: chapter,
    children: [],
    page_start: chapter.page_start,
    page_end: chapter.page_end,
    char_count: chapter.char_count,
  };
}

function addChildToGroup(group: ChapterGroup, chapter: Chapter) {
  const normalized = normalizeHeading(chapter.title);
  const existing = group.children.find((child) => normalizeHeading(child.title) === normalized && samePage(child, chapter));
  if (existing) return;
  group.children.push(chapter);
}

function mergeChapterIntoGroup(group: ChapterGroup, chapter: Chapter) {
  if (!group.overview) group.overview = chapter;
  mergeGroupStats(group, chapter);
}

function mergeGroupStats(group: ChapterGroup, chapter: Chapter) {
  group.page_start = minPage(group.page_start, chapter.page_start);
  group.page_end = maxPage(group.page_end, chapter.page_end);
  group.char_count += chapter.char_count;
}

function classifyChapterTitle(title: string): "level1" | "level2" | "front" | "other" {
  const normalized = normalizeHeading(title);
  if (!normalized || normalized === "正文") return "front";
  if (normalized === "绪论" || /^第[一二三四五六七八九十百千万零〇两\d]+章/.test(normalized)) return "level1";
  if (/^第[一二三四五六七八九十百千万零〇两\d]+节/.test(normalized)) return "level2";
  if (/^[一二三四五六七八九十]+、/.test(normalized) || /^（[一二三四五六七八九十]+）/.test(normalized)) return "level2";
  return "other";
}

function normalizeHeading(title: string) {
  return title.replace(/\s+/g, "").replace(/[|｜]/g, "").trim();
}

function samePage(left: Chapter, right: Chapter) {
  return Boolean(left.page_start && right.page_start && left.page_start === right.page_start);
}

function minPage(left: number | null, right: number | null) {
  if (!left) return right;
  if (!right) return left;
  return Math.min(left, right);
}

function maxPage(left: number | null, right: number | null) {
  if (!left) return right;
  if (!right) return left;
  return Math.max(left, right);
}

function formatGroupPages(group: ChapterGroup) {
  if (!group.page_start) return "页码未知";
  if (!group.page_end || group.page_end === group.page_start) return `第 ${group.page_start} 页`;
  return `第 ${group.page_start}-${group.page_end} 页`;
}

function StatusBadge({ status }: { status: ParseStatus }) {
  if (status === "parsing") {
    return (
      <span className="badge parsing">
        <Loader2 size={14} />
        解析中
      </span>
    );
  }
  if (status === "completed") {
    return (
      <span className="badge completed">
        <CheckCircle2 size={14} />
        已完成
      </span>
    );
  }
  return (
    <span className="badge failed">
      <AlertCircle size={14} />
      失败
    </span>
  );
}

function formatBytes(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function formatPages(chapter: Chapter) {
  if (!chapter.page_start) return "页码未知";
  if (!chapter.page_end || chapter.page_end === chapter.page_start) return `第 ${chapter.page_start} 页`;
  return `第 ${chapter.page_start}-${chapter.page_end} 页`;
}

createRoot(document.getElementById("root")!).render(<App />);
