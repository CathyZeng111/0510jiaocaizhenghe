import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent, WheelEvent } from "react";
import { Search, ZoomIn, ZoomOut } from "lucide-react";
import "./InteractiveGraph.css";
import type {
  GraphNodeId,
  GraphViewport,
  KnowledgeGraphData,
  KnowledgeGraphEdge,
  KnowledgeGraphNode,
  TextbookSource,
} from "./graphInteractionTypes";

type Point = {
  x: number;
  y: number;
};

type InteractiveGraphProps = {
  data: KnowledgeGraphData;
  className?: string;
  height?: number | string;
  initialSelectedNodeId?: GraphNodeId | null;
  emptyMessage?: string;
  onNodeSelect?: (node: KnowledgeGraphNode | null) => void;
};

type DragState =
  | {
      type: "node";
      nodeId: GraphNodeId;
      offset: Point;
    }
  | {
      type: "canvas";
      pointerStart: Point;
      viewportStart: GraphViewport;
    }
  | null;

const WORLD_WIDTH = 1200;
const WORLD_HEIGHT = 760;
const MIN_SCALE = 0.45;
const MAX_SCALE = 2.8;
const SOURCE_COLORS = ["#2f7dd3", "#d95f43", "#1f8a70", "#7a5ccf", "#c78717", "#20889a", "#c44f7a", "#5c7f2c"];
const SHAPE_LEGEND = [
  { category: "核心概念", shape: "circle", label: "核心概念" },
  { category: "内容块", shape: "square", label: "内容块" },
  { category: "方法", shape: "diamond", label: "方法/机制" },
  { category: "章节主题", shape: "hexagon", label: "章节主题" },
] as const;

export function InteractiveGraph({
  data,
  className = "",
  height = 640,
  initialSelectedNodeId = null,
  emptyMessage = "暂无可展示的知识图谱。",
  onNodeSelect,
}: InteractiveGraphProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [query, setQuery] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState<GraphNodeId | null>(initialSelectedNodeId);
  const [nodePositions, setNodePositions] = useState<Record<GraphNodeId, Point>>({});
  const [viewport, setViewport] = useState<GraphViewport>({ x: 0, y: 0, scale: 1 });
  const [dragState, setDragState] = useState<DragState>(null);

  const nodeSignature = useMemo(() => data.nodes.map((node) => `${node.id}:${node.x ?? ""}:${node.y ?? ""}`).join("|"), [
    data.nodes,
  ]);

  useEffect(() => {
    setNodePositions(createInitialPositions(data.nodes, data.edges));
  }, [nodeSignature, data.nodes, data.edges]);

  const nodeById = useMemo(() => new Map(data.nodes.map((node) => [node.id, node])), [data.nodes]);

  const sourceMap = useMemo(() => {
    const map = new Map<string, TextbookSource & { color: string }>();
    data.sources?.forEach((source, index) => {
      map.set(source.id, {
        ...source,
        color: source.color ?? SOURCE_COLORS[index % SOURCE_COLORS.length],
      });
    });

    data.nodes.forEach((node) => {
      if (!node.sourceId || map.has(node.sourceId)) return;
      map.set(node.sourceId, {
        id: node.sourceId,
        title: node.sourceTitle ?? node.sourceId,
        color: SOURCE_COLORS[map.size % SOURCE_COLORS.length],
      });
    });

    return map;
  }, [data.nodes, data.sources]);

  const normalizedQuery = query.trim().toLowerCase();
  const matchedNodeIds = useMemo(() => {
    if (!normalizedQuery) return new Set<GraphNodeId>();
    return new Set(
      data.nodes
        .filter((node) => {
          const source = node.sourceId ? sourceMap.get(node.sourceId) : null;
          return [node.label, node.description, node.chapterTitle, node.sourceTitle, source?.title, ...(node.keywords ?? [])]
            .filter(Boolean)
            .some((value) => String(value).toLowerCase().includes(normalizedQuery));
        })
        .map((node) => node.id),
    );
  }, [data.nodes, normalizedQuery, sourceMap]);

  const visibleEdges = useMemo(
    () => data.edges.filter((edge) => nodePositions[edge.source] && nodePositions[edge.target]),
    [data.edges, nodePositions],
  );

  const selectedNode = selectedNodeId ? nodeById.get(selectedNodeId) ?? null : null;

  useEffect(() => {
    if (selectedNodeId && !nodeById.has(selectedNodeId)) {
      setSelectedNodeId(null);
      onNodeSelect?.(null);
    }
  }, [nodeById, onNodeSelect, selectedNodeId]);

  const selectNode = useCallback(
    (nodeId: GraphNodeId | null) => {
      setSelectedNodeId(nodeId);
      onNodeSelect?.(nodeId ? nodeById.get(nodeId) ?? null : null);
    },
    [nodeById, onNodeSelect],
  );

  const clientToSvgPoint = useCallback((clientX: number, clientY: number): Point => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return {
      x: ((clientX - rect.left) / rect.width) * WORLD_WIDTH,
      y: ((clientY - rect.top) / rect.height) * WORLD_HEIGHT,
    };
  }, []);

  const clientToWorldPoint = useCallback(
    (clientX: number, clientY: number): Point => {
      const point = clientToSvgPoint(clientX, clientY);
      return {
        x: (point.x - viewport.x) / viewport.scale,
        y: (point.y - viewport.y) / viewport.scale,
      };
    },
    [clientToSvgPoint, viewport],
  );

  function startNodeDrag(event: PointerEvent<SVGGElement>, nodeId: GraphNodeId) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    selectNode(nodeId);
    svgRef.current?.setPointerCapture(event.pointerId);

    const pointer = clientToWorldPoint(event.clientX, event.clientY);
    const position = nodePositions[nodeId] ?? pointer;
    setDragState({
      type: "node",
      nodeId,
      offset: {
        x: pointer.x - position.x,
        y: pointer.y - position.y,
      },
    });
  }

  function startCanvasDrag(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0) return;
    svgRef.current?.setPointerCapture(event.pointerId);
    setDragState({
      type: "canvas",
      pointerStart: clientToSvgPoint(event.clientX, event.clientY),
      viewportStart: viewport,
    });
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    if (!dragState) return;

    if (dragState.type === "node") {
      const pointer = clientToWorldPoint(event.clientX, event.clientY);
      setNodePositions((current) => ({
        ...current,
        [dragState.nodeId]: {
          x: clamp(pointer.x - dragState.offset.x, 48, WORLD_WIDTH - 48),
          y: clamp(pointer.y - dragState.offset.y, 48, WORLD_HEIGHT - 48),
        },
      }));
      return;
    }

    const pointer = clientToSvgPoint(event.clientX, event.clientY);
    setViewport({
      ...dragState.viewportStart,
      x: dragState.viewportStart.x + pointer.x - dragState.pointerStart.x,
      y: dragState.viewportStart.y + pointer.y - dragState.pointerStart.y,
    });
  }

  function stopDrag(event: PointerEvent<SVGSVGElement>) {
    if (!dragState) return;
    svgRef.current?.releasePointerCapture(event.pointerId);
    setDragState(null);
  }

  function handleWheel(event: WheelEvent<SVGSVGElement>) {
    event.preventDefault();
    const anchor = clientToSvgPoint(event.clientX, event.clientY);
    setViewport((current) => {
      const nextScale = clamp(current.scale * (event.deltaY > 0 ? 0.9 : 1.1), MIN_SCALE, MAX_SCALE);
      const worldPoint = {
        x: (anchor.x - current.x) / current.scale,
        y: (anchor.y - current.y) / current.scale,
      };
      return {
        scale: nextScale,
        x: anchor.x - worldPoint.x * nextScale,
        y: anchor.y - worldPoint.y * nextScale,
      };
    });
  }

  function zoomBy(factor: number) {
    const anchor = { x: WORLD_WIDTH / 2, y: WORLD_HEIGHT / 2 };
    setViewport((current) => {
      const nextScale = clamp(current.scale * factor, MIN_SCALE, MAX_SCALE);
      const worldPoint = {
        x: (anchor.x - current.x) / current.scale,
        y: (anchor.y - current.y) / current.scale,
      };
      return {
        scale: nextScale,
        x: anchor.x - worldPoint.x * nextScale,
        y: anchor.y - worldPoint.y * nextScale,
      };
    });
  }

  function resetView() {
    setViewport({ x: 0, y: 0, scale: 1 });
  }

  function handleNodeKeyDown(event: KeyboardEvent<SVGGElement>, nodeId: GraphNodeId) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectNode(nodeId);
    }
  }

  const searchActive = normalizedQuery.length > 0;
  const hasNodes = data.nodes.length > 0;

  return (
    <section className={`interactive-graph ${className}`} style={{ height }}>
      <div className="interactive-graph__toolbar">
        <label className="interactive-graph__search">
          <Search size={16} aria-hidden="true" />
          <input
            aria-label="搜索图谱节点"
            placeholder="搜索概念、章节或关键词"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>

        <div className="interactive-graph__actions" aria-label="图谱缩放控制">
          <button type="button" onClick={() => zoomBy(1.16)} title="放大">
            <ZoomIn size={17} aria-hidden="true" />
          </button>
          <button type="button" onClick={() => zoomBy(0.86)} title="缩小">
            <ZoomOut size={17} aria-hidden="true" />
          </button>
          <button type="button" onClick={resetView}>
            重置
          </button>
        </div>
      </div>

      <div className="interactive-graph__legend" aria-label="图谱视觉编码图例">
        <div className="interactive-graph__legend-group">
          <strong>形状=类别</strong>
          {SHAPE_LEGEND.map((item) => (
            <span key={item.category}>
              <ShapeLegendIcon shape={item.shape} />
              {item.label}
            </span>
          ))}
        </div>
        {sourceMap.size > 0 ? (
          <div className="interactive-graph__legend-group">
            <strong>颜色=来源</strong>
            {Array.from(sourceMap.values()).map((source) => (
              <span key={source.id}>
                <i className="interactive-graph__legend-color" style={{ backgroundColor: source.color }} />
                {source.title}
              </span>
            ))}
          </div>
        ) : null}
        <div className="interactive-graph__legend-group">
          <strong>大小=频次</strong>
          <span>
            <i className="interactive-graph__legend-size small" />
            少
          </span>
          <span>
            <i className="interactive-graph__legend-size large" />
            多
          </span>
        </div>
      </div>

      <div className="interactive-graph__body">
        <div className="interactive-graph__canvas">
          {hasNodes ? (
            <svg
              ref={svgRef}
              className={`interactive-graph__svg ${dragState?.type === "canvas" ? "is-panning" : ""}`}
              viewBox={`0 0 ${WORLD_WIDTH} ${WORLD_HEIGHT}`}
              role="img"
              aria-label="知识图谱交互画布"
              onPointerDown={startCanvasDrag}
              onPointerMove={handlePointerMove}
              onPointerUp={stopDrag}
              onPointerCancel={stopDrag}
              onWheel={handleWheel}
            >
              <rect className="interactive-graph__backdrop" width={WORLD_WIDTH} height={WORLD_HEIGHT} />
              <g transform={`translate(${viewport.x} ${viewport.y}) scale(${viewport.scale})`}>
                <g className="interactive-graph__edges">
                  {visibleEdges.map((edge) => {
                    const source = nodePositions[edge.source];
                    const target = nodePositions[edge.target];
                    const isConnectedToSelection =
                      selectedNodeId === edge.source || selectedNodeId === edge.target || selectedNodeId === null;
                    return (
                      <GraphEdgeLine
                        edge={edge}
                        isDimmed={Boolean(selectedNodeId && !isConnectedToSelection)}
                        key={edge.id ?? `${edge.source}-${edge.target}-${edge.label ?? ""}`}
                        source={source}
                        target={target}
                      />
                    );
                  })}
                </g>

                <g className="interactive-graph__nodes">
                  {data.nodes.map((node) => {
                    const position = nodePositions[node.id];
                    if (!position) return null;
                    const source = node.sourceId ? sourceMap.get(node.sourceId) : null;
                    const isMatched = matchedNodeIds.has(node.id);
                    const isSelected = selectedNodeId === node.id;
                    const isDimmed = searchActive && !isMatched;
                    const radius = getNodeRadius(node.frequency);

                    return (
                      <g
                        aria-label={`${node.label}，出现 ${node.frequency} 次`}
                        className={[
                          "interactive-graph__node",
                          isSelected ? "is-selected" : "",
                          isMatched ? "is-matched" : "",
                          isDimmed ? "is-dimmed" : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        key={node.id}
                        role="button"
                        tabIndex={0}
                        transform={`translate(${position.x} ${position.y})`}
                        onKeyDown={(event) => handleNodeKeyDown(event, node.id)}
                        onPointerDown={(event) => startNodeDrag(event, node.id)}
                      >
                        <circle r={radius + 8} className="interactive-graph__node-hitarea" />
                        <NodeShape node={node} radius={radius} fill={source?.color ?? "#68778f"} />
                        <text y={radius + 18}>{node.label}</text>
                      </g>
                    );
                  })}
                </g>
              </g>
            </svg>
          ) : (
            <div className="interactive-graph__empty">{emptyMessage}</div>
          )}
        </div>

        <aside className="interactive-graph__details" aria-label="节点详情">
          {selectedNode ? (
            <NodeDetails
              node={selectedNode}
              source={selectedNode.sourceId ? sourceMap.get(selectedNode.sourceId) ?? null : null}
            />
          ) : (
            <div className="interactive-graph__details-empty">
              <strong>选择一个节点</strong>
              <span>点击节点查看来源、频次、章节、定义与原文出处。</span>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}

function NodeShape({ node, radius, fill }: { node: KnowledgeGraphNode; radius: number; fill: string }) {
  const shape = getNodeShape(node);
  const className = "interactive-graph__node-shape";
  if (shape === "square") {
    const side = radius * 1.7;
    return <rect className={className} height={side} rx={3} style={{ fill }} width={side} x={-side / 2} y={-side / 2} />;
  }
  if (shape === "diamond") {
    const points = [
      `0,${-radius}`,
      `${radius},0`,
      `0,${radius}`,
      `${-radius},0`,
    ].join(" ");
    return <polygon className={className} points={points} style={{ fill }} />;
  }
  if (shape === "hexagon") {
    return <polygon className={className} points={hexagonPoints(radius)} style={{ fill }} />;
  }
  return <circle r={radius} className={className} style={{ fill }} />;
}

function ShapeLegendIcon({ shape }: { shape: (typeof SHAPE_LEGEND)[number]["shape"] }) {
  return (
    <svg aria-hidden="true" className="interactive-graph__legend-shape" viewBox="-12 -12 24 24">
      {shape === "square" ? <rect x="-8" y="-8" width="16" height="16" rx="2" /> : null}
      {shape === "diamond" ? <polygon points="0,-10 10,0 0,10 -10,0" /> : null}
      {shape === "hexagon" ? <polygon points={hexagonPoints(10)} /> : null}
      {shape === "circle" ? <circle r="8" /> : null}
    </svg>
  );
}

function getNodeShape(node: KnowledgeGraphNode) {
  const category = String(node.metadata?.category ?? "");
  const type = String(node.metadata?.type ?? "");
  const level = String(node.metadata?.level ?? "");
  if (category.includes("方法") || category.includes("机制") || category.includes("现象")) return "diamond";
  if (category.includes("内容块") || type === "secondary") return "square";
  if (level === "primary" || type === "primary" || category.includes("章节")) return "hexagon";
  return "circle";
}

function hexagonPoints(radius: number) {
  return Array.from({ length: 6 }, (_, index) => {
    const angle = (Math.PI / 3) * index - Math.PI / 6;
    return `${Math.cos(angle) * radius},${Math.sin(angle) * radius}`;
  }).join(" ");
}

function GraphEdgeLine({
  edge,
  source,
  target,
  isDimmed,
}: {
  edge: KnowledgeGraphEdge;
  source: Point;
  target: Point;
  isDimmed: boolean;
}) {
  const weight = clamp(edge.weight ?? 1, 1, 8);
  const midpoint = {
    x: (source.x + target.x) / 2,
    y: (source.y + target.y) / 2,
  };

  return (
    <g className={`interactive-graph__edge ${isDimmed ? "is-dimmed" : ""}`}>
      <line x1={source.x} y1={source.y} x2={target.x} y2={target.y} strokeWidth={1.2 + weight * 0.45} />
      {edge.label ? (
        <text x={midpoint.x} y={midpoint.y - 8}>
          {edge.label}
        </text>
      ) : null}
    </g>
  );
}

function NodeDetails({ node, source }: { node: KnowledgeGraphNode; source: (TextbookSource & { color: string }) | null }) {
  const conflicts = Array.isArray(node.metadata?.conflicts) ? node.metadata.conflicts : [];
  const metadataEntries = Object.entries(node.metadata ?? {}).filter(
    ([key, value]) => key !== "conflicts" && value !== undefined && value !== null,
  );

  return (
    <>
      <div className="interactive-graph__details-header">
        <span className="interactive-graph__source-dot" style={{ backgroundColor: source?.color ?? "#68778f" }} />
        <div>
          <h3>{node.label}</h3>
          <p>{source?.title ?? node.sourceTitle ?? "未知教材来源"}</p>
        </div>
      </div>

      <dl className="interactive-graph__facts">
        <div>
          <dt>出现频次</dt>
          <dd>{node.frequency.toLocaleString()}</dd>
        </div>
        {node.chapterTitle ? (
          <div>
            <dt>关联章节</dt>
            <dd>{node.chapterTitle}</dd>
          </div>
        ) : null}
        {node.keywords?.length ? (
          <div>
            <dt>关键词</dt>
            <dd>{node.keywords.join("、")}</dd>
          </div>
        ) : null}
      </dl>

      {node.description ? <p className="interactive-graph__description">{node.description}</p> : null}

      {conflicts.length ? (
        <div className="interactive-graph__conflicts" aria-label="知识冲突检测">
          <strong>⚠ 定义不一致</strong>
          {conflicts.slice(0, 3).map((conflict, index) => (
            <article key={readConflictString(conflict, "conflict_id") || index}>
              <p>
                {formatConflictText(conflict)}
              </p>
              <small>
                定义重合度 {formatConflictScore(conflict, "definition_jaccard")} · 名称重合度{" "}
                {formatConflictScore(conflict, "token_jaccard")}
              </small>
            </article>
          ))}
        </div>
      ) : null}

      {node.sourceExcerpt ? (
        <blockquote className="interactive-graph__excerpt">
          <strong>原文出处</strong>
          <span>{node.sourceExcerpt}</span>
        </blockquote>
      ) : null}

      {metadataEntries.length > 0 ? (
        <dl className="interactive-graph__metadata">
          {metadataEntries.map(([key, value]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>{String(value)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
    </>
  );
}

function formatConflictText(conflict: unknown) {
  const titles = readConflictArray(conflict, "textbook_titles");
  if (titles.length >= 2) {
    return `${titles.slice(0, 2).join(" / ")} 定义不一致`;
  }
  const names = readConflictArray(conflict, "node_names");
  return names.length ? `${names.join(" / ")} 定义不一致` : "不同教材对该知识点的定义不一致";
}

function readConflictArray(conflict: unknown, key: string) {
  if (!conflict || typeof conflict !== "object") return [];
  const value = (conflict as Record<string, unknown>)[key];
  return Array.isArray(value) ? value.map((item) => String(item)).filter(Boolean) : [];
}

function readConflictString(conflict: unknown, key: string) {
  if (!conflict || typeof conflict !== "object") return "";
  const value = (conflict as Record<string, unknown>)[key];
  return typeof value === "string" ? value : "";
}

function formatConflictScore(conflict: unknown, key: string) {
  if (!conflict || typeof conflict !== "object") return "-";
  const value = (conflict as Record<string, unknown>)[key];
  return typeof value === "number" ? value.toFixed(2) : "-";
}

function createInitialPositions(nodes: KnowledgeGraphNode[], edges: KnowledgeGraphEdge[]): Record<GraphNodeId, Point> {
  const providedPositions = new Map(nodes.filter(hasPosition).map((node) => [node.id, { x: node.x, y: node.y }]));
  const degreeMap = new Map<GraphNodeId, number>();
  edges.forEach((edge) => {
    degreeMap.set(edge.source, (degreeMap.get(edge.source) ?? 0) + 1);
    degreeMap.set(edge.target, (degreeMap.get(edge.target) ?? 0) + 1);
  });

  const sortedNodes = [...nodes].sort((first, second) => {
    const degreeDelta = (degreeMap.get(second.id) ?? 0) - (degreeMap.get(first.id) ?? 0);
    if (degreeDelta !== 0) return degreeDelta;
    return second.frequency - first.frequency;
  });

  const center = { x: WORLD_WIDTH / 2, y: WORLD_HEIGHT / 2 };
  const result: Record<GraphNodeId, Point> = {};

  sortedNodes.forEach((node, index) => {
    const provided = providedPositions.get(node.id);
    if (provided) {
      result[node.id] = {
        x: clamp(provided.x, 48, WORLD_WIDTH - 48),
        y: clamp(provided.y, 48, WORLD_HEIGHT - 48),
      };
      return;
    }

    if (index === 0) {
      result[node.id] = center;
      return;
    }

    const ring = Math.ceil(Math.sqrt(index));
    const angle = index * 2.399963229728653;
    const radius = 92 + ring * 46;
    result[node.id] = {
      x: clamp(center.x + Math.cos(angle) * radius, 54, WORLD_WIDTH - 54),
      y: clamp(center.y + Math.sin(angle) * radius * 0.72, 54, WORLD_HEIGHT - 54),
    };
  });

  return result;
}

function hasPosition(node: KnowledgeGraphNode): node is KnowledgeGraphNode & { x: number; y: number } {
  return typeof node.x === "number" && typeof node.y === "number";
}

function getNodeRadius(frequency: number) {
  return clamp(15 + Math.sqrt(Math.max(0, frequency)) * 3.2, 17, 42);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export default InteractiveGraph;
