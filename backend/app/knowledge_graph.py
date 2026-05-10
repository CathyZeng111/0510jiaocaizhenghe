from __future__ import annotations

import re
from typing import Optional

from .integration_schemas import KnowledgeEdge, KnowledgeNode, TextbookKnowledgeGraph
from .modelscope_client import ModelScopeError, chat_completion, extract_json_object, is_modelscope_configured
from .schemas import Chapter, Textbook

TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,12}|[A-Za-z][A-Za-z0-9_\-]{2,}")
STOP_TERMS = {
    "教材",
    "章节",
    "内容",
    "系统",
    "功能",
    "要求",
    "进行",
    "包括",
    "通过",
    "可以",
    "需要",
    "例如",
    "the",
    "and",
    "for",
    "with",
    "action",
    "activity",
    "figure",
    "table",
    "case",
    "overview",
    "summary",
}
INVALID_TITLE_PATTERNS = (
    re.compile(r"^\d+[\s\-_]"),
    re.compile(r"^[A-Za-z]{1,5}$"),
    re.compile(r"^[A-Za-z ]+$"),
    re.compile(r"^[图表]\d"),
    re.compile(r"^第?\d+页$"),
)
RELATION_TYPES = ("contains", "prerequisite", "parallel", "applies_to")
ALLOWED_RELATIONS = {"prerequisite", "parallel", "contains", "applies_to"}


def build_textbook_graph(
    textbook: Textbook,
    max_nodes_per_chapter: int = 8,
    use_llm: bool = False,
) -> TextbookKnowledgeGraph:
    nodes: list[KnowledgeNode] = []
    edges: list[KnowledgeEdge] = []
    chapter_nodes: dict[str, list[KnowledgeNode]] = {}
    current_primary_id: Optional[str] = None
    primary_id_by_title: dict[str, str] = {}

    for chapter in textbook.chapters:
        chapter_level = classify_chapter_level(chapter.title)
        if chapter_level == "primary":
            normalized_title = normalize_heading(chapter.title)
            existing_primary_id = primary_id_by_title.get(normalized_title)
            if existing_primary_id:
                current_primary_id = existing_primary_id
                continue

            current_primary_id = f"{textbook.textbook_id}_{chapter.chapter_id}_primary"
            primary_id_by_title[normalized_title] = current_primary_id
            nodes.append(
                KnowledgeNode(
                    node_id=current_primary_id,
                    name=chapter.title,
                    definition=extract_definition(chapter.content, chapter.title) or chapter.content[:220],
                    aliases=[],
                    source_textbook_id=textbook.textbook_id,
                    chapter_id=chapter.chapter_id,
                    node_type="primary",
                    metadata={
                        "chapter_title": chapter.title,
                        "page_start": chapter.page_start,
                        "page_end": chapter.page_end,
                        "frequency": max(1, chapter.char_count // 1000),
                        "source_excerpt": source_excerpt(chapter.content, chapter.title),
                        "level": "primary",
                    },
                )
            )
            chapter_nodes[chapter.chapter_id] = []
            continue

        parent_id = current_primary_id
        chapter_nodes[chapter.chapter_id] = []
        if not parent_id or not is_content_block_chapter(chapter):
            continue

        llm_graph = extract_llm_chapter_graph(textbook, chapter, max_nodes_per_chapter) if use_llm else None
        if llm_graph:
            llm_nodes, llm_edges = build_llm_chapter_nodes(textbook, chapter, parent_id, llm_graph)
            if not llm_nodes:
                node = build_content_block_node(textbook, chapter, parent_id)
                nodes.append(node)
                chapter_nodes[chapter.chapter_id].append(node)
                edges.extend(build_parent_edges(textbook, chapter, parent_id, [node], len(edges)))
                continue
            nodes.extend(llm_nodes)
            chapter_nodes[chapter.chapter_id].extend(llm_nodes)
            edges.extend(build_parent_edges(textbook, chapter, parent_id, llm_nodes, len(edges)))
            edges.extend(build_llm_relation_edges(textbook, chapter, llm_nodes, llm_edges, len(edges)))
            if not llm_edges:
                edges.extend(build_chapter_edges(textbook, chapter, llm_nodes, len(edges)))
        else:
            node = build_content_block_node(textbook, chapter, parent_id)
            nodes.append(node)
            chapter_nodes[chapter.chapter_id].append(node)
            edges.extend(build_parent_edges(textbook, chapter, parent_id, [node], len(edges)))

    edges.extend(build_prerequisite_edges(textbook, chapter_nodes))
    return TextbookKnowledgeGraph(
        graph_id=f"graph_{textbook.textbook_id}",
        textbook_id=textbook.textbook_id,
        textbook_title=textbook.title,
        nodes=nodes,
        edges=edges,
    )


def build_all_graphs(textbooks: list[Textbook], use_llm: bool = False) -> list[TextbookKnowledgeGraph]:
    return [build_textbook_graph(textbook, use_llm=use_llm) for textbook in textbooks]


def classify_chapter_level(title: str) -> str:
    normalized = normalize_heading(title)
    if normalized == "绪论" or re.match(r"^第[一二三四五六七八九十百千万零〇两\d]+章", normalized):
        return "primary"
    if re.match(r"^第[一二三四五六七八九十百千万零〇两\d]+节", normalized):
        return "secondary"
    if re.match(r"^[一二三四五六七八九十]+、", normalized) or re.match(r"^（[一二三四五六七八九十]+）", normalized):
        return "secondary"
    return "content"


def build_content_block_node(textbook: Textbook, chapter: Chapter, parent_id: str) -> KnowledgeNode:
    return KnowledgeNode(
        node_id=f"{textbook.textbook_id}_{chapter.chapter_id}_block",
        name=normalize_display_title(chapter.title),
        definition=summarize_content_block(chapter),
        aliases=[],
        source_textbook_id=textbook.textbook_id,
        chapter_id=chapter.chapter_id,
        node_type="secondary",
        metadata={
            "chapter_title": chapter.title,
            "page_start": chapter.page_start,
            "page_end": chapter.page_end,
            "frequency": max(1, chapter.char_count // 800),
            "source_excerpt": source_excerpt(chapter.content, chapter.title),
            "category": "内容块",
            "level": "secondary",
            "parent_id": parent_id,
            "char_count": chapter.char_count,
        },
    )


def is_content_block_chapter(chapter: Chapter) -> bool:
    title = normalize_heading(chapter.title)
    if not title or title == "正文":
        return False
    if chapter.char_count < 80:
        return False
    if title in {"本章数字资源", "推荐阅读", "中英文名词对照索引"}:
        return False
    if re.match(r"^\d+主编|^\d+副主编", title):
        return False
    if re.fullmatch(r"[A-Za-z ]+", chapter.title.strip()):
        return False
    return True


def summarize_content_block(chapter: Chapter) -> Optional[str]:
    content = re.sub(r"\s+", " ", chapter.content or "").strip()
    if not content:
        return None
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[。！？!?；;])", content) if sentence.strip()]
    for sentence in sentences:
        if len(sentence) >= 28:
            return sentence[:260]
    return content[:260]


def normalize_heading(title: str) -> str:
    return re.sub(r"[\s|｜]+", "", title or "").strip()


def normalize_display_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").replace("|", " | ")).strip()


def extract_llm_chapter_graph(textbook: Textbook, chapter: Chapter, max_nodes: int) -> Optional[dict[str, object]]:
    if not is_modelscope_configured():
        return None
    prompt = f"""
请从教材章节中提取核心知识点并识别关系。输出严格 JSON，不要输出解释。
JSON schema:
{{
  "nodes": [
    {{
      "id": "node_001",
      "name": "动作电位",
      "definition": "细胞受到刺激后，膜电位发生的一次快速而可逆的倒转。",
      "category": "核心概念",
      "chapter": "第二章 细胞的基本功能",
      "page": 35
    }}
  ],
  "edges": [
    {{
      "source": "node_001",
      "target": "node_002",
      "relation_type": "prerequisite",
      "description": "理解动作电位需要先掌握静息电位的概念。"
    }}
  ]
}}
要求：
- 最多 {max_nodes} 个知识点，必须是能代表一块教材内容的概念、方法、结构、现象或临床应用。
- 不要抽取孤立英文词、单个动词、普通名词、人名、页码、图表编号、题注、编者信息。
- 每个 name 应该像教材二级/三级知识块标题，通常是 2-18 个汉字或常用医学英文全称，不要输出 action、case、table、figure 这类词。
- 关系类型只能使用 prerequisite、parallel、contains、applies_to，至少尽量覆盖其中三类；如果章节内容不足，宁可少输出关系也不要编造。
- 只基于给定章节正文。
- 每次只处理当前一个章节，避免跨章节臆测。

few-shot:
输入章节《细胞的基本功能》：
输出节点可包含“静息电位”“动作电位”“钠通道激活”“兴奋-收缩耦联”；
输出关系可包含“静息电位 prerequisite 动作电位”“动作电位 applies_to 兴奋-收缩耦联”。

教材：{textbook.title}
章节：{chapter.title}
起始页：{chapter.page_start or "未知"}
正文：
{chapter.content[:5000]}
""".strip()
    try:
        content = chat_completion(
            [
                {"role": "system", "content": "你是医学教材知识图谱抽取助手，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
        data = extract_json_object(content)
    except (ModelScopeError, ValueError):
        return None

    if not isinstance(data.get("nodes"), list):
        return None
    if not isinstance(data.get("edges", []), list):
        data["edges"] = []
    return data


def build_llm_chapter_nodes(
    textbook: Textbook,
    chapter: Chapter,
    parent_id: str,
    llm_graph: dict[str, object],
) -> tuple[list[KnowledgeNode], list[dict[str, object]]]:
    raw_nodes = llm_graph.get("nodes", [])
    raw_edges = llm_graph.get("edges", [])
    if not isinstance(raw_nodes, list):
        return [], []

    nodes: list[KnowledgeNode] = []
    seen_names: set[str] = set()
    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not is_valid_knowledge_point_name(name):
            continue
        normalized = normalize_term(name)
        if normalized in seen_names:
            continue
        seen_names.add(normalized)
        node_id = f"{textbook.textbook_id}_{chapter.chapter_id}_kp_{len(nodes) + 1:03d}"
        nodes.append(
            KnowledgeNode(
                node_id=node_id,
                name=name,
                definition=str(item.get("definition", "")).strip() or extract_definition(chapter.content, name),
                aliases=[],
                source_textbook_id=textbook.textbook_id,
                chapter_id=chapter.chapter_id,
                node_type="knowledge_point",
                metadata={
                    "llm_id": str(item.get("id", f"node_{len(nodes) + 1:03d}")),
                    "chapter_title": str(item.get("chapter") or chapter.title),
                    "page_start": item.get("page") or chapter.page_start,
                    "page_end": chapter.page_end,
                    "frequency": max(1, chapter.content.count(name)),
                    "source_excerpt": source_excerpt(chapter.content, name),
                    "category": str(item.get("category", "核心概念")).strip() or "核心概念",
                    "level": "secondary",
                    "parent_id": parent_id,
                    "extractor": "modelscope",
                },
            )
        )
    return nodes, raw_edges if isinstance(raw_edges, list) else []


def is_valid_knowledge_point_name(name: str) -> bool:
    normalized = name.strip().lower()
    if normalized in STOP_TERMS or is_numeric_term(name):
        return False
    if len(name) < 2 or len(name) > 28:
        return False
    if any(pattern.search(name.strip()) for pattern in INVALID_TITLE_PATTERNS):
        return False
    if re.fullmatch(r"[a-zA-Z][a-zA-Z\- ]+", name):
        useful_english_terms = {"inflammation", "apoptosis", "homeostasis", "necrosis", "immunity", "infection"}
        return normalized in useful_english_terms or len(normalized) >= 10
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", name):
        return False
    return True


def is_numeric_term(term: str) -> bool:
    return bool(re.fullmatch(r"[\d一二三四五六七八九十百千万零〇两]+", term))


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", "", term).lower()


def extract_definition(content: str, term: str) -> Optional[str]:
    if not content:
        return None
    sentences = re.split(r"(?<=[。！？!?；;])|\n+", content)
    for sentence in sentences:
        sentence = sentence.strip()
        if term in sentence and 12 <= len(sentence) <= 260:
            return sentence
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= 24:
            return sentence[:260]
    return content[:260].strip() or None


def source_excerpt(content: str, term: object, window: int = 180) -> str:
    text = re.sub(r"\s+", " ", content or "").strip()
    if not text:
        return ""
    term_text = str(term)
    index = text.find(term_text) if term_text else -1
    if index == -1:
        return text[: window * 2]
    start = max(0, index - window)
    end = min(len(text), index + len(term_text) + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def build_parent_edges(
    textbook: Textbook,
    chapter: Chapter,
    parent_id: str,
    nodes: list[KnowledgeNode],
    offset: int,
) -> list[KnowledgeEdge]:
    return [
        KnowledgeEdge(
            edge_id=f"{textbook.textbook_id}_{chapter.chapter_id}_parent_{offset + index:03d}",
            source=parent_id,
            target=node.node_id,
            relation="contains",
            weight=1.0,
            source_textbook_id=textbook.textbook_id,
            evidence=[chapter.title],
            metadata={"hierarchy": True, "description": f"{chapter.title} 包含知识点：{node.name}"},
        )
        for index, node in enumerate(nodes, start=1)
    ]


def build_llm_relation_edges(
    textbook: Textbook,
    chapter: Chapter,
    nodes: list[KnowledgeNode],
    raw_edges: list[dict[str, object]],
    offset: int,
) -> list[KnowledgeEdge]:
    by_llm_id = {str(node.metadata.get("llm_id")): node for node in nodes}
    by_name = {normalize_term(node.name): node for node in nodes}
    edges: list[KnowledgeEdge] = []
    for raw in raw_edges:
        if not isinstance(raw, dict):
            continue
        relation = str(raw.get("relation_type", "")).strip()
        if relation not in ALLOWED_RELATIONS:
            continue
        source = resolve_llm_node(raw.get("source"), by_llm_id, by_name)
        target = resolve_llm_node(raw.get("target"), by_llm_id, by_name)
        if not source or not target or source.node_id == target.node_id:
            continue
        edges.append(
            KnowledgeEdge(
                edge_id=f"{textbook.textbook_id}_{chapter.chapter_id}_llm_edge_{offset + len(edges) + 1:03d}",
                source=source.node_id,
                target=target.node_id,
                relation=relation,
                weight=1.0,
                source_textbook_id=textbook.textbook_id,
                evidence=[chapter.title],
                metadata={
                    "description": str(raw.get("description", "")).strip()
                    or f"{source.name} 与 {target.name} 存在 {relation} 关系",
                    "chapter_id": chapter.chapter_id,
                },
            )
        )
    return edges


def resolve_llm_node(
    value: object,
    by_llm_id: dict[str, KnowledgeNode],
    by_name: dict[str, KnowledgeNode],
) -> Optional[KnowledgeNode]:
    key = str(value or "").strip()
    if not key:
        return None
    return by_llm_id.get(key) or by_name.get(normalize_term(key))


def build_chapter_edges(textbook: Textbook, chapter: Chapter, nodes: list[KnowledgeNode], offset: int = 0) -> list[KnowledgeEdge]:
    edges: list[KnowledgeEdge] = []
    if len(nodes) < 2:
        return edges

    root = nodes[0]
    for index, node in enumerate(nodes[1:], start=1):
        relation = RELATION_TYPES[index % len(RELATION_TYPES)]
        edges.append(
            KnowledgeEdge(
                edge_id=f"{textbook.textbook_id}_{chapter.chapter_id}_edge_{offset + index:03d}",
                source=root.node_id,
                target=node.node_id,
                relation=relation,
                weight=1.0,
                source_textbook_id=textbook.textbook_id,
                evidence=[chapter.title],
                metadata={"chapter_id": chapter.chapter_id, "description": f"{root.name} 与 {node.name} 同属 {chapter.title}"},
            )
        )
    return edges


def build_prerequisite_edges(
    textbook: Textbook,
    chapter_nodes: dict[str, list[KnowledgeNode]],
) -> list[KnowledgeEdge]:
    edges: list[KnowledgeEdge] = []
    ordered = [nodes[0] for _, nodes in chapter_nodes.items() if nodes]
    for index in range(len(ordered) - 1):
        edges.append(
            KnowledgeEdge(
                edge_id=f"{textbook.textbook_id}_chapter_prereq_{index + 1:03d}",
                source=ordered[index].node_id,
                target=ordered[index + 1].node_id,
                relation="prerequisite",
                weight=0.8,
                source_textbook_id=textbook.textbook_id,
                evidence=["相邻章节主概念形成学习顺序"],
                metadata={},
            )
        )
    return edges
