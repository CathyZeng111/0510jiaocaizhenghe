from __future__ import annotations

import copy
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from .integration_schemas import (
    CompressionStats,
    GraphIntegrationResult,
    IntegratedKnowledgeGraph,
    IntegrationDecision,
    KnowledgeConflict,
    KnowledgeEdge,
    KnowledgeNode,
    SimilarityBreakdown,
    TeacherFeedbackChange,
    TeacherFeedbackResult,
    TextbookKnowledgeGraph,
)
from .modelscope_client import ModelScopeError, create_embeddings


MERGE_THRESHOLD = 0.72
WEAK_KEEP_THRESHOLD = 0.48
CHAR_BUDGET_RATIO = 0.30
PUNCTUATION_RE = re.compile(r"[\s\-_·•:：,，.。;；、/\\|()[\]{}<>《》“”\"'`~!！?？]+")
LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EMPTY_NAME_RE = re.compile(r"^(无|空|n/?a|none|null|unknown|未命名)$", re.IGNORECASE)
SPLIT_SEPARATORS_RE = re.compile(r"\s*(?:、|,|，|/|和|与|及|\+)\s*")
SEMANTIC_SYNONYMS = {
    "leukocyte": "白细胞",
    "leucocyte": "白细胞",
    "whitebloodcell": "白细胞",
    "whitebloodcells": "白细胞",
    "白blood细胞": "白细胞",
    "白血球": "白细胞",
}


@dataclass(frozen=True)
class NodeRef:
    graph: TextbookKnowledgeGraph
    node: KnowledgeNode


def integrate_textbook_graphs(
    graphs: list[TextbookKnowledgeGraph],
    merge_threshold: float = MERGE_THRESHOLD,
) -> GraphIntegrationResult:
    """Integrate textbook knowledge graphs with embedding + lexical semantic alignment."""
    refs = [NodeRef(graph=graph, node=node) for graph in graphs for node in graph.nodes]
    removed_refs = [ref for ref in refs if should_remove_node(ref.node)]
    valid_refs = [ref for ref in refs if ref not in removed_refs]
    embedding_map = build_embedding_map(valid_refs)
    conflicts = detect_definition_conflicts(valid_refs)
    groups = build_merge_groups(valid_refs, merge_threshold, embedding_map)

    decisions: list[IntegrationDecision] = []
    node_id_map: dict[str, str] = {}
    merged_nodes: list[KnowledgeNode] = []

    for index, ref in enumerate(removed_refs, start=1):
        decisions.append(
            IntegrationDecision(
                decision_id=f"decision_remove_{index:03d}",
                action="remove",
                affected_nodes=[qualified_node_id(ref.graph, ref.node)],
                result_node=None,
                decision="remove",
                node_ids=[qualified_node_id(ref.graph, ref.node)],
                canonical_name=ref.node.name,
                score=1.0,
                confidence=1.0,
                reason="节点名称为空或占位，移出整合图谱。",
            )
        )

    for group_index, group in enumerate(groups, start=1):
        canonical = choose_canonical_node(group)
        merged_id = f"kg_node_{group_index:04d}"
        merged_node = merge_nodes(merged_id, group)
        merged_nodes.append(merged_node)

        for ref in group:
            node_id_map[qualified_node_id(ref.graph, ref.node)] = merged_id

        if len(group) > 1:
            comparison = best_group_similarity(group, embedding_map)
            decisions.append(
                IntegrationDecision(
                    decision_id=f"decision_merge_{group_index:03d}",
                    action="merge",
                    affected_nodes=[qualified_node_id(ref.graph, ref.node) for ref in group],
                    result_node=merged_id,
                    decision="merge",
                    node_ids=[qualified_node_id(ref.graph, ref.node) for ref in group],
                    canonical_name=canonical.name,
                    score=comparison.final_score,
                    confidence=comparison.final_score,
                    reason=f"{len(group)} 个跨教材节点在名称、定义或 embedding 语义上等价，合并为 {canonical.name}。",
                    similarity=comparison,
                )
            )
        else:
            ref = group[0]
            best_score = best_similarity_against_group(
                ref,
                [candidate for candidate in valid_refs if candidate != ref],
                embedding_map,
            )
            decisions.append(
                IntegrationDecision(
                    decision_id=f"decision_keep_{group_index:03d}",
                    action="keep",
                    affected_nodes=[qualified_node_id(ref.graph, ref.node)],
                    result_node=merged_id,
                    decision="keep",
                    node_ids=[qualified_node_id(ref.graph, ref.node)],
                    canonical_name=ref.node.name,
                    score=best_score,
                    confidence=round(1 - min(best_score, merge_threshold) / max(merge_threshold, 0.01), 4),
                    reason=(
                        "存在弱相似概念但未达到合并阈值，保留为独立节点。"
                        if best_score >= WEAK_KEEP_THRESHOLD
                        else "未发现足够相似的跨教材概念，保留为独立节点。"
                    ),
                )
            )

    merged_edges = merge_edges(graphs, node_id_map)
    merged_graph = IntegratedKnowledgeGraph(
        textbook_ids=[graph.textbook_id for graph in graphs],
        nodes=merged_nodes,
        edges=merged_edges,
    )
    enforce_char_budget(merged_graph, decisions, refs)
    attach_conflicts_to_merged_nodes(merged_graph, node_id_map, conflicts)
    return GraphIntegrationResult(
        decisions=decisions,
        merged_graph=merged_graph,
        stats=build_stats(graphs, merged_graph, decisions),
        conflicts=conflicts,
    )


def apply_teacher_feedback(
    integration_result: GraphIntegrationResult,
    feedback_text: str,
) -> TeacherFeedbackResult:
    """Apply simple teacher text feedback: 保留/删除/拆分/合并."""
    result = copy.deepcopy(integration_result)
    changes: list[TeacherFeedbackChange] = []
    unapplied: list[str] = []

    for sentence in split_feedback_sentences(feedback_text):
        action = detect_feedback_action(sentence)
        if action == "remove":
            change = apply_remove_feedback(result, sentence)
        elif action == "keep":
            change = apply_keep_feedback(result, sentence)
        elif action == "split":
            change = apply_split_feedback(result, sentence)
        elif action == "merge":
            change = apply_merge_feedback(result, sentence)
        else:
            change = None

        if change:
            changes.append(change)
        else:
            unapplied.append(sentence)

    refresh_stats_after_feedback(result)
    return TeacherFeedbackResult(result=result, changes=changes, unapplied=unapplied)


def respond_to_teacher_message(
    integration_result: GraphIntegrationResult,
    message: str,
) -> tuple[TeacherFeedbackResult, str]:
    """Answer explanation questions or apply teacher feedback to the integrated graph."""
    cleaned = (message or "").strip()
    if not cleaned:
        return TeacherFeedbackResult(result=copy.deepcopy(integration_result)), "请输入需要解释或调整的整合建议。"

    if is_explanation_request(cleaned):
        return TeacherFeedbackResult(result=copy.deepcopy(integration_result)), explain_integration_decision(
            integration_result,
            cleaned,
        )

    feedback_result = apply_teacher_feedback(integration_result, cleaned)
    if feedback_result.changes:
        return feedback_result, summarize_feedback_changes(feedback_result)

    fallback = explain_integration_decision(integration_result, cleaned)
    if fallback.startswith("没有找到"):
        return feedback_result, "我没有匹配到可调整的知识点。请尽量使用知识点原名，例如：请保留“免疫应答”，或把“抗原”和“免疫原”拆开。"
    return feedback_result, fallback


def is_explanation_request(message: str) -> bool:
    return bool(re.search(r"为什么|为何|原因|解释|怎么判定|依据|凭什么", message))


def summarize_feedback_changes(feedback_result: TeacherFeedbackResult) -> str:
    action_labels = {
        "keep": "保留",
        "remove": "删除",
        "split": "拆分",
        "merge": "合并",
        "unknown": "未识别",
    }
    parts = [
        f"{action_labels.get(change.action, change.action)}：{change.note}（{len(change.affected_node_ids)} 个节点）"
        for change in feedback_result.changes
    ]
    reply = "已根据教师反馈更新整合结果：" + "；".join(parts)
    if feedback_result.unapplied:
        reply += f"。未处理：{'；'.join(feedback_result.unapplied)}"
    return reply


def explain_integration_decision(result: GraphIntegrationResult, question: str) -> str:
    matches = rank_matching_decisions(result, question)
    if not matches:
        return "没有找到与这条问题直接相关的整合决策。可以引用知识点名称来追问，例如：为什么合并“炎症”和“炎症反应”？"

    lines: list[str] = []
    for decision in matches[:3]:
        action_label = {"merge": "合并", "keep": "保留", "remove": "删除"}.get(decision.action, decision.action)
        source_names = source_names_for_decision(result, decision)
        node_hint = f"；涉及节点：{'、'.join(source_names[:6])}" if source_names else ""
        similarity_hint = ""
        if decision.similarity:
            similarity_hint = (
                f"；相似度：embedding {decision.similarity.embedding_similarity:.2f}，"
                f"综合 {decision.similarity.final_score:.2f}"
            )
        lines.append(
            f"{decision.canonical_name} 的决策是“{action_label}”。理由：{decision.reason}"
            f"{node_hint}{similarity_hint}；置信度 {decision.confidence:.2f}。"
        )
    return "\n".join(lines)


def rank_matching_decisions(result: GraphIntegrationResult, question: str) -> list[IntegrationDecision]:
    targets = extract_quoted_or_named_targets(question)
    normalized_targets = [normalize_name(target) for target in targets if normalize_name(target)]
    wanted_actions: set[str] = set()
    if re.search(r"合并|归并|并入", question):
        wanted_actions.add("merge")
    if re.search(r"删除|移除|去掉", question):
        wanted_actions.add("remove")
    if re.search(r"保留|单独|不应该", question):
        wanted_actions.add("keep")

    scored: list[tuple[int, IntegrationDecision]] = []
    for decision in result.decisions:
        score = 0
        decision_text = normalize_name(
            " ".join(
                [
                    decision.canonical_name,
                    decision.reason,
                    decision.result_node or "",
                    " ".join(decision.affected_nodes),
                    " ".join(decision.node_ids),
                    " ".join(source_names_for_decision(result, decision)),
                ]
            )
        )
        if wanted_actions and decision.action in wanted_actions:
            score += 2
        for target in normalized_targets:
            if target and (target in decision_text or decision_text in target):
                score += 4
        if not normalized_targets and score:
            score += 1
        if score:
            scored.append((score, decision))

    scored.sort(key=lambda item: (item[0], item[1].confidence, item[1].score), reverse=True)
    return [decision for _, decision in scored]


def source_names_for_decision(result: GraphIntegrationResult, decision: IntegrationDecision) -> list[str]:
    names: list[str] = []
    result_ids = {decision.result_node, *decision.node_ids, *decision.affected_nodes}
    for node in result.merged_graph.nodes:
        node_source_ids = {
            f"{source.get('textbook_id')}:{source.get('node_id')}"
            for source in node.metadata.get("sources", [])
            if isinstance(source, dict) and source.get("textbook_id") and source.get("node_id")
        }
        if node.node_id in result_ids or node_source_ids & result_ids:
            names.extend([node.name, *node.aliases])
    return unique_sorted([name for name in names if name])


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    normalized = PUNCTUATION_RE.sub("", normalized)
    return normalized.strip()


def semantic_key(value: str) -> str:
    normalized = normalize_name(value)
    if normalized in SEMANTIC_SYNONYMS:
        return SEMANTIC_SYNONYMS[normalized]
    for source, target in SEMANTIC_SYNONYMS.items():
        if source in normalized:
            normalized = normalized.replace(source, target)
    return normalized


def tokenize(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    latin_tokens = set(LATIN_TOKEN_RE.findall(normalized))
    cjk_chars = CJK_RE.findall(normalized)
    cjk_tokens = set(cjk_chars)
    cjk_tokens.update("".join(cjk_chars[index : index + 2]) for index in range(max(0, len(cjk_chars) - 1)))
    return {token for token in latin_tokens | cjk_tokens if token}


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def compare_nodes(
    left: KnowledgeNode,
    right: KnowledgeNode,
    left_embedding: Optional[list[float]] = None,
    right_embedding: Optional[list[float]] = None,
) -> SimilarityBreakdown:
    left_name = normalize_name(left.name)
    right_name = normalize_name(right.name)
    left_semantic = semantic_key(left.name)
    right_semantic = semantic_key(right.name)
    exact_name = 1.0 if left_name and left_name == right_name else 0.0
    semantic_exact = 1.0 if left_semantic and left_semantic == right_semantic else 0.0
    substring_name = 0.85 if left_name and right_name and (left_name in right_name or right_name in left_name) else 0.0
    token_score = jaccard(tokenize(" ".join([left.name, *left.aliases])), tokenize(" ".join([right.name, *right.aliases])))
    definition_score = jaccard(tokenize(left.definition or ""), tokenize(right.definition or ""))
    embedding_score = vector_cosine(left_embedding, right_embedding)
    normalized_score = max(exact_name, semantic_exact, substring_name)
    lexical_score = (0.45 * normalized_score) + (0.35 * token_score) + (0.20 * definition_score)
    semantic_score = (0.50 * embedding_score) + (0.25 * token_score) + (0.15 * definition_score) + (0.10 * normalized_score)
    final_score = round(max(lexical_score, semantic_score), 4)
    if exact_name or semantic_exact:
        final_score = max(final_score, 0.9)
    return SimilarityBreakdown(
        normalized_name=round(normalized_score, 4),
        token_jaccard=round(token_score, 4),
        definition_jaccard=round(definition_score, 4),
        embedding_similarity=round(embedding_score, 4),
        final_score=round(final_score, 4),
    )


def detect_definition_conflicts(refs: list[NodeRef]) -> list[KnowledgeConflict]:
    """Find same-name concepts whose definitions diverge across textbooks."""
    grouped: dict[str, list[NodeRef]] = defaultdict(list)
    for ref in refs:
        normalized = normalize_name(ref.node.name)
        if len(normalized) < 2 or EMPTY_NAME_RE.match(normalized):
            continue
        if is_generic_conflict_name(ref.node.name, normalized):
            continue
        if len(ref.node.definition or "") < 20:
            continue
        grouped[normalized].append(ref)

    conflicts: list[KnowledgeConflict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for normalized_name, group in grouped.items():
        if len(group) < 2:
            continue
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if left.graph.textbook_id == right.graph.textbook_id:
                    continue
                left_id = qualified_node_id(left.graph, left.node)
                right_id = qualified_node_id(right.graph, right.node)
                pair_key = tuple(sorted([left_id, right_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                similarity = compare_nodes(left.node, right.node)
                if similarity.token_jaccard <= 0.6 or similarity.definition_jaccard >= 0.3:
                    continue

                conflicts.append(
                    KnowledgeConflict(
                        conflict_id=f"conflict_{len(conflicts) + 1:03d}",
                        normalized_name=normalized_name,
                        node_ids=[left_id, right_id],
                        node_names=unique_sorted([left.node.name, right.node.name]),
                        textbook_titles=unique_sorted([left.graph.textbook_title, right.graph.textbook_title]),
                        token_jaccard=similarity.token_jaccard,
                        definition_jaccard=similarity.definition_jaccard,
                        reason=(
                            "知识点名称高度一致，但两本教材给出的定义文本重合度较低，"
                            "建议教师检查是否存在口径、范围或教学侧重点冲突。"
                        ),
                        definitions=[
                            conflict_definition_source(left),
                            conflict_definition_source(right),
                        ],
                    )
                )
    return conflicts


def is_generic_conflict_name(name: str, normalized: str) -> bool:
    if re.fullmatch(r"第[一二三四五六七八九十百千万零〇两\d]+[章节篇编部讲]", normalized):
        return True
    if re.fullmatch(r"第[一二三四五六七八九十百千万零〇两\d]+节概述", normalized):
        return True
    if any(marker in normalized for marker in ("概述", "临床病例分析", "本章小结", "学习目标", "复习题")):
        return True
    clean = normalize_name(name)
    return clean in {"正文", "目录", "前言", "绪论", "附录"}


def conflict_definition_source(ref: NodeRef):
    return {
        "node_id": qualified_node_id(ref.graph, ref.node),
        "node_name": ref.node.name,
        "textbook_id": ref.graph.textbook_id,
        "textbook_title": ref.graph.textbook_title,
        "definition": trim_conflict_definition(ref.node.definition),
    }


def trim_conflict_definition(definition: Optional[str]) -> Optional[str]:
    if not definition:
        return definition
    cleaned = collapse_space(definition)
    return cleaned if len(cleaned) <= 260 else cleaned[:260].rstrip() + "..."


def collapse_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def attach_conflicts_to_merged_nodes(
    merged_graph: IntegratedKnowledgeGraph,
    node_id_map: dict[str, str],
    conflicts: list[KnowledgeConflict],
) -> None:
    node_by_id = {node.node_id: node for node in merged_graph.nodes}
    for conflict in conflicts:
        target_ids = {node_id_map.get(node_id) for node_id in conflict.node_ids}
        for target_id in target_ids:
            if not target_id or target_id not in node_by_id:
                continue
            node = node_by_id[target_id]
            existing = node.metadata.get("conflicts")
            conflict_payload = conflict.model_dump() if hasattr(conflict, "model_dump") else conflict.dict()
            if isinstance(existing, list):
                existing.append(conflict_payload)
            else:
                node.metadata["conflicts"] = [conflict_payload]


def build_embedding_map(refs: list[NodeRef]) -> dict[str, list[float]]:
    texts = [embedding_text(ref.node) for ref in refs]
    node_ids = [qualified_node_id(ref.graph, ref.node) for ref in refs]
    embeddings: dict[str, list[float]] = {}
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_ids = node_ids[start : start + batch_size]
        try:
            batch_embeddings = create_embeddings(batch_texts)
        except ModelScopeError:
            return embeddings
        for node_id, embedding in zip(batch_ids, batch_embeddings):
            embeddings[node_id] = embedding
    return embeddings


def embedding_text(node: KnowledgeNode) -> str:
    parts = [
        node.name,
        " ".join(node.aliases),
        node.definition or "",
        str(node.metadata.get("category", "")),
        str(node.metadata.get("chapter_title", "")),
    ]
    return "\n".join(part for part in parts if part).strip()[:1200]


def build_merge_groups(
    refs: list[NodeRef],
    merge_threshold: float,
    embedding_map: dict[str, list[float]],
) -> list[list[NodeRef]]:
    parent = {qualified_node_id(ref.graph, ref.node): qualified_node_id(ref.graph, ref.node) for ref in refs}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(refs):
        for right in refs[left_index + 1 :]:
            similarity = compare_refs(left, right, embedding_map)
            same_book = left.graph.textbook_id == right.graph.textbook_id
            threshold = max(0.9, merge_threshold) if same_book else merge_threshold
            if is_structure_node(left.node) or is_structure_node(right.node):
                threshold = max(threshold, 0.96)
            if similarity.final_score >= threshold:
                union(qualified_node_id(left.graph, left.node), qualified_node_id(right.graph, right.node))

    grouped: dict[str, list[NodeRef]] = defaultdict(list)
    for ref in refs:
        grouped[find(qualified_node_id(ref.graph, ref.node))].append(ref)

    return sorted(grouped.values(), key=lambda group: normalize_name(choose_canonical_node(group).name))


def compare_refs(left: NodeRef, right: NodeRef, embedding_map: dict[str, list[float]]) -> SimilarityBreakdown:
    left_id = qualified_node_id(left.graph, left.node)
    right_id = qualified_node_id(right.graph, right.node)
    return compare_nodes(left.node, right.node, embedding_map.get(left_id), embedding_map.get(right_id))


def vector_cosine(left: Optional[list[float]], right: Optional[list[float]]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))


def should_remove_node(node: KnowledgeNode) -> bool:
    normalized = normalize_name(node.name)
    return not normalized or bool(EMPTY_NAME_RE.fullmatch(normalized))


def is_structure_node(node: KnowledgeNode) -> bool:
    return node.metadata.get("level") == "primary" or node.node_type == "primary"


def choose_canonical_node(group: list[NodeRef]) -> KnowledgeNode:
    return sorted(
        (ref.node for ref in group),
        key=lambda node: (0 if CJK_RE.search(node.name or "") else 1, -len(node.definition or ""), len(node.name), node.name),
    )[0]


def merge_nodes(merged_id: str, group: list[NodeRef]) -> KnowledgeNode:
    canonical = choose_canonical_node(group)
    aliases: list[str] = []
    definitions: list[str] = []
    sources: list[dict[str, Optional[str]]] = []

    for ref in group:
        if ref.node.name != canonical.name:
            aliases.append(ref.node.name)
        aliases.extend(ref.node.aliases)
        if ref.node.definition:
            definitions.append(ref.node.definition)
        sources.append(
            {
                "textbook_id": ref.graph.textbook_id,
                "textbook_title": ref.graph.textbook_title,
                "node_id": ref.node.node_id,
                "chapter_id": ref.node.chapter_id,
            }
        )

    return KnowledgeNode(
        node_id=merged_id,
        name=canonical.name,
        definition=select_definition(definitions),
        aliases=unique_sorted(aliases),
        source_textbook_id=canonical.source_textbook_id,
        chapter_id=canonical.chapter_id,
        node_type=canonical.node_type,
        metadata={
            "source_count": len(group),
            "sources": sources,
            "normalized_name": normalize_name(canonical.name),
        },
    )


def select_definition(definitions: list[str]) -> Optional[str]:
    clean_definitions = [definition.strip() for definition in definitions if definition and definition.strip()]
    if not clean_definitions:
        return None
    return max(clean_definitions, key=len)


def unique_sorted(values: Iterable[str]) -> list[str]:
    deduped = {value.strip() for value in values if value and value.strip()}
    return sorted(deduped, key=lambda value: (normalize_name(value), value))


def merge_edges(graphs: list[TextbookKnowledgeGraph], node_id_map: dict[str, str]) -> list[KnowledgeEdge]:
    edge_groups: dict[tuple[str, str, str], list[KnowledgeEdge]] = defaultdict(list)

    for graph in graphs:
        for edge in graph.edges:
            source = node_id_map.get(f"{graph.textbook_id}:{edge.source}")
            target = node_id_map.get(f"{graph.textbook_id}:{edge.target}")
            if not source or not target or source == target:
                continue
            relation = edge.relation or "related"
            key = tuple(sorted([source, target]) + [relation])
            edge_groups[key].append(edge)

    merged_edges: list[KnowledgeEdge] = []
    for index, ((source, target, relation), edges) in enumerate(sorted(edge_groups.items()), start=1):
        merged_edges.append(
            KnowledgeEdge(
                edge_id=f"kg_edge_{index:04d}",
                source=source,
                target=target,
                relation=relation,
                weight=round(sum(edge.weight for edge in edges) / len(edges), 4),
                evidence=unique_sorted(evidence for edge in edges for evidence in edge.evidence),
                metadata={"source_edge_count": len(edges)},
            )
        )
    return merged_edges


def enforce_char_budget(
    merged_graph: IntegratedKnowledgeGraph,
    decisions: list[IntegrationDecision],
    source_refs: list[NodeRef],
) -> None:
    source_chars = sum(node_char_count(ref.node) for ref in source_refs)
    if source_chars <= 0:
        return
    budget = max(1, int(source_chars * CHAR_BUDGET_RATIO))
    if graph_char_count(merged_graph.nodes) <= budget:
        return

    edge_degree = node_edge_degree(merged_graph.edges)
    ranked = sorted(
        merged_graph.nodes,
        key=lambda node: node_importance(node, edge_degree),
        reverse=True,
    )
    kept: list[KnowledgeNode] = []
    removed: list[KnowledgeNode] = []
    used_chars = 0

    for node in ranked:
        char_count = node_char_count(node)
        if used_chars + char_count <= budget:
            kept.append(node)
            used_chars += char_count
            continue

        remaining = budget - used_chars
        if remaining > len(node.name) + 36:
            trimmed = copy.deepcopy(node)
            available_definition_chars = max(0, remaining - len(trimmed.name) - alias_char_count(trimmed.aliases) - 4)
            if trimmed.definition and available_definition_chars > 20:
                trimmed.definition = trimmed.definition[:available_definition_chars].rstrip() + "..."
            kept.append(trimmed)
            used_chars += node_char_count(trimmed)
        else:
            removed.append(node)

    kept_ids = {node.node_id for node in kept}
    merged_graph.nodes = kept
    merged_graph.edges = [
        edge for edge in merged_graph.edges if edge.source in kept_ids and edge.target in kept_ids
    ]

    start_index = len(decisions) + 1
    for index, node in enumerate(removed, start=start_index):
        source_ids = [
            f"{source.get('textbook_id')}:{source.get('node_id')}"
            for source in node.metadata.get("sources", [])
            if isinstance(source, dict) and source.get("textbook_id") and source.get("node_id")
        ] or [node.node_id]
        decisions.append(
            IntegrationDecision(
                decision_id=f"decision_budget_remove_{index:03d}",
                action="remove",
                affected_nodes=source_ids,
                result_node=None,
                decision="remove",
                node_ids=source_ids,
                canonical_name=node.name,
                score=0.72,
                confidence=0.72,
                reason="为满足整合后内容字数不超过原始字数 30% 的压缩预算，删除低优先级冗余节点。",
            )
        )


def node_edge_degree(edges: list[KnowledgeEdge]) -> dict[str, int]:
    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    return degree


def node_importance(node: KnowledgeNode, edge_degree: dict[str, int]) -> tuple[int, int, int, int]:
    level_bonus = 3 if node.metadata.get("level") == "primary" else 0
    source_count = int(node.metadata.get("source_count", 1) or 1)
    degree = edge_degree.get(node.node_id, 0)
    definition_bonus = 1 if node.definition else 0
    return (source_count, level_bonus, degree, definition_bonus)


def graph_char_count(nodes: list[KnowledgeNode]) -> int:
    return sum(node_char_count(node) for node in nodes)


def node_char_count(node: KnowledgeNode) -> int:
    return len(node.name or "") + len(node.definition or "") + alias_char_count(node.aliases)


def alias_char_count(aliases: list[str]) -> int:
    return sum(len(alias) for alias in aliases)


def best_group_similarity(group: list[NodeRef], embedding_map: dict[str, list[float]]) -> SimilarityBreakdown:
    best: Optional[SimilarityBreakdown] = None
    for left_index, left in enumerate(group):
        for right in group[left_index + 1 :]:
            similarity = compare_refs(left, right, embedding_map)
            if best is None or similarity.final_score > best.final_score:
                best = similarity
    return best or SimilarityBreakdown(
        normalized_name=0,
        token_jaccard=0,
        definition_jaccard=0,
        embedding_similarity=0,
        final_score=0,
    )


def best_similarity_against_group(
    ref: NodeRef,
    candidates: list[NodeRef],
    embedding_map: dict[str, list[float]],
) -> float:
    if not candidates:
        return 0.0
    return max(compare_refs(ref, candidate, embedding_map).final_score for candidate in candidates)


def build_stats(
    graphs: list[TextbookKnowledgeGraph],
    merged_graph: IntegratedKnowledgeGraph,
    decisions: list[IntegrationDecision],
) -> CompressionStats:
    source_node_count = sum(len(graph.nodes) for graph in graphs)
    source_edge_count = sum(len(graph.edges) for graph in graphs)
    source_char_count = sum(node_char_count(node) for graph in graphs for node in graph.nodes)
    merged_char_count = graph_char_count(merged_graph.nodes)
    char_budget = max(1, int(source_char_count * CHAR_BUDGET_RATIO)) if source_char_count else 0
    merged_node_count = len(merged_graph.nodes)
    removed_node_count = sum(1 for decision in decisions if decision.decision == "remove")
    merge_group_count = sum(1 for decision in decisions if decision.decision == "merge")
    return CompressionStats(
        source_graph_count=len(graphs),
        source_node_count=source_node_count,
        source_edge_count=source_edge_count,
        merged_node_count=merged_node_count,
        merged_edge_count=len(merged_graph.edges),
        removed_node_count=removed_node_count,
        merge_group_count=merge_group_count,
        compression_ratio=round(merged_node_count / source_node_count, 4) if source_node_count else 1.0,
        node_reduction_ratio=round((source_node_count - merged_node_count) / source_node_count, 4)
        if source_node_count
        else 0.0,
        source_char_count=source_char_count,
        merged_char_count=merged_char_count,
        char_compression_ratio=round(merged_char_count / source_char_count, 4) if source_char_count else 1.0,
        char_budget=char_budget,
        budget_met=merged_char_count <= char_budget if char_budget else True,
    )


def split_feedback_sentences(feedback_text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"[\n。；;]+", feedback_text or "")
        if sentence and sentence.strip()
    ]


def detect_feedback_action(sentence: str) -> str:
    if re.search(r"不要合并|保留|留下|单独", sentence):
        return "keep"
    if re.search(r"删除|移除|去掉|不要", sentence):
        return "remove"
    if re.search(r"拆分|拆开|拆成|分成|分为", sentence):
        return "split"
    if re.search(r"合并|并入|归并", sentence):
        return "merge"
    return "unknown"


def apply_remove_feedback(result: GraphIntegrationResult, sentence: str) -> Optional[TeacherFeedbackChange]:
    matches = find_nodes_by_sentence(result, sentence)
    if not matches:
        return None

    matched_ids = {node.node_id for node in matches}
    result.merged_graph.nodes = [node for node in result.merged_graph.nodes if node.node_id not in matched_ids]
    result.merged_graph.edges = [
        edge for edge in result.merged_graph.edges if edge.source not in matched_ids and edge.target not in matched_ids
    ]
    for node in matches:
        result.decisions.append(
            IntegrationDecision(
                decision_id=next_feedback_decision_id(result),
                action="remove",
                affected_nodes=[node.node_id],
                result_node=None,
                decision="remove",
                node_ids=[node.node_id],
                canonical_name=node.name,
                score=1.0,
                confidence=1.0,
                reason=f"教师反馈删除：{sentence}",
            )
        )
    return TeacherFeedbackChange(
        action="remove",
        target=sentence,
        affected_node_ids=sorted(matched_ids),
        note="已删除匹配节点及相关边。",
    )


def apply_keep_feedback(result: GraphIntegrationResult, sentence: str) -> Optional[TeacherFeedbackChange]:
    matches = find_nodes_by_sentence(result, sentence)
    if not matches:
        return None

    for node in matches:
        node.metadata["teacher_feedback"] = "keep"
        result.decisions.append(
            IntegrationDecision(
                decision_id=next_feedback_decision_id(result),
                action="keep",
                affected_nodes=[node.node_id],
                result_node=node.node_id,
                decision="keep",
                node_ids=[node.node_id],
                canonical_name=node.name,
                score=1.0,
                confidence=1.0,
                reason=f"教师反馈保留：{sentence}",
            )
        )
    return TeacherFeedbackChange(
        action="keep",
        target=sentence,
        affected_node_ids=[node.node_id for node in matches],
        note="已标记为教师要求保留。",
    )


def apply_split_feedback(result: GraphIntegrationResult, sentence: str) -> Optional[TeacherFeedbackChange]:
    matches = find_nodes_by_sentence(result, sentence)
    if not matches:
        return None

    affected: list[str] = []
    for node in matches:
        parts = infer_split_parts(sentence, node.name)
        if len(parts) < 2:
            parts = split_alias_parts(node)
        if len(parts) < 2:
            continue

        result.merged_graph.nodes = [candidate for candidate in result.merged_graph.nodes if candidate.node_id != node.node_id]
        created_ids: list[str] = []
        for part in parts:
            new_id = next_node_id(result)
            created_ids.append(new_id)
            affected.append(new_id)
            result.merged_graph.nodes.append(
                KnowledgeNode(
                    node_id=new_id,
                    name=part,
                    definition=node.definition,
                    aliases=[],
                    node_type=node.node_type,
                    metadata={**node.metadata, "teacher_feedback": "split", "split_from": node.node_id},
                )
            )
        rewire_split_edges(result, node.node_id, created_ids)
        result.decisions.append(
            IntegrationDecision(
                decision_id=next_feedback_decision_id(result),
                action="keep",
                affected_nodes=[node.node_id, *created_ids],
                result_node=None,
                decision="keep",
                node_ids=created_ids,
                canonical_name=" / ".join(parts),
                score=1.0,
                confidence=1.0,
                reason=f"教师反馈拆分：{sentence}",
            )
        )

    if not affected:
        return None
    return TeacherFeedbackChange(
        action="split",
        target=sentence,
        affected_node_ids=affected,
        note="已将匹配节点拆分为多个独立节点。",
    )


def apply_merge_feedback(result: GraphIntegrationResult, sentence: str) -> Optional[TeacherFeedbackChange]:
    matches = find_nodes_by_sentence(result, sentence)
    if len(matches) < 2:
        explicit_names = extract_quoted_or_named_targets(sentence)
        matches = [node for name in explicit_names for node in find_nodes_by_name(result, name)]
    if len(matches) < 2:
        return None

    base = matches[0]
    merged_id = base.node_id
    merged_names = [node.name for node in matches]
    base.aliases = unique_sorted([*base.aliases, *merged_names[1:], *(alias for node in matches[1:] for alias in node.aliases)])
    base.definition = select_definition([node.definition or "" for node in matches])
    base.metadata["teacher_feedback"] = "merge"
    base.metadata["merged_by_teacher"] = [node.node_id for node in matches]

    removed_ids = {node.node_id for node in matches[1:]}
    result.merged_graph.nodes = [node for node in result.merged_graph.nodes if node.node_id not in removed_ids]
    for edge in result.merged_graph.edges:
        if edge.source in removed_ids:
            edge.source = merged_id
        if edge.target in removed_ids:
            edge.target = merged_id
    dedupe_feedback_edges(result)
    result.decisions.append(
        IntegrationDecision(
            decision_id=next_feedback_decision_id(result),
            action="merge",
            affected_nodes=[node.node_id for node in matches],
            result_node=merged_id,
            decision="merge",
            node_ids=[node.node_id for node in matches],
            canonical_name=base.name,
            score=1.0,
            confidence=1.0,
            reason=f"教师反馈合并：{sentence}",
        )
    )
    return TeacherFeedbackChange(
        action="merge",
        target=sentence,
        affected_node_ids=[node.node_id for node in matches],
        note="已按教师反馈合并匹配节点。",
    )


def find_nodes_by_sentence(result: GraphIntegrationResult, sentence: str) -> list[KnowledgeNode]:
    targets = extract_quoted_or_named_targets(sentence)
    matched: list[KnowledgeNode] = []
    for target in targets:
        matched.extend(find_nodes_by_name(result, target))
    if matched:
        return dedupe_nodes(matched)

    normalized_sentence = normalize_name(sentence)
    return [
        node
        for node in result.merged_graph.nodes
        if normalize_name(node.name) and normalize_name(node.name) in normalized_sentence
    ]


def find_nodes_by_name(result: GraphIntegrationResult, name: str) -> list[KnowledgeNode]:
    target = normalize_name(name)
    if not target:
        return []
    exact_matches = [
        node
        for node in result.merged_graph.nodes
        if target == normalize_name(node.name)
        or target in {normalize_name(alias) for alias in node.aliases}
    ]
    if exact_matches:
        return exact_matches

    return [
        node
        for node in result.merged_graph.nodes
        if normalize_name(node.name) in target or target in normalize_name(node.name)
    ]


def extract_quoted_or_named_targets(sentence: str) -> list[str]:
    quoted = re.findall(r"[“\"'《]([^”\"'》]{1,40})[”\"'》]", sentence)
    if quoted:
        return [target.strip() for target in quoted if target.strip()]

    cleaned = re.sub(
        r"为什么|为何|原因|解释|依据|觉得|认为|应该|不应该|它们|他们|不是|同一个|这个|那个|请|把|将|节点|概念|知识点|保留|删除|移除|去掉|拆分|拆开|拆成|分成|分为|合并|并入|归并|不要|单独|了|吗|呢",
        " ",
        sentence,
    )
    candidates = [
        part.strip(" ：:，,。？！?；;")
        for part in SPLIT_SEPARATORS_RE.split(cleaned)
        if part.strip(" ：:，,。？！?；;")
    ]
    return [candidate for candidate in candidates if 1 <= len(candidate) <= 40]


def infer_split_parts(sentence: str, fallback: str) -> list[str]:
    match = re.search(r"(?:拆成|分成|分为)\s*(.+)$", sentence)
    if not match:
        return []
    raw_parts = [part.strip(" ：:，,。") for part in SPLIT_SEPARATORS_RE.split(match.group(1)) if part.strip()]
    return [part for part in raw_parts if normalize_name(part) and normalize_name(part) != normalize_name(fallback)]


def split_alias_parts(node: KnowledgeNode) -> list[str]:
    parts = [node.name, *node.aliases]
    return unique_sorted(part for part in parts if normalize_name(part) != normalize_name(node.name) or len(parts) > 1)


def rewire_split_edges(result: GraphIntegrationResult, old_id: str, new_ids: list[str]) -> None:
    new_edges: list[KnowledgeEdge] = []
    for edge in result.merged_graph.edges:
        if edge.source == old_id:
            for new_id in new_ids:
                clone = copy.deepcopy(edge)
                clone.edge_id = next_edge_id(result, new_edges)
                clone.source = new_id
                new_edges.append(clone)
        elif edge.target == old_id:
            for new_id in new_ids:
                clone = copy.deepcopy(edge)
                clone.edge_id = next_edge_id(result, new_edges)
                clone.target = new_id
                new_edges.append(clone)
        else:
            new_edges.append(edge)
    result.merged_graph.edges = new_edges
    dedupe_feedback_edges(result)


def dedupe_feedback_edges(result: GraphIntegrationResult) -> None:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[KnowledgeEdge] = []
    for edge in result.merged_graph.edges:
        if edge.source == edge.target:
            continue
        key = tuple(sorted([edge.source, edge.target]) + [edge.relation])
        if key in seen:
            continue
        seen.add(key)
        edge.edge_id = f"kg_edge_{len(deduped) + 1:04d}"
        deduped.append(edge)
    result.merged_graph.edges = deduped


def refresh_stats_after_feedback(result: GraphIntegrationResult) -> None:
    source_node_count = result.stats.source_node_count
    result.stats.merged_node_count = len(result.merged_graph.nodes)
    result.stats.merged_edge_count = len(result.merged_graph.edges)
    result.stats.removed_node_count = sum(1 for decision in result.decisions if decision.decision == "remove")
    result.stats.merge_group_count = sum(1 for decision in result.decisions if decision.decision == "merge")
    result.stats.compression_ratio = round(len(result.merged_graph.nodes) / source_node_count, 4) if source_node_count else 1.0
    result.stats.node_reduction_ratio = (
        round((source_node_count - len(result.merged_graph.nodes)) / source_node_count, 4)
        if source_node_count
        else 0.0
    )
    result.stats.merged_char_count = graph_char_count(result.merged_graph.nodes)
    result.stats.char_compression_ratio = (
        round(result.stats.merged_char_count / result.stats.source_char_count, 4)
        if result.stats.source_char_count
        else 1.0
    )
    result.stats.budget_met = result.stats.merged_char_count <= result.stats.char_budget if result.stats.char_budget else True


def dedupe_nodes(nodes: list[KnowledgeNode]) -> list[KnowledgeNode]:
    by_id: dict[str, KnowledgeNode] = {}
    for node in nodes:
        by_id[node.node_id] = node
    return list(by_id.values())


def next_feedback_decision_id(result: GraphIntegrationResult) -> str:
    return f"decision_feedback_{len(result.decisions) + 1:03d}"


def next_node_id(result: GraphIntegrationResult) -> str:
    max_index = 0
    for node in result.merged_graph.nodes:
        match = re.search(r"(\d+)$", node.node_id)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"kg_node_{max_index + 1:04d}"


def next_edge_id(result: GraphIntegrationResult, pending_edges: Optional[list[KnowledgeEdge]] = None) -> str:
    max_index = 0
    for edge in [*result.merged_graph.edges, *(pending_edges or [])]:
        match = re.search(r"(\d+)$", edge.edge_id)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"kg_edge_{max_index + 1:04d}"


def qualified_node_id(graph: TextbookKnowledgeGraph, node: KnowledgeNode) -> str:
    return f"{graph.textbook_id}:{node.node_id}"
