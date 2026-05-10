from __future__ import annotations

from typing import Optional

from .integration_schemas import GraphIntegrationResult, IntegrationChatMessage, TextbookKnowledgeGraph

graph_cache: dict[str, TextbookKnowledgeGraph] = {}
latest_integration: Optional[GraphIntegrationResult] = None
integration_chat_history: dict[str, list[IntegrationChatMessage]] = {}
