from __future__ import annotations

from typing import Optional

from .integration_schemas import GraphIntegrationResult, TextbookKnowledgeGraph

graph_cache: dict[str, TextbookKnowledgeGraph] = {}
latest_integration: Optional[GraphIntegrationResult] = None
