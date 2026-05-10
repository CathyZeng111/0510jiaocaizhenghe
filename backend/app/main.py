from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import app_state
from .graph_integration import apply_teacher_feedback, integrate_textbook_graphs
from .integration_schemas import GraphIntegrationResult, TextbookKnowledgeGraph, TeacherFeedbackResult
from .knowledge_graph import build_all_graphs
from .modelscope_client import is_modelscope_configured
from .parsers import SUPPORTED_EXTENSIONS, parse_file_path, parse_upload
from .rag_eval import router as rag_eval_router
from .rag import router as rag_router
from .schemas import UploadResponse, UploadResult
from .textbook_store import textbook_store

app = FastAPI(title="AI Textbook Integrator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)
app.include_router(rag_eval_router)


class GraphBuildRequest(BaseModel):
    textbook_ids: Optional[list[str]] = None
    use_llm: bool = False


class GraphBuildResponse(BaseModel):
    graphs: list[TextbookKnowledgeGraph]


class IntegrationBuildRequest(BaseModel):
    graphs: Optional[list[TextbookKnowledgeGraph]] = None


class FeedbackRequest(BaseModel):
    feedback: str


@app.get("/api/health")
def health_check() -> dict[str, object]:
    return {"status": "ok", "modelscope_configured": is_modelscope_configured()}


@app.post("/api/textbooks/upload", response_model=UploadResponse)
async def upload_textbooks(files: list[UploadFile] = File(...)) -> UploadResponse:
    results: list[UploadResult] = []

    for index, file in enumerate(files, start=1):
        result = await parse_upload(file, index)
        results.append(result)

    return UploadResponse(results=results)


@app.get("/api/textbooks", response_model=list[UploadResult])
def list_textbooks() -> list[UploadResult]:
    return [
        UploadResult(
            filename=record.textbook.filename,
            format=Path(record.textbook.filename).suffix.lower().lstrip(".") or "unknown",
            size=0,
            status="completed",
            textbook=record.textbook,
        )
        for record in textbook_store.list_records()
    ]


@app.post("/api/textbooks/load-local", response_model=UploadResponse)
def load_local_textbooks() -> UploadResponse:
    textbook_dir = Path.cwd() / "textbooks"
    if not textbook_dir.exists():
        raise HTTPException(status_code=404, detail="当前项目目录下未找到 textbooks 文件夹")

    files = sorted(
        path
        for path in textbook_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise HTTPException(status_code=404, detail="textbooks 文件夹中没有支持的教材文件")

    textbook_store.clear()
    app_state.graph_cache.clear()
    app_state.latest_integration = None

    results: list[UploadResult] = []
    for index, path in enumerate(files, start=1):
        try:
            textbook = parse_file_path(path, path.name, f"book_{index:02d}")
            result = UploadResult(
                filename=path.name,
                format=path.suffix.lower().lstrip("."),
                size=path.stat().st_size,
                status="completed",
                textbook=textbook,
            )
            results.append(textbook_store.cache_upload_result(result))
        except Exception as exc:
            results.append(
                UploadResult(
                    filename=path.name,
                    format=path.suffix.lower().lstrip("."),
                    size=path.stat().st_size,
                    status="failed",
                    error=str(exc),
                )
            )

    return UploadResponse(results=results)


@app.post("/api/graphs/build", response_model=GraphBuildResponse)
def build_graphs(request: GraphBuildRequest) -> GraphBuildResponse:
    requested_ids = request.textbook_ids or [book.textbook_id for book in textbook_store.list_textbooks()]
    textbooks = [textbook_store.get_textbook(textbook_id) for textbook_id in requested_ids]
    missing = [textbook_id for textbook_id, textbook in zip(requested_ids, textbooks) if textbook is None]
    if missing:
        raise HTTPException(status_code=404, detail=f"未找到教材：{', '.join(missing)}")

    graphs = build_all_graphs([textbook for textbook in textbooks if textbook is not None], use_llm=request.use_llm)
    for graph in graphs:
        app_state.graph_cache[graph.textbook_id] = graph
    return GraphBuildResponse(graphs=graphs)


@app.get("/api/graphs/{textbook_id}", response_model=TextbookKnowledgeGraph)
def get_graph(textbook_id: str) -> TextbookKnowledgeGraph:
    graph = app_state.graph_cache.get(textbook_id)
    if graph is None:
        textbook = textbook_store.get_textbook(textbook_id)
        if textbook is None:
            raise HTTPException(status_code=404, detail=f"未找到教材：{textbook_id}")
        graph = build_all_graphs([textbook])[0]
        app_state.graph_cache[textbook_id] = graph
    return graph


@app.post("/api/integration/build", response_model=GraphIntegrationResult)
def build_integration(request: IntegrationBuildRequest) -> GraphIntegrationResult:
    graphs = request.graphs or list(app_state.graph_cache.values())
    if len(graphs) < 2:
        textbooks = textbook_store.list_textbooks()
        if len(textbooks) >= 2:
            graphs = build_all_graphs(textbooks, use_llm=False)
            for graph in graphs:
                app_state.graph_cache[graph.textbook_id] = graph
    if len(graphs) < 2:
        raise HTTPException(status_code=400, detail="至少需要 2 本教材图谱才能进行跨教材整合")

    app_state.latest_integration = integrate_textbook_graphs(graphs)
    return app_state.latest_integration


@app.post("/api/integration/feedback", response_model=TeacherFeedbackResult)
def apply_integration_feedback(request: FeedbackRequest) -> TeacherFeedbackResult:
    if app_state.latest_integration is None:
        raise HTTPException(status_code=400, detail="请先执行跨教材整合")
    feedback_result = apply_teacher_feedback(app_state.latest_integration, request.feedback)
    app_state.latest_integration = feedback_result.result
    return feedback_result
