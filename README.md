# 学科知识整合智能体

当前阶段完成 P0 的教材加载解析，并加入单本教材知识图谱构建与可视化。

## 功能

- 支持批量上传 `PDF`、`Markdown`、`TXT`、`DOCX`
- 上传后显示文件名、格式、大小、解析状态和错误信息
- 将教材解析为统一结构：教材元数据、章节列表、页码范围、正文和字数
- PDF 使用 PyMuPDF 逐页解析，包含基础章节识别、页眉页脚过滤和文本块提取
- 支持选择单本教材生成知识图谱 JSON，节点包含 `id/name/definition/category/chapter/page`
- 支持 `contains`、`prerequisite`、`parallel`、`applies_to` 关系，并以一级章节优先、点击展开二级知识点的方式可视化
- RAG Pipeline：650 字滑窗分块、80 字重叠、OpenRouter 中文 embedding、Chroma 持久向量库、top5 向量检索、带教材/章节/页码引用的 LLM 回答

## 环境依赖

- Python 3.9+
- Node.js 20+

## 安装

```bash
python3 -m pip install -r requirements.txt
npm install --prefix frontend
```

## 配置 ModelScope（可选）

如需启用 ModelScope 模型增强知识点抽取和 RAG 生成回答，请设置环境变量或复制 `.env.example` 为 `.env` 后填写：

```bash
export MODELSCOPE_API_KEY="你的 ModelScope API Key"
export MODELSCOPE_BASE_URL="http://api-inference.modelscope.cn/v1"
export MODELSCOPE_MODEL="deepseek-ai/DeepSeek-V4-Flash"
export EMBEDDING_PROVIDER="openrouter"
export OPENROUTER_API_KEY="你的 OpenRouter API Key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_EMBEDDING_MODEL="qwen/qwen3-embedding-8b"
```

不要把真实 API Key 提交到 GitHub。项目提供了 `.env.example` 作为配置模板。

## 启动

后端：

```bash
uvicorn backend.app.main:app --reload --port 8000
```

前端：

```bash
npm run dev
```

打开 `http://localhost:5173` 使用上传解析界面。

## API

### `POST /api/textbooks/upload`

请求类型：`multipart/form-data`

字段：

- `files`: 一个或多个教材文件

返回：

```json
{
  "results": [
    {
      "filename": "生理学.pdf",
      "format": "pdf",
      "size": 1024,
      "status": "completed",
      "error": null,
      "textbook": {
        "textbook_id": "book_01",
        "filename": "生理学.pdf",
        "title": "生理学",
        "total_pages": 520,
        "total_chars": 385000,
        "chapters": [
          {
            "chapter_id": "ch_01",
            "title": "第一章 绪论",
            "page_start": 1,
            "page_end": 15,
            "content": "生理学是研究生物体正常生命活动规律的科学...",
            "char_count": 8500
          }
        ]
      }
    }
  ]
}
```

### `POST /api/graphs/build`

请求类型：`application/json`

字段：

- `textbook_ids`: 要生成图谱的教材 ID 列表
- `use_llm`: 是否按章节调用 ModelScope LLM 抽取核心知识点

返回：

```json
{
  "graphs": [
    {
      "graph_id": "graph_book_01",
      "textbook_id": "book_01",
      "textbook_title": "生理学",
      "nodes": [
        {
          "node_id": "node_001",
          "name": "动作电位",
          "definition": "细胞受到刺激后，膜电位发生的一次快速而可逆的倒转。",
          "metadata": {
            "category": "核心概念",
            "chapter_title": "第二章 细胞的基本功能",
            "page_start": 35
          }
        }
      ],
      "edges": [
        {
          "source": "node_001",
          "target": "node_002",
          "relation": "prerequisite",
          "metadata": {
            "description": "理解动作电位需要先掌握静息电位的概念。"
          }
        }
      ]
    }
  ]
}
```
