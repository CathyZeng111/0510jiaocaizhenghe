# 功能 5：完整 RAG Pipeline

## 目标

本功能在已解析教材结构上实现完整 RAG 问答链路：

1. 文档分块
2. 向量嵌入
3. 向量存储与混合检索
4. 基于检索上下文生成带来源引用的回答

当知识库没有命中相关内容时，固定返回：

```text
当前知识库中未找到相关信息
```

## Step 1：文档分块

默认参数：

- `chunk_size = 650`，接口限制为 500-800 字。
- `chunk_overlap = 80`，接口限制为 50-100 字。
- 分块策略：sliding window，并尽量在句号、问号、分号或换行附近切断。

选择 650 字和 80 字重叠的理由：

- 医学教材中一个概念定义、机制解释或临床说明通常在数百字内可以完整表达，650 字足够容纳一个局部知识单元。
- 500-800 字适合塞入 LLM 上下文，不会让 top-5 chunk 过长。
- 80 字重叠可以降低“定义在上一块、解释在下一块”造成的截断风险。

每个 chunk 保留元数据：

- 教材：`textbook_id`、`textbook_title`、`filename`
- 章节：`chapter_id`、`chapter_title`
- 页码：`page_start`、`page_end`
- 分块：`chunk_index`、`char_start`、`char_end`
- 参数：`chunk_size`、`chunk_overlap`
- 检索：`retrieval_model`、`embedding_model`

## Step 2：向量嵌入

索引阶段会将每个 chunk 发送到 embedding 模型，当前默认：

```text
OpenRouter: qwen/qwen3-embedding-8b
```

该模型支持中文教材语义表示。若 embedding 调用失败，系统不会中断，会降级为关键词/字符 ngram 检索，并在响应中体现 `embedding_count = 0`。

## Step 3：向量存储与检索

当前实现使用 Chroma 持久向量库：

```text
vector_store_type = chroma
```

索引阶段会把所有 chunk embedding、原文和元数据写入 `data/chroma`。用户提问时：

1. 将问题转成向量。
2. 在向量库中检索相似 chunk。
3. 取 top-5 作为默认返回数量。

检索方式：

- 将用户问题向量化。
- 调用 Chroma collection 的向量相似度检索。
- 返回最相关的 top-5 chunk。

## Step 4：生成回答

回答阶段将 top-5 chunk 注入 LLM prompt。Prompt 明确约束：

- 只基于给定上下文回答，不使用自身知识。
- 每个关键结论后必须添加来源引用。
- 引用格式为 `[教材名称, 章节标题, 第 X 页]`。
- 如果上下文中找不到答案，只回答 `当前知识库中未找到相关信息`。

当前 LLM：

```text
ModelScope: deepseek-ai/DeepSeek-V4-Flash
```

## API

- `POST /api/rag/index`：建立 RAG 索引。
- `POST /api/rag/query`：检索并回答。

`/api/rag/index` 返回：

```json
{
  "index_id": "default",
  "textbook_count": 1,
  "chapter_count": 10,
  "chunk_count": 120,
  "chunk_size": 650,
  "chunk_overlap": 80,
  "embedding_model": "qwen/qwen3-embedding-8b",
  "embedding_count": 120,
  "vector_store_type": "chroma",
  "chunking_strategy": "sliding_window_500_800_overlap_50_100"
}
```

`/api/rag/query` 返回：

```json
{
  "answer": "动作电位依赖离子通道开放。[生理学, 第二章 细胞的基本功能, 第 35 页]",
  "citations": [],
  "source_chunks": [],
  "retrieval": {
    "index_id": "default",
    "retrieval_mode": "vector",
    "retrieval_model": "chroma_vector_similarity_v1",
    "embedding_model": "qwen/qwen3-embedding-8b",
    "embedding_count": 120,
    "vector_store_type": "chroma",
    "chunking_strategy": "sliding_window_500_800_overlap_50_100",
    "top_k": 5,
    "min_score": 0.08,
    "matched_chunks": 5
  }
}
```
