# 功能 6：RAG 评测与数据驱动优化

## 目标

补齐一条可复验的 RAG benchmark 闭环：

- 自动生成 20-50 道真实教材题目
- 为每题标注标准答案与预期引用来源
- 批量跑不同 RAG 配置
- 统计回答准确率、引用准确率、引用召回率、平均响应时间、Token 消耗
- 根据指标选择更优配置

## 新增接口

- `POST /api/rag-eval/generate`
- `POST /api/rag-eval/run`
- `POST /api/rag-eval/optimize`

对应代码：

- [backend/app/rag_eval.py](../backend/app/rag_eval.py)
- [backend/app/rag_eval_schemas.py](../backend/app/rag_eval_schemas.py)

## 评测集生成策略

1. 自动加载 `textbooks/` 下 7 本教材。
2. 过滤前言、目录、编委、主编简介、封面等低信噪比章节。
3. 对有效章节做切片，保留教材、章节、页码元数据。
4. 用 LLM 按四类题型自动出题：
   `factual`、`reasoning`、`comparative`、`cross_textbook`
5. 保存 `question`、`expected_answer`、`expected_citations`。

当前已生成评测集：

- [artifacts/rag_eval_dataset.json](../artifacts/rag_eval_dataset.json)

## 当前数据集概况

数据集 `rag_eval_20260510_054323`：

- 题目总数：24
- 题型覆盖：`factual 8`、`reasoning 6`、`comparative 5`、`cross_textbook 5`
- 难度覆盖：`easy 8`、`medium 11`、`hard 5`

教材覆盖：

- `01_局部解剖学`: 12
- `02_组织学与胚胎学`: 6
- `03_生理学`: 4
- `04_医学微生物学`: 2
- `05_病理学`: 5
- `06_传染病学`: 1
- `07_病理生理学`: 6

## 指标定义

- `answer_accuracy`
  当前默认使用 lexical judge，比对标准答案和模型答案的关键词覆盖情况。
- `citation_accuracy`
  预测引用中，有多少比例真正支撑了题目答案。
- `citation_recall`
  预期来源中，有多少被预测引用覆盖到。
- `exact_citation_hit_rate`
  每题是否完整覆盖预期支撑来源。
- `avg_response_time_ms`
  平均单题响应时间。
- `token_usage`
  当前按输入输出文本长度估算。

## 一次实际优化结果

我用 24 道真实教材题，对 3 组配置做了自动评测：

| Config | answer_accuracy | citation_accuracy | citation_recall | exact_citation_hit_rate | avg_response_time_ms | estimated_total_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `keyword_650` | 0.0252 | 0.1917 | 0.5208 | 0.4583 | 3016.08 | 1830 |
| `hybrid_650` | 0.0243 | 0.1917 | 0.5208 | 0.5000 | 2889.43 | 1398 |
| `keyword_560` | 0.0177 | 0.2000 | 0.5208 | 0.5000 | 3123.44 | 1290 |

当前推荐配置：

- `retrieval_mode = hybrid`
- `chunk_size = 650`
- `chunk_overlap = 80`
- `top_k = 5`
- `min_score = 0.0`
- `answer_mode = extractive`

## 结果解读

这轮 benchmark 给出两个明显信号：

1. 检索已经能较稳定地找到支撑来源。
   `citation_recall` 超过 0.52，说明问题通常能把相关片段捞出来。

2. 当前主要短板在答案组织，而不是找不到内容。
   `answer_accuracy` 明显低于引用指标，说明下一步更值得优化的是生成与多片段融合。

## 使用方式

生成题集：

```bash
curl -X POST http://localhost:8000/api/rag-eval/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "question_count": 24,
    "use_llm": true,
    "persist_path": "artifacts/rag_eval_dataset.json"
  }'
```

执行评测：

```bash
curl -X POST http://localhost:8000/api/rag-eval/run \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "rag_eval_20260510_054323"
  }'
```

执行配置优化：

```bash
curl -X POST http://localhost:8000/api/rag-eval/optimize \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_id": "rag_eval_20260510_054323"
  }'
```
