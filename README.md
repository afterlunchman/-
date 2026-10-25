# 校园安全制度问答助手

《自然语言处理》课程项目 · 方向 A 领域 RAG 问答  
语料：东北大学安全管理委员会 155 份制度（145 万字，106 个附件）

## 当前进度

- [x] 仓库初始化、目录骨架、配置加载
- [x] 语料解析：155/155 份，0 缺失
- [x] 条文级切块 + 父级扩展接口：4,976 块（条文 3,780 / 段落 1,196）
- [x] BM25、Word2Vec 平均、BGE dense、hybrid RRF、CrossEncoder rerank
- [x] RAG 生成、拒答、规则引用校验、Streamlit 界面
- [x] 80 题评测集草稿（70 可答 + 10 不可答，`annotator=ai-draft`）
- [x] 检索层评测：70 道可答题，五条检索链路
- [x] 答案层评测：B0 闭卷、B1 BM25、B3 BGE、B5 hybrid+rerank
- [x] rerank 证据：B5 答案评测 20 题 Hit@5 = 0.950（全量 70 题因 CPU 成本未跑）
- [x] 24 题人工抽检表已生成（`results/human_check.csv`，待组员评分）
- [x] 项目报告（Word + PDF）、项目摘要、4 份贡献声明、AI 使用声明
- [ ] 填写姓名、学号与工作量占比，签署贡献声明
- [ ] 完成 24 题人工评分，录制 2 分钟演示视频并上传网盘

语料统计：103 份识别出条号；52 份检查规范表、预案与 OCR 噪声较重的长规程回退为段落切分。

## 提交材料

提交压缩包内按交付物分文件夹整理，完整说明见压缩包根目录的
`00_提交说明/提交说明与检查清单.md`：

```text
./
├── 00_提交说明/          # 提交清单、待填写信息
├── 01_项目报告/          # Word + PDF
├── 02_项目摘要/          # 项目摘要.md + .docx（297 字）
├── 03_源码与README/      # 源码、依赖、评测集、结果、语料正文
├── 04_个人贡献声明/      # 4 人各一份 Word
├── 05_AI使用声明/        # Word
└── 06_演示录屏/          # 演示脚本、链接占位
```

语料正文约 4.1 MB，为保证一键复现随包提供；142 MB 附件 PDF 不打包，只保留官网链接。
重新生成文档与压缩包（仓库内开发脚本）：

```bash
.venv/Scripts/python.exe scripts/build_report_docx.py
.venv/Scripts/python.exe scripts/build_other_docs.py
.venv/Scripts/python.exe scripts/package_submission.py
```


## 检索结果（70 道可答题）

| 检索器 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | 平均延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.7571 | 0.8714 | 0.8857 | 0.9286 | 0.8188 | 10.8 ms |
| Word2Vec 平均 | 0.4429 | 0.6714 | 0.7286 | 0.8143 | 0.5700 | 0.5 ms |
| BGE dense | 0.6714 | 0.8286 | 0.8571 | 0.8857 | 0.7470 | 8.5 ms |
| Hybrid RRF | 0.7571 | 0.8714 | 0.8857 | 0.9143 | 0.8140 | 26.0 ms |
| Hybrid + rerank | 未跑全量 | 未跑全量 | 未跑全量 | 未跑全量 | 未跑全量 | CPU 成本高 |

> rerank 全量 70 题检索评测未完成：CPU 上每个查询约 40 s，70 题超过 50 min。B5 答案评测在同一 20 题上 Hit@5 = 0.950（B1 0.850、B3 0.800），作为重排证据。

初步结论：法条检索中关键词匹配很强，BM25 的 Recall@5 已达 0.886；词向量平均明显落后，符合第 5 讲「词袋丢失语序与否定」的预期；BGE dense 在语义匹配上更好，但 Top-1 仍低于 BM25；hybrid 当前未超过 BM25，重排是下一步主要增益点。

词向量补充证据：OOV 率 1.37%；「宿舍可以留宿外来人员吗」与「宿舍不可以留宿外来人员吗」的句向量余弦为 0.972，Top-5 结果完全相同，说明平均词向量几乎无法区分否定。

## 答案层结果（20 可答 + 10 不可答）

| 方案 | 正确率 | 引用支持率 | 规则引用通过率 | 误拒率 | 不可答拒答率 | Hit@5 | 平均延迟 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 闭卷 | 0.350 | — | — | 0.350 | 0.100 | — | 4.95 s |
| B1 BM25 | 0.800 | 0.895 | 1.000 | 0.050 | 1.000 | 0.850 | 2.28 s |
| B3 BGE | 0.800 | 1.000 | 0.938 | 0.200 | 1.000 | 0.800 | 2.03 s |
| B5 hybrid+rerank | 0.900 | 0.842 | 0.947 | 0.050 | 1.000 | 0.950 | 8.10 s |

结论：RAG 把闭卷正确率从 0.350 提升到 0.800—0.900，并把不可答拒答率从 0.100 提升到 1.000。B3 的引用支持率最高（1.000）但误拒率 0.20；B5 的正确率与 Hit@5 最高（0.900 / 0.950），代价是 8.10 s 延迟与 0.842 的引用支持率。B1 是延迟与质量最均衡的方案。

结果图：`docs/figures/retrieval_recall.png`、`docs/figures/answer_metrics.png`。

## 环境与安装

- Python 3.13（见 `.python-version`）
- [uv](https://docs.astral.sh/uv/) 0.11+

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

若 Windows 上 `jieba` 构建时报 `unicodedata` 文件锁，改用非隔离构建：

```bash
uv pip install --python .venv/Scripts/python.exe setuptools wheel
uv pip install --python .venv/Scripts/python.exe --no-build-isolation jieba
```

## 配置

```bash
cp config.example.yaml config.yaml
```

填写 `llm.base_url`、`llm.api_key`、`llm.model`。`config.yaml` 已被 `.gitignore` 排除，**不要提交或打包**；提交包只放 `config.example.yaml`。

## 常用命令

语料统计：

```bash
.venv/Scripts/python.exe -m scripts.check_corpus
```

生成统一 chunk 文件：

```bash
.venv/Scripts/python.exe -m scripts.build_index
```

检索：

```bash
.venv/Scripts/python.exe -m scripts.search --retriever bm25 --query "宿舍可以使用大功率电器吗" --top-k 5
```

```bash
.venv/Scripts/python.exe -m scripts.search --retriever hybrid --query "实验室事故多久内报告" --top-k 5
```

端到端问答：

```bash
.venv/Scripts/python.exe -m scripts.ask --retriever hybrid --question "宿舍可以使用大功率电器吗" --top-k 5
```

检索层评测：

```bash
.venv/Scripts/python.exe -m scripts.eval_retrieval --retrievers bm25,word2vec,dense,hybrid
```

答案层评测（可断点续跑，结果写 `results/answers_*.jsonl`）：

```bash
.venv/Scripts/python.exe -m scripts.eval_answers --pipelines b0,b1,b3,b5 --answerable 20 --unanswerable 10
```

汇总与绘图：

```bash
.venv/Scripts/python.exe scripts/summarize_results.py
.venv/Scripts/python.exe scripts/plot_results.py
```

Web 界面：

```bash
.venv/Scripts/python.exe -m streamlit run src/app/streamlit_app.py
```

## 向量索引说明

Chroma 1.5.9 在本机 Windows 上二次打开 HNSW 索引时报 `Error loading hnsw index`，因此改用 **numpy 持久化向量索引**：

- `data/index/vectors/embeddings.npy`：4,976 × 512 的 float32 矩阵
- `data/index/vectors/chunk_ids.json`：与行号对应的 chunk id
- `data/index/vectors/meta.json`：模型名、数量、维度

单查询为一次矩阵乘法，CPU 上约 8.5 ms；索引约 10 MB，可整体重建。该实现满足「嵌入式、本地持久化、可复现」，报告「实现细节」一节会记录 Chroma 的踩坑。

## 目录结构

```text
.
├── config.example.yaml        # 配置模板
├── requirements.txt
├── 需求分析.md                 # 需求、指标、实验矩阵、分工
├── 任务清单.md                 # 后续任务与里程碑
├── 学校规章制度/               # 语料（附件 PDF 不入库）
├── src/
│   ├── config.py              # 配置加载 + 环境变量覆盖
│   ├── ingest/                # 解析、条文切块、统计
│   ├── retrieval/             # BM25 / 词向量 / BGE / hybrid / rerank
│   ├── generation/            # API 封装、提示词、引用校验、问答编排
│   ├── evaluation/            # 指标、LLM 评审、批量评测
│   └── app/                   # Streamlit
├── scripts/
│   ├── check_corpus.py        # 语料统计
│   ├── build_index.py         # 导出 chunk
│   ├── search.py              # 检索 CLI
│   ├── ask.py                 # 端到端问答 CLI
│   ├── build_eval_set.py      # 评测集草稿生成
│   ├── eval_retrieval.py      # 检索层评测
│   └── eval_answers.py        # 答案层评测
├── data/                      # 索引与评测集
├── results/                   # 统计与评测结果
└── logs/
```

## 切块口径

- 按「第 X 条」切块，一条一块；条号需连续递增，正文中的「根据第 X 条」等交叉引用会被过滤。
- 单条超过 1,200 字时按窗口切分，保留同一 `clause_no`。
- 无条号或条号无法连成序列的文档，回退为 600 字 / 100 字重叠的段落切块。
- 每个 chunk 带 `doc_id`、`title`、`clause_no`、`page`、`confidence`、`source_url`、`attachment_url`。

## 复现主结果

最短路径只跑检索，不需要 API Key：

```bash
.venv/Scripts/python.exe -m scripts.build_index
.venv/Scripts/python.exe -m scripts.eval_retrieval --retrievers bm25,word2vec,dense,hybrid
```

预期输出写入 `results/retrieval_metrics.json`，与报告表 6 一致：

| 检索器 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.7571 | 0.8714 | 0.8857 | 0.9286 | 0.8188 |
| word2vec | 0.4429 | 0.6714 | 0.7286 | 0.8143 | 0.5700 |
| dense | 0.6714 | 0.8286 | 0.8571 | 0.8857 | 0.7470 |
| hybrid | 0.7571 | 0.8714 | 0.8857 | 0.9143 | 0.8140 |

随机种子：Word2Vec 训练 `seed=42`；检索与答案评测使用同一份 `data/eval/questions.jsonl`；
生成温度固定 0.1。完整答案层复现需要 `config.yaml` 中的 API 配置：

```bash
.venv/Scripts/python.exe -m scripts.eval_answers --pipelines b0,b1,b3,b5 --answerable 20 --unanswerable 10
```

预期输出：`results/answer_metrics.json` 中 B1 正确率 0.800、B3 0.800、B5 0.900；
不可答拒答率均为 1.000。数字与报告表 7 一致。

## 下一步

1. 填写姓名、学号与工作量占比，签署 4 份贡献声明；报告第⑧节同步。
2. 完成 24 题人工抽检，计算与 LLM judge 的一致率。
3. 按《演示脚本.md》录制 2 分钟视频，上传网盘并填写链接。
4. 按 `00_提交说明/提交说明与检查清单.md` 改名、打包、提交。


## 安全

- API Key 只放在 `config.yaml` 或环境变量中，提交前执行 `grep -rn "sk-" .` 自查。
- 只向外部 API 发送公开制度文本，不发送个人信息。
