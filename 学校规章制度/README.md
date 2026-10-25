# 东北大学安全管理委员会 · 规章制度全量语料库

> 来源站点：**东北大学安全管理委员会办公室**（`http://www.osc.neu.edu.cn`）  
> 抓取范围：**规章制度** 板块 5 个子板块及其全部可展开子栏目  
> 抓取快照：2026-10-24 ｜ 扫描件 OCR 补全：2026-10-24
> 规模：**155** 份制度 · **106** 个附件原件（137 MB）· **1,337,704** 字正文 · 全部可入库

本目录是为**建向量数据库 + 原文溯源**准备的结构化语料：一份制度一个 Markdown 文件，按官网栏目层级分目录。
**155/155 条均有正文**——63 条取自网页文字、63 条从 PDF 文本层无损抽取、0 条来自 Word、剩余 29 条扫描件已逐页 OCR 转录（共 767 页）。

---

## 一、目录结构

```
学校规章制度/
├── README.md                    ← 本文件（字段字典 + 建库指引）
├── 00_抓取校验报告.md            ← 完整性对账、正文可得性、OCR 引擎与质量分级
├── index.csv                    ← 总索引（Excel 可直接打开，UTF-8 BOM）
├── index.jsonl                  ← 总索引（每行一条 JSON，程序直接读）
├── 栏目清单.csv                  ← 38 个栏目节点的声明数 vs 实抓数对账
└── 规章制度/
    ├── 01-综合法律法规/  16 条
    ├── 02-专项法律法规/  51 条
    │   ├── 学生安全/  1 条
    │   └── 实验室技术安全/  15 条
    │       …… 共 12 个有内容子栏目（另有 4 个空栏目）
    ├── 03-综合安全制度/  14 条
    ├── 04-专项安全制度/  69 条
    │   ├── 研究生安全/  3 条
    │   └── 本科生安全/  3 条
    │       …… 共 19 个有内容子栏目
    └── 05-应急预案/  5 条
```

文件名规则：`{制度标题}_{文章ID}.md`，标题已做 Windows 合法字符清洗，文章 ID 取自官网 URL 的 `c{栏目}a{文章ID}`，**与源站永久对应、可反查**。

---

## 二、字段字典（Markdown frontmatter）

```yaml
doc_id: "204336"                # 官网文章 ID，全局唯一
title / section / group         # 制度名 / 栏目路径 / 5 大板块之一
source_url: "http://..."        # ★ 溯源锚点：制度原文页
listing_url                     # 所属栏目列表页
publisher / published_at        # 页面「来源」「发布时间」
crawled_at                      # 抓取时刻
content_form: "file_only"       # 官网原始承载形态
text_source: "pdf_text_layer"   # ★ 正文从哪来，见下方取值表
char_count                      # 去空白正文字符数
ocr_needed: false               # true = 正文仍不完整（本轮已全部 false）
ocr_engine / ocr_pages / ocr_chars / ocr_at / ocr_note   # 仅 OCR 条目：引擎、页数、字数、时刻、质量提示
missing_text_attachments: []    # 无文本层且未转录的附件名（现已全部为空）
attachments:
  - path / url / bytes / sha256 # 原件相对路径、官网直链、大小、哈希前16位
    pages / text_chars / text_layer   # PDF 页数、抽到字符数、是否有文本层
    ocr: {engine, pages, done_at}     # 该原件的 OCR 转录记录
```

`text_source` 取值与置信度（**建库时按此分级加权或标注**）：

| 取值 | 条数 | 置信度 | 说明 |
|---|---|---|---|
| `web_page` | 63 | 最高 | 官网网页原文，未加工 |
| `pdf_text_layer` | 63 | 高 | PDF 原生文本层，等同复制粘贴，仅可能丢版式 |
| `docx_text` | 0 | 高 | Word 原件抽取 |
| `ocr_vision` | 13 | 中高 | **视觉模型**逐页转录，保留表格结构与条号 |
| `ocr_rapidocr_local` | 13 | 中 | **本地 RapidOCR** 转录，有字符级误差、表格被展平 |
| `ocr_vision+rapidocr` | 3 | 中高/中混合 | 同一条目的不同附件用了不同引擎 |

---

## 三、导入向量数据库

```python
import json
docs = [json.loads(l) for l in open("index.jsonl", encoding="utf-8")]
ready  = [d for d in docs if not d["ocr_needed"]]        # 全部 155 条均已具备正文
trusted = [d for d in docs if not d["text_source"].startswith("ocr_")]   # 126 条非 OCR，可直接采信
```

**切块建议**

- 法规类优先按条切：正则 `第[一二三四五六七八九十百零\d]+条`，命中标题即可直接引用条文；
- OCR 条目正文含 `【第 n 页】` 标记，**切块时把它转成 chunk 的 `page` 元数据**，回答即可给到「第几页」，溯源粒度最细；
- `web_page` / `pdf_text_layer` 自带段落空行，按 `\n\n` 聚合到 500–800 字/块、重叠 100 字；
- 建议给 chunk 加 `confidence` 字段：`web_page` > `pdf_text_layer` > `ocr_vision` > `ocr_rapidocr_local`，
  检索时低置信块可降权，或在回答里提示「此段来自扫描件识别，建议核对原件」。

**溯源引用范式**：制度名 + 条号/页号 + `source_url`；涉及关键数值时附 `attachments[].url` 让用户核对 PDF 原件。

---

## 四、必须知道的限制

1. **29 条正文来自 OCR，不是权威文本**。其中 13 条由本地 RapidOCR 识别，存在字符级混淆（`一`↔`-`、`；`↔`：`、`（一）`→`（-一）`）且**表格被展平成逐行文本**；13 条由视觉模型转录、3 条混用。引用数值、期限、罚额、条文编号时请以 `附件/` 原件为准（`sha256` 可验证原件未被改动）。
2. **PDF 文本层抽取与 OCR 都不是像素级还原**：分栏、印章、脚注会丢，表格类制度以 PDF 为准。
3. `特种设备安全` 下「电梯类 / 锅炉类 / 压力容器类」**标题是分类名不是制度名**（如电梯类正文实为《辽宁省电梯安全管理办法》），建库时建议按正文内标题二次识别。
4. `published_at` 是**页面发布时间，不是法规生效/施行日期**，也不能证明是现行有效版本。
5. **单次快照**（2026-10-24）。源站新增或修订不会同步，重跑抓取流程即可增量更新。

两点小提示：栏目名 `专项安全制度 > 医疗卫生安` 缺「全」字是**源站自身截断**（为溯源一致未改）；空栏目 4 个（源站本身无内容，非遗漏）：道路交通安全、核和辐射安全、水电暖设施安全、外委施工安全。

---

## 五、快速核对

```bash
find 规章制度 -name "*.md" ! -name "README.md" | wc -l     # 应为 155
find . -path "*附件*" -type f | wc -l                       # 应为 106
grep -l 'ocr_needed: true' -r 规章制度 | wc -l               # 应为 0
# 单条溯源：index.csv 取 source_url 打开官网页，应与同名 md 正文一致
```
