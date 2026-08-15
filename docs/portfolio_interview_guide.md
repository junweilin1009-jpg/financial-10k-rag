# Portfolio and Interview Guide（中文）

这份指南基于仓库中可以验证的代码、测试和实验记录。建议先理解逻辑，再用自己的语言表达，
不要逐字背诵。

## 一句话定位

这是一个面向 Alphabet、Amazon 和 Microsoft 2025 Form 10-K 的证据约束型 RAG 系统：
它通过财务语义检索、公司级路由、表格/整页补充和严格的来源边界，回答需要事实提取、计算、
跨公司比较和风险分析的问题，并返回实际进入模型上下文的 PDF 页级来源。

## 可验证的项目规模

- 3 份 2025 Form 10-K，共 397 页；
- 117 道公开问题：62 道开发题、30 道隐藏泛化题、15 道受保护 holdout、10 道跨组题；
- 42 项离线测试，覆盖单元、失败路径和核心问答集成流程；
- CI 配置覆盖 Python 3.11、3.12、3.13；
- 记录过 5 个模型在同一最终检索架构上的 10 题人工评审实验。

这些是仓库当前可以核对的规模，不代表生产用户数、商业收益或线上 SLA。

## 我的实际职责（Junwei Lin）

- **数据准备：** 预处理并整理三家公司的 397 页 10-K 文件，为后续解析、公司识别和 PDF
  页级溯源提供一致的输入；
- **评估设计：** 参与构建 117 题分阶段问题库，覆盖事实检索、计算、跨公司比较、多语言、
  定性分析和对抗性问题；
- **质量验证：** 核对参考答案、filing/页码依据和重复出现的失败类型，使改进依据证据质量，
  而不只是答案是否流畅；
- **作品集发布：** 为可复现安装、测试、文档、CI 和招聘者可读性定义验收标准，完成最终检查
  并协调 GitHub 公开发布。

这四项职责可以概括为：**把原始年报变成可用数据，把问题变成可审计评估，再把课程成果变成
可公开展示的工程项目。**

## 30 秒版本：HR

我参与构建了一个金融 10-K RAG 项目，主要负责三份年报的数据预处理、117 题评估库建设、
来源与答案验证，以及作品集发布验收。系统读取 Alphabet、Amazon 和 Microsoft 共 397 页的
年度报告，使用 OpenAI embeddings、FAISS 和 GPT 模型回答财务事实、计算及跨公司比较问题。
我重点保证输入文件、参考答案和引用页面能够被追踪，并推动最终版本达到可安装、可测试、有 CI、
有完整文档的 GitHub 项目标准。

## 1 分钟版本：Hiring Manager

这个项目解决的是金融报告问答中的“答案看起来合理，但证据可能错位”问题。输入是三家公司的
2025 Form 10-K，输出是带 PDF 文件和页码来源的答案，以及检索策略、延迟和 token 元数据。

初始方案主要依赖通用向量相似度，但跨公司问题容易只检索到一家公司的证据，复杂表格也容易
丢失行列语义。我们因此加入公司识别与过滤、财务问题分类、精确指标扩展、表格页与整页补充、
去重和上下文限制。工程化阶段又补上内容级 filing 校验、安全 FAISS/JSON 缓存、显式凭证传递、
holdout 防泄漏和 CI。最终价值是把一次性的课程演示变成一个可复现、可审计、可以在技术面试中
解释设计取舍的完整项目。

## 3 分钟版本：Technical Interview

### Problem

通用 RAG 在财务报告上有三个主要风险：检索到错误公司或期间；从复杂表格拿到相邻但错误的
字段；跨公司比较时证据覆盖不完整。最终数字正确也不一定说明证据链正确。

### Data

语料是三份 2025 Form 10-K，共 397 页。入口会从 PDF 内容验证 issuer、Form 10-K 类型和
报告期，不能只靠文件名。`data/manifest.csv` 固定随仓库提供文件的字节数和 SHA-256，防止
语料被静默替换。

### Approach

`pdfplumber` 保留页面布局并提取表格密集页；文本按 1,000 字符、150 字符 overlap 切分，
同时保留公司、文件、PDF 页码、财年和文档类型元数据。OpenAI embedding 写入本地 FAISS。
查询阶段先识别公司和问题类型，再执行公司过滤的 similarity/MMR 检索，并针对税率、现金税、
分部收入、capex 等任务补充完整证据页。去重后，只有真正进入上下文的文档才能出现在最终
`sources` 中。

### Engineering

所有界面共享 `src/financial_rag` 包。配置用经过校验的 frozen dataclass；API key 显式传给
客户端；缓存使用原生 FAISS 文件和 JSON mapping，避免反序列化 pickle。批量评估默认排除
holdout，运行 holdout 必须显式确认且 Git 工作树干净，并在结果中记录 commit SHA 和 UTC
时间。42 项离线测试覆盖配置、PDF、检索、缓存、错误降级和完整 answer contract。

### Result and limitation

记录的 10 题人工评审实验中，最终 Sol 配置得到 20/20，其他四个模型为 19.5/20；由于这
10 题参与过后续检索改进，它们是回归证据，不是完全未见泛化结果。更可信的下一步是在代码
冻结后只运行一次 15 题 holdout。系统也不是实时行情或投资建议工具，页码引用可追踪输入证据，
但不等于形式化的 citation entailment 验证。

## 5 分钟完整 Walkthrough

1. **背景：** 财务问答的难点不是让模型说得流畅，而是确保 issuer、period、metric、unit、
   numerator/denominator 和 comparability 全部对齐。
2. **输入：** 三份内容验证过的 2025 10-K；manifest 固定当前课程语料，应用仍允许上传同一
   组公司和报告期的替代副本。
3. **索引：** 页面文本和表格补充保留元数据，切分后通过 OpenAI embedding 写入 FAISS；
   source bytes 与 embedding/chunk 配置共同形成 cache fingerprint。
4. **检索：** 识别目标公司和问题类型；跨公司问题保证每家公司都有候选；精确财务问题增加
   focused evidence、table supplement 和 task-complete evidence pages。
5. **生成：** 上下文有严格字符上限；若早期证据放不下，不会跳过它去引用后面的文档；系统
   prompt 要求展示公式、修正错误前提并拒绝把缺失上下文写成 filing-wide absence。
6. **输出：** typed `AnswerResult` 包含答案、真实上下文来源、策略、公司、token、延迟和停止
   原因；Streamlit、CLI 和 evaluator 使用同一契约。
7. **质量：** 42 项离线测试、Ruff、依赖检查和 Python 3.11-3.13 CI；付费 API 测试不放进
   每次 push 的 CI。
8. **评估：** 117 题按开发、隐藏泛化、holdout 和跨组阶段管理；holdout 有显式执行保护和
   Git provenance。
9. **结果：** 报告真实记录的模型质量、延迟和成本快照，但不承诺未来价格或确定性输出。
10. **边界：** 非实时数据、非投资建议；复杂表格仍可能解析失败；`langchain-community`
    FAISS 集成已提示未来需要迁移；公开发布前还需确认软件许可证和课程 PDF 再分发权。

## 主要重构：Before / After / Why / Trade-off

| Before | After | Why | Trade-off |
|---|---|---|---|
| 依赖文件重复维护 | `pyproject.toml` 为唯一依赖来源 | 防止版本漂移，支持标准安装 | 使用者需熟悉 extras，如 `.[dev]` |
| API key 可能进入全局环境状态 | key 显式传给模型与 embedding client | 减少跨 session 泄漏和隐藏副作用 | 函数签名多一个依赖 |
| 根据文件名推断 filing | 从 PDF 内容校验 issuer、form、period | 防止错误期间或公司进入索引 | 启动时需读取 PDF 开头页面 |
| FAISS pickle 恢复 | 原生 FAISS index + JSON document map | 避免加载可执行 pickle | 自己维护格式校验和版本升级 |
| 所有检索候选都可能显示为来源 | 只返回真正进入模型上下文的文档 | 引用与模型输入一致 | 被 context cap 截断的候选不展示 |
| 单个大类承担多数职责 | cache、generation、schemas、filings 分离 | 降低耦合并支持独立测试 | `engine.py` 仍保留领域编排复杂度 |
| 表格/整页解析失败时静默继续 | 记录 warning 并保留文本或原 chunk | 失败可诊断且功能可降级 | 日志中会出现可接受的非致命告警 |
| `--all` 可能包含 holdout | 默认排除，显式确认并要求 clean commit | 减少评估泄漏 | holdout 流程更严格、操作更多 |

## 高频技术追问与回答框架

### 为什么用 RAG，而不是把整份 PDF 直接交给模型？

三份文件共 397 页，整份输入成本高，也难保证跨公司证据均衡。RAG 先缩小到任务相关页面，
保留可审计来源，并允许对财务问题使用有针对性的完整页面补充。代价是检索召回成为新的失败点。

### 为什么选 FAISS？

语料固定、规模小、单机运行，不需要远程数据库的并发和运维能力。FAISS 足够快且易于本地复现。
如果未来扩展到大量公司、多人并发和增量更新，再评估带 metadata filter 的托管向量数据库。

### 为什么不使用 YAML 或 Pydantic Settings？

项目只有一个经过实验冻结的配置族，frozen dataclass 已能提供类型、默认值和 fail-fast 校验。
引入额外配置框架会增加依赖和抽象；多环境部署出现后再升级更合理。

### 为什么 `engine.py` 仍然较大？

它保留的是共享 `vector_store`、table pages、evidence pages 和 build stats 的有状态领域编排。
无状态的 cache、generation、schema 和 filing validation 已拆出。当前没有第二种后端或第二套
金融规则实现，继续拆成许多类会增加跳转成本；未来加入多 backend 时再抽象 retriever interface。

### 如何避免错误引用？

来源不是从全部 retrieval candidates 生成，而是由 context builder 返回“实际包含的 documents”，
`AnswerResult.sources` 只根据这一集合构造。测试覆盖 context cap 截断和 source count。

### 表格解析失败怎么办？

表格是 enrichment，不是唯一证据路径。失败时记录文件和页码 warning，保留布局文本；整页扩展
失败则保留原 similarity chunk。这样不会静默，也不会因为可选增强失败而丢掉全部检索能力。

### 如何防止 data leakage？

开发、隐藏泛化、holdout 和跨组题分 stage。`--all` 不运行 holdout；holdout 要显式 acknowledgement，
且拒绝 dirty worktree。输出记录 commit SHA、UTC 时间和 dirty 状态。运行后不应继续针对 holdout
调参，否则它就不再是 holdout。

### 20/20 是否代表系统达到 100% accuracy？

不是。它只是 10 道跨组问题、每题 2 分的人工评审结果，而且这些问题后来影响了检索改进，
所以最终分数属于回归验证。不能外推为总体准确率。15 道未运行 holdout 才是下一次更干净的检查。

### 如果生产化，先做什么？

先确认数据许可和产品边界，再增加认证、请求限流、持久化观测、API 错误分类和成本预算；然后按
实际规模决定远程对象存储/向量库、异步索引和增量更新。不会一开始就加入 Kubernetes、Kafka
或微服务，因为当前没有对应负载证据。

## Behavioral Interview：STAR 故事

### 情境（Situation）

团队最初认为模型和 embedding 组合是质量差异的主要来源，但不同成员的实现混入了不同的 PDF
处理和检索逻辑，实验无法公平比较。

### 任务（Task）

需要找出真正的质量驱动因素，统一代码基础，同时保留可复现的实验和团队协作记录。

### 行动（Action）

我们对强弱实现进行了代码级比较，发现更强版本保留页面布局、表格页、issuer/page metadata，
并使用 query-aware retrieval。团队随后统一架构，再在同一 retrieval pipeline 上比较模型。
后续每次失败先分类为 retrieval、prompt、PDF parsing 或 missing evidence，再把它转化成通用规则和
回归问题，而不是把参考答案写进代码。

### 结果（Result）

项目形成统一、可解释的金融证据管线；最终 5 模型实验可以在相同检索基础上比较。工程化后又
加入 42 项离线测试、安全缓存、holdout 保护和 CI。最重要的学习是：金融 RAG 的关键往往是
证据工程与评估设计，而不是单纯更换更大的模型。

## Resume Bullets（可按岗位选择 3–4 条）

下面前三条最贴近 Junwei Lin 在团队中的职责；后面的条目适合在能够清楚解释团队协作边界时使用。

- Prepared and validated a 397-page corpus of Alphabet, Amazon, and Microsoft Form 10-K filings,
  supporting consistent PDF processing and page-level evidence traceability.
- Co-developed a staged 117-question evaluation bank spanning factual retrieval, calculations,
  cross-company comparisons, multilingual prompts, qualitative analysis, and adversarial cases.
- Reviewed reference answers, filing/page evidence, and recurring failure modes; defined release
  acceptance criteria for reproducibility, testing, CI, documentation, and GitHub presentation.

- Engineered an evidence-grounded RAG pipeline over 397 pages of Alphabet, Amazon, and Microsoft
  Form 10-K filings using OpenAI embeddings, FAISS, issuer-aware retrieval, and page-level source
  metadata for auditable financial Q&A.
- Built content-based filing validation and a SHA-256 corpus manifest, preventing wrong-issuer,
  wrong-period, or silently changed documents from entering the reproducible indexing workflow.
- Refactored a course prototype into an installable Python package with safe FAISS/JSON caching,
  typed result contracts, explicit credential handling, structured failure logging, and 42 offline
  unit, failure-path, and integration tests.
- Designed a 117-question staged evaluation workflow with protected holdout execution, Git commit
  provenance, and CSV/Markdown exports; added Python 3.11-3.13 CI for dependency, lint, formatting,
  test, and CLI packaging checks.

如果简历空间有限，优先选择与目标岗位最相关的三条。不要写“production deployment”、真实用户数、
收入提升或总体 accuracy；仓库中没有这些证据。

## 发布和面试前最后确认

- 与团队确认软件许可证以及三份课程 PDF 是否允许随公开仓库再分发；
- 发布 GitHub remote 后确认 Actions 三个 Python 版本全部通过；
- 冻结并提交代码，再决定是否使用真实 API 运行一次 15 题 holdout；
- holdout 结果无论好坏都如实报告，不再基于它调参；
- 面试前实际运行一次安装、CLI `--help`、测试和 Streamlit，确保演示环境可用；
- 明确说明模型价格、可用性、延迟和输出会随时间变化。
