# Copilot / AI Agent Instructions for Enhanced AI Stock Analyzer (v3.1) ✅

## 一句话概览
- 这是一个 Python Web 应用（Flask）用于多市场股票分析（A股/港股/美股），核心是 `EnhancedWebStockAnalyzer`（数据采集 -> 分析 -> 可选 AI 深度分析），Web 层在 `enhanced_flask_server.py` 提供 SSE 流式输出。

## 关键文件与职责 🔑
- `enhanced_web_stock_analyzer.py` — 核心分析器：数据获取（akshare / pandas_datareader / yfinance）、技术/基本面/情绪分析、构建 AI Prompt、触发 AI 服务（OpenAI / Anthropic / Zhipu）并支持流式回调。
- `enhanced_flask_server.py` — Flask Web 服务：API 路由、SSE 管理 (`SSEManager`)、使用 `StreamingAnalyzer` 将 `stream_callback` 转为 SSE 事件发送到前端。
- `config.json` — 运行时配置（API 密钥、model_preference、缓存/流式/分析参数等），README 中有字段说明。
- `README.md` — 启动指引与配置说明（注意：README 提到 `requirements.txt`，但仓库中没有该文件）。

## 快速上手 / 运行（高优先级） ▶️
- 准备依赖（仓库缺少 `requirements.txt`，建议安装至少以下包）：
  ```bash
  pip install flask flask-cors pandas numpy akshare yfinance pandas-datareader requests openai anthropic zhipuai
  ```
- 配置 `config.json`：至少填写一个 `api_keys`（`openai` / `anthropic` / `zhipu`）。
- 启动服务：
  ```bash
  python enhanced_flask_server.py
  # 访问: http://localhost:5000
  ```
- 本地单文件测试（无需 Flask）：
  ```python
  from enhanced_web_stock_analyzer import EnhancedWebStockAnalyzer
  a = EnhancedWebStockAnalyzer()
  a.analyze_stock('600519', enable_streaming=True, stream_callback=print)
  ```

## AI 集成与重要契约 🤖
- `config.json` 中：
  - `ai.model_preference` 指定首选服务（`openai` / `anthropic` / `zhipu`）。如果首选不可用，代码会尝试其它已配置的服务作为备用。
  - `ai.models.*` 指定各提供商的模型名称；`ai.api_base_urls.openai` 支持自定义中转地址。
- 流式回调约定：任何 `stream_callback` 都会以多次 `stream_callback(content: str)` 的形式被调用；具体验证点：
  - `EnhancedWebStockAnalyzer.generate_ai_analysis(..., enable_streaming=True, stream_callback=cb)` 会把 `cb` 向下传到 `_call_openai_api/_call_claude_api/_call_zhipu_api`。
  - `enhanced_flask_server.py` 中 `StreamingAnalyzer.send_ai_stream` 将 `content` 包装为 SSE 事件 `ai_stream` 发送到前端。

## SSE / 前端交互要点 ⚡
- 事件名（常用）：`ai_stream` (部分AI文本块)，`final_result` (单个分析结果)，`batch_result` (批量分析)，`analysis_complete`，`analysis_error`。
- SSE 负载会先调用 `clean_data_for_json` 做序列化友好化（处理 NaN/日期/numpy 类型）。

## 数据源与变通策略 🛰️
- A股 / 港股 / 美股 数据主要使用 `akshare`；美股有备用 `pandas_datareader (stooq)` 流程。若某接口失败，代码会尝试备用接口并记录日志。
- 价格 / 基本面 / 新闻均为内存缓存（`price_cache`, `fundamental_cache`, `news_cache`），缓存持续时间由 `config.json` 控制（小时为单位）。注意：缓存是非持久化的。

## 代码风格与约定 🧭
- 股票代码处理：`normalize_stock_code()` 与 `detect_market()` 是全项目标准化入口（港股补零到 5 位，HK 前缀剥离，A 股用 6 位数字判断，美股为字母代码）。修改此处需同时更新前端输入验证逻辑。
- 列名标准化：`_standardize_price_data_columns()` 会尝试把外部数据映射到 `date, open, close, high, low, volume` 等列，所有数据处理应先走该函数来保证后续分析稳定。

## 小心事项 / 已知差异 ⚠️
- README 提到 `requirements.txt`，仓库中缺失；CI 或开发环境需显式安装依赖。
- `custom_prompts` 在 README/配置中有说明，但代码中未显式读取或使用 `custom_prompts.analysis_template`（这是可扩展点）。
- 没有包含单元测试或 CI 配置；新增逻辑建议添加针对 `normalize_stock_code`, `_standardize_price_data_columns` 以及 AI prompt 生成（deterministic 部分）的单元测试。

## Helpful code snippets (usage examples) ✂️
- 流式打印示例（CLI）：
  ```python
  def print_stream(content):
      print(content, end='', flush=True)
  analyzer.analyze_stock('TSLA', enable_streaming=True, stream_callback=print_stream)
  ```
- 在 Flask 中转换为 SSE（已存在实现）：参考 `StreamingAnalyzer` 中 `send_ai_stream`, `send_final_result`。

## Where to change things safely 🛠️
- 添加/切换 AI 提示模板：修改或替换 `_build_enhanced_ai_analysis_prompt()`；若要支持 `custom_prompts`，把 `config.json` 中的模板注入到该方法并保留回退逻辑。
- 添加强化缓存（持久化）或增加单元测试：`price_cache/fundamental_cache` 的使用点集中在 `get_stock_data` / `get_comprehensive_fundamental_data`，可以在这些位置序列化到磁盘或 Redis。

---
如果你希望，我可以：
1) 把 `custom_prompts` 支持补上并添加测试；
2) 生成 `requirements.txt`（基于当前 import）并在 README 中补充启动/部署要点；
3) 创建基础单元测试套件并设置 GitHub Actions CI。 

请告诉我你想优先做哪一项或对于说明文档还有哪些不清晰的地方，我会按需迭代。 ✨