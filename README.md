# ✈️ SmartTrip — AI Travel Planning Agent

> 输入目的地 + 日期 + 预算，AI 自动搜索机票、酒店、景点、天气，生成完整旅行攻略。

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-orange)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.42%2B-FB7185)](https://gradio.app)

---

## 📖 项目简介

SmartTrip 是一个基于 **LangGraph ReAct Agent** 的智能旅行规划助手。它通过 LLM 驱动的自主决策，串联多个搜索工具，一站式完成从需求输入到完整攻略输出的全过程。

**用户只需说**：「去成都玩 5 天，预算 5000，从北京出发」
**Agent 自动完成**：搜机票 → 搜酒店 → 搜景点美食 → 查天气 → 算预算 → 输出结构化行程

---

## 🏗 系统架构

```
用户输入 ──► Gradio UI (demo.py)
                │
                ▼
         FastAPI /plan 接口 (api/app.py)
                │
                ▼
         LangGraph ReAct Agent (agent/graph.py)
                │
                ├── flight_search     ──► SerpAPI              (搜机票)
                ├── hotel_search      ──► SerpAPI              (搜酒店)
                ├── attraction_search ──► SerpAPI              (搜景点/美食)
                ├── weather_check     ──► SerpAPI              (查天气)
                └── budget_calculator ──► Python 沙箱执行       (算预算)
                │
                ▼
         结构化 Markdown 攻略
```

### 核心技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| Agent 框架 | **LangGraph** | 状态图驱动 ReAct 循环（Plan → Tools → Reflect） |
| 大模型 | **Qwen3.5** (NSCC MaaS) | LLM 推理与工具调用 |
| 后端 | **FastAPI** | RESTful API，Pydantic 校验 |
| 前端 | **Gradio** | 交互式 UI 演示 |
| 搜索 | **SerpAPI** | 航班、酒店、景点、天气实时数据 |
| 记忆 | 双层记忆系统 | 短期窗口 + 长期偏好压缩 |
| 部署 | **Docker Compose** | 一键启动 |

---

## 🚀 快速开始

### 前置条件

- Python 3.12+
- Conda（推荐）或 pip
- 有效的 LLM API Key（本工程使用 NSCC MaaS 平台）
- 有效的 SerpAPI Key（可选，用于实时搜索）

### 1. 克隆与安装

```bash
git clone <your-repo-url>
cd smarttrip-travel-agent

# 推荐使用 Conda 环境
conda create -n travel_agent python=3.12
conda activate travel_agent

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

| 变量 | 说明 | 获取方式 |
|------|------|---------|
| `LLM_API_KEY` | LLM 模型 API Key | NSCC MaaS / 其他 OpenAI 兼容平台 |
| `LLM_BASE_URL` | API 端点地址 | 默认为 NSCC MaaS 地址 |
| `LLM_MODEL` | 模型名称 | 如 `Qwen3.5`, `qwen-turbo` 等 |
| `SERPAPI_API_KEY` | 搜索 API Key（可选） | [serpapi.com](https://serpapi.com) |

### 3. 启动服务

**终端 1 — 启动后端：**

```bash
conda activate travel_agent
python main.py
```

访问 `http://localhost:8000/docs` 查看 API 文档。

**终端 2 — 启动前端：**

```bash
conda activate travel_agent
python demo.py
```

访问 `http://localhost:7860` 体验交互界面。

### 4. 使用示例

```bash
# 或直接通过 API 调用
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Chengdu",
    "origin": "Beijing",
    "start_date": "2026-07-01",
    "end_date": "2026-07-05",
    "budget": 5000,
    "preferences": "food, nature",
    "travelers": 2
  }'
```

### Docker 部署

```bash
docker-compose up -d
```

---

## 📁 项目结构

```
smarttrip-travel-agent/
├── agent/
│   ├── graph.py           # LangGraph ReAct 状态图（核心）
│   └── prompts.py         # 旅行规划 System Prompt
├── tools/
│   └── registry.py        # 5 个旅行工具 + Pydantic Schema
├── memory/
│   └── manager.py         # 双层记忆系统
├── api/
│   └── app.py             # FastAPI 接口（/plan /history /tools）
├── demo.py                # Gradio 交互界面
├── main.py                # 后端启动入口
├── tests/
│   └── test_agent.py      # Pytest 测试
├── docker-compose.yml     # Docker 部署
├── Dockerfile
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量模板
└── .gitignore
```

---

## 🧪 测试

```bash
pytest tests/ -v
```

---

## 📮 API 文档

启动后端后访问 `http://localhost:8000/docs` 查看 Swagger 交互式文档。

### 主要端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/plan` | 提交旅行规划请求 |
| GET | `/plan/{session_id}` | 获取指定会话的行程 |
| GET | `/history/{session_id}` | 获取对话历史 |
| DELETE | `/session/{session_id}` | 清除会话 |
| GET | `/tools` | 列出可用工具 |
| GET | `/health` | 健康检查 |

---

## 🧠 关键技术决策

| 问题 | 方案 | 效果 |
|------|------|------|
| 工具调用参数幻觉 | Pydantic Schema 强制校验输入 | 错误率从 ~20% → 接近 0 |
| 多轮对话上下文丢失 | 短期窗口 + 长期压缩记忆 | 跨轮记住用户偏好（预算倾向、目的地偏好等）|
| Agent 无限循环 | MAX_STEPS 硬限制 | 所有会话在有限步数内终止 |
| 搜索结果碎片化 | 明确提示词要求先搜后综合 | 输出从零散信息变为结构化 Markdown 表格 |

---

## 📄 License

MIT
