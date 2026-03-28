LangChain v1 生产级代理构建指南（README）

APIYI_BASE_URL=https://api.deepseek.com
APIYI_API_KEY=sk-34e695e18f664dcc9676ff47ec4bc2f9
LLM_MODEL=deepseek-chat


状态：基于 LangChain v1 的“专注代理基础”整理而成，聚焦 create_agent、标准内容块（content_blocks）与简化命名空间。
适用读者：需要在生产环境构建可靠 LLM 代理、RAG、HITL、多代理系统的工程师与架构师。

目录

1. v1 有什么新变化

2. 安装与准备

3. 快速开始

4. 简化后的命名空间

5. 代理（Agent）核心

5.1 create_agent API 速览

5.2 模型选择：静态与动态

5.3 系统提示与调用方式

6. 消息与标准内容块 content_blocks

7. 工具（Tools）与 ToolRuntime

7.1 使用 @tool 定义工具

7.2 高级入参：Pydantic/JSON Schema

7.3 ToolRuntime：读取状态/上下文/存储与流写入

7.4 用 Command 更新代理状态

7.5 工具错误处理 @wrap_tool_call

8. 中间件（Middleware）

8.1 能力与场景

8.2 内置中间件一览 + 用法示例

8.3 执行顺序与“跳转”

8.4 自定义中间件（装饰器/类）

9. 结构化输出（Structured Output）

9.1 ProviderStrategy vs ToolStrategy

9.2 错误处理 handle_errors

10. 短期记忆（对话状态）

10.1 Checkpointer 与生产持久化

10.2 截断/删除/总结消息

10.3 在工具与中间件中访问状态

11. 流式处理（Streaming）

12. 代理中的上下文工程（Context Engineering）

13. 人工干预（HITL）

14. 多代理（Multi‑Agent）模式

15. 检索与 RAG

16. 模型上下文协议（MCP）

17. 长期记忆（Store）

18. 生产上线清单

19. 迁移指引

20. 常见问题 & 故障排查

1. v1 有什么新变化

create_agent：v1 中构建代理的标准方法，取代 langgraph.prebuilt.create_react_agent。接口更简洁，可通过中间件实现深度定制。

标准内容块 content_blocks：跨模型提供商统一访问推理、文本、工具调用等现代 LLM 能力。

简化命名空间：langchain 专注代理核心，旧能力迁移到 langchain-classic。

2. 安装与准备
pip install -U langchain  # v1 代理核心
# 提供商包（按需）
pip install -U langchain-openai langchain-anthropic
# LangGraph 持久化/检查点器（按需）
pip install -U langgraph langgraph-checkpoint-postgres
# MCP（可选）
pip install -U langchain-mcp-adapters mcp


提示：示例中的模型名仅示意。请根据你实际可用的提供商/模型替换，如 gpt-4o, gpt-4o-mini, claude-sonnet-4-5-20250929 等。

3. 快速开始
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

result = agent.invoke({
    "messages": [HumanMessage("What's the weather in SF?")]
})
print(result["messages"][-1].content)

4. 简化后的命名空间
模块	可用内容	备注
langchain.agents	create_agent, AgentState	代理创建/状态
langchain.messages	消息类型、content_blocks、trim_messages	从 langchain-core 复导
langchain.tools	@tool, BaseTool, 注入助手	从 langchain-core 复导
langchain.chat_models	init_chat_model, BaseChatModel	统一模型初始化
langchain.embeddings	Embeddings, init_embeddings	嵌入模型

常用导入：

# Agent building
from langchain.agents import create_agent, AgentState
# Messages and content
from langchain.messages import AIMessage, HumanMessage, ToolMessage
# Tools
from langchain.tools import tool
# Model initialization
from langchain.chat_models import init_chat_model
# Embeddings
from langchain.embeddings import init_embeddings

5. 代理（Agent）核心
5.1 create_agent API 速览

典型签名（概念化）：

create_agent(
    model,                       # str | BaseChatModel
    tools=None,                  # list[Tool] = []
    system_prompt=None,          # str | None
    middleware=None,             # list[AgentMiddleware] = []
    response_format=None,        # ProviderStrategy | ToolStrategy | schema type | None
    state_schema=None,           # type[AgentState] | TypedDict | None
    context_schema=None,         # dataclass | TypedDict | BaseModel | None
    checkpointer=None,           # LangGraph checkpointer | None
    store=None,                  # LangGraph store | None (long-term memory)
)


调用：

result = agent.invoke({"messages": [HumanMessage("Hello")]})

# 流式（进度/消息/自定义）
for chunk in agent.stream({"messages": "Search AI news"}, stream_mode="updates"):
    ...

5.2 模型选择：静态与动态

静态模型（字符串或模型实例）

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 直接传字符串
agent = create_agent("gpt-4o-mini", tools=[])

# 或显式实例化（可控更多参数）
model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=1000, timeout=30)
agent = create_agent(model, tools=[])


动态模型（中间件 @wrap_model_call）

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic = ChatOpenAI(model="gpt-4o-mini")
pro = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(req: ModelRequest, handler) -> ModelResponse:
    if len(req.state["messages"]) > 10:
        req.model = pro
    else:
        req.model = basic
    return handler(req)

agent = create_agent(model=basic, tools=[], middleware=[dynamic_model_selection])

5.3 系统提示与调用方式
agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="You are a helpful assistant. Be concise and accurate."
)

res = agent.invoke({"messages": [{"role": "user", "content": "Explain ML in 3 bullets."}]})

6. 消息与标准内容块 content_blocks

跨提供商统一访问推理、文本、工具调用等块：

from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
resp = model.invoke("What's the capital of France?")

for block in resp.content_blocks:
    if block["type"] == "reasoning":
        print("Reasoning:", block["reasoning"])
    elif block["type"] == "text":
        print("Text:", block["text"])
    elif block["type"] == "tool_call":
        print(f"Tool: {block['name']}({block['args']})")


好处：提供商无关、类型安全、向后兼容。

7. 工具（Tools）与 ToolRuntime
7.1 使用 @tool 定义工具
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Max results to return
    """
    return f"Found {limit} results for '{query}'"

# 自定义名称/描述
@tool("web_search", description="Search the web for information.")
def search(query: str) -> str:
    return f"Results for: {query}"

7.2 高级入参：Pydantic/JSON Schema
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool

class WeatherInput(BaseModel):
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = "celsius"
    include_forecast: bool = False

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    temp = 22 if units == "celsius" else 72
    out = f"Current weather in {location}: {temp}° {units}"
    if include_forecast:
        out += "\nNext 5 days: Sunny"
    return out

7.3 ToolRuntime：读取状态/上下文/存储与流写入
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

# 读取状态
@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    msgs = runtime.state["messages"]
    h = sum(m.__class__.__name__ == "HumanMessage" for m in msgs)
    a = sum(m.__class__.__name__ == "AIMessage" for m in msgs)
    t = sum(m.__class__.__name__ == "ToolMessage" for m in msgs)
    return f"Conversation: {h} human, {a} ai, {t} tool messages."

# 读取上下文（例如当前用户）
from dataclasses import dataclass
@dataclass
class UserContext:
    user_id: str

USER_DB = {"u1": {"name": "Alice", "tier": "Pro"}}

@tool
def current_user(runtime: ToolRuntime[UserContext]) -> str:
    uid = runtime.context.user_id
    return str(USER_DB.get(uid, {})) or "Unknown"

# 访问长期存储（Store）
store = InMemoryStore()
store.put(("users",), "u1", {"name": "Alice"})

@tool
def read_user(uid: str, runtime: ToolRuntime) -> str:
    v = runtime.store.get(("users",), uid)
    return str(v.value) if v else "Not found"

# 流写入（工具执行时推送进度）
@tool
def long_running(city: str, runtime: ToolRuntime) -> str:
    runtime.stream_writer(f"Fetching data for {city}...")
    runtime.stream_writer("Done.")
    return f"OK for {city}"

7.4 用 Command 更新代理状态
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool, ToolRuntime

@tool
def clear_conversation() -> Command:
    """Clear all messages in the state."""
    return Command(update={"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})

@tool
def update_user_name(new_name: str, runtime: ToolRuntime) -> Command:
    return Command(update={"user_name": new_name})

7.5 工具错误处理 @wrap_tool_call
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({e})",
            tool_call_id=request.tool_call["id"]
        )

8. 中间件（Middleware）
8.1 能力与场景

监控：日志/可观测性/分析

修改：动态提示、工具选择、输出格式

控制：重试、回退、终止、流控

合规：PII 检测、过滤、HITL 审批

添加到代理：

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[...],
)

8.2 内置中间件一览 + 用法示例

以下均可组合使用。

SummarizationMiddleware：上下文过长自动摘要

from langchain.agents.middleware import SummarizationMiddleware
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=4000,
            messages_to_keep=20,
            # summary_prompt="...",  # 可选自定义摘要提示
        )
    ],
)


HumanInTheLoopMiddleware（HITL）：敏感工具调用前人工审批

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email_tool": False,
            },
            description_prefix="Tool execution pending approval",
        )
    ],
)


AnthropicPromptCachingMiddleware：Anthropic 提示缓存

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain.messages import HumanMessage

LONG = "Please be a helpful assistant.\n<lots of context ...>"

agent = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
    system_prompt=LONG,
    middleware=[AnthropicPromptCachingMiddleware(ttl="5m")]
)
agent.invoke({"messages": [HumanMessage("Hi, my name is Bob")]})
agent.invoke({"messages": [HumanMessage("What's my name?")]})


ModelCallLimitMiddleware：限制模型调用次数

from langchain.agents.middleware import ModelCallLimitMiddleware
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end")]
)


ToolCallLimitMiddleware：限制工具调用次数（全局/按工具）

from langchain.agents.middleware import ToolCallLimitMiddleware

global_limit = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)
search_limit = ToolCallLimitMiddleware(tool_name="search", thread_limit=5, run_limit=3)

agent = create_agent(model="gpt-4o", tools=[...], middleware=[global_limit, search_limit])


ModelFallbackMiddleware：主模型失败自动回退

from langchain.agents.middleware import ModelFallbackMiddleware
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[ModelFallbackMiddleware("gpt-4o-mini", "claude-3-5-sonnet-20241022")]
)


PIIMiddleware：PII 检测/处理（遮蔽、脱敏、阻断）

from langchain.agents.middleware import PIIMiddleware
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
    ]
)


TodoListMiddleware：复杂任务 TODO/进度管理

from langchain.agents.middleware import TodoListMiddleware
from langchain.messages import HumanMessage

agent = create_agent(model="gpt-4o", tools=[...], middleware=[TodoListMiddleware()])
res = agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})
print(res["todos"])  # 带状态追踪的 todo 数组


LLMToolSelectorMiddleware：调用主模型前，先通过便宜模型选择相关工具

from langchain.agents.middleware import LLMToolSelectorMiddleware
agent = create_agent(
    model="gpt-4o",
    tools=[tool1, tool2, tool3, ...],
    middleware=[LLMToolSelectorMiddleware(model="gpt-4o-mini", max_tools=3, always_include=["search"])]
)


ToolRetryMiddleware：工具调用失败自动重试

from langchain.agents.middleware import ToolRetryMiddleware
agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0, max_delay=60.0, jitter=True)]
)


LLMToolEmulator：以 LLM 模拟工具执行（测试用）

from langchain.agents.middleware import LLMToolEmulator
agent = create_agent(
    model="gpt-4o",
    tools=[get_weather, search_database, send_email],
    middleware=[LLMToolEmulator()]  # 或 LLMToolEmulator(tools=["get_weather"], model="claude-...")
)


ContextEditingMiddleware / ClearToolUsesEdit：清理历史工具使用

from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[ContextEditingMiddleware(edits=[ClearToolUsesEdit(max_tokens=1000)])]
)

8.3 执行顺序与“跳转”

before_*：按顺序执行（从前到后）

after_*：逆序执行（从后到前）

wrap_*：嵌套包装（第一个最外层）

从中间件提前退出：在钩子中返回 {"jump_to": "end" | "tools" | "model"}。

8.4 自定义中间件（装饰器/类）

装饰器风格：

from typing import Any, Callable
from langchain.agents.middleware import (
    before_model, after_model, wrap_model_call, dynamic_prompt,
    AgentState, ModelRequest, ModelResponse
)
from langchain.messages import AIMessage

@before_model
def log_before(state: AgentState, runtime) -> dict[str, Any] | None:
    print(f"Calling model with {len(state['messages'])} messages")
    return None

@after_model(can_jump_to=["end"])
def validate(state: AgentState, runtime) -> dict[str, Any] | None:
    last = state["messages"][-1]
    if "BLOCKED" in last.content:
        return {"messages": [AIMessage("I cannot respond to that request.")], "jump_to": "end"}
    return None

@wrap_model_call
def retry(req: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    for i in range(3):
        try:
            return handler(req)
        except Exception:
            if i == 2:
                raise

@dynamic_prompt
def personalized(req: ModelRequest) -> str:
    uid = req.runtime.context.get("user_id", "guest")
    return f"You are a helpful assistant for user {uid}. Be concise."


类风格：

from langchain.agents.middleware import AgentMiddleware, AgentState

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime) -> dict | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None
    def after_model(self, state: AgentState, runtime) -> dict | None:
        print("Model returned:", state["messages"][-1].content)
        return None

9. 结构化输出（Structured Output）
9.1 ProviderStrategy vs ToolStrategy

两种策略：

ProviderStrategy：使用提供商原生结构化输出（如 OpenAI/Grok）。更可靠。

ToolStrategy：用“人工工具调用”实现结构化输出，适用于所有支持工具调用的模型。

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

class ContactInfo(BaseModel):
    name: str = Field(..., description="Name")
    email: str = Field(..., description="Email")
    phone: str = Field(..., description="Phone")

# 推荐：若模型支持原生结构化输出，可直接传 schema（自动选择 ProviderStrategy）
agent1 = create_agent(model="gpt-4o", tools=[], response_format=ContactInfo)

# 兼容方案：ToolStrategy（适用所有支持 tool-calls 的模型）
agent2 = create_agent(model="gpt-4o-mini", tools=[], response_format=ToolStrategy(ContactInfo))


更多例子（供应天气结构化）：

class Weather(BaseModel):
    temperature: float
    condition: str

def weather_tool(city: str) -> str:
    return f"it's sunny and 70 degrees in {city}"

agent = create_agent("gpt-4o-mini", tools=[weather_tool], response_format=ToolStrategy(Weather))

result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in SF?"}]})
print(result["structured_response"])  # Weather(temperature=70.0, condition='sunny')

9.2 错误处理 handle_errors
from typing import Union
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Event name")
    date: str = Field(description="Event date")

# 仅允许单一结构化输出；当模型错误地返回多个时自动提示重试
agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails])  # 默认 handle_errors=True
)


自定义消息：

ToolStrategy(schema=ContactInfo, handle_errors="Please provide valid contact info only.")


限制异常类型或自定义函数也支持：handle_errors=(ValueError, TypeError) 或 handle_errors=lambda e: "..."。

10. 短期记忆（对话状态）
10.1 Checkpointer 与生产持久化
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent("gpt-4o", tools=[], checkpointer=InMemorySaver())
agent.invoke({"messages": [{"role": "user", "content": "Hi, my name is Bob."}]},
             {"configurable": {"thread_id": "1"}})


生产建议：使用数据库支持的检查点器（如 Postgres）。

# pip install langgraph-checkpoint-postgres
from langgraph.checkpoint.postgres import PostgresSaver

DB = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB) as cp:
    cp.setup()
    agent = create_agent("gpt-4o", tools=[], checkpointer=cp)


自定义 State Schema：

from typing import TypedDict
from langchain.agents import AgentState

class CustomState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent("gpt-4o", tools=[], state_schema=CustomState, checkpointer=InMemorySaver())

10.2 截断/删除/总结消息

截断（before_model）：

from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@before_model
def trim_messages(state: AgentState, runtime) -> dict | None:
    msgs = state["messages"]
    if len(msgs) <= 3:
        return None
    first = msgs[0]
    recent = msgs[-3:] if len(msgs) % 2 == 0 else msgs[-4:]
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), first, *recent]}


删除（after_model）：

from langchain.agents.middleware import after_model

@after_model
def delete_old(state: AgentState, runtime) -> dict | None:
    msgs = state["messages"]
    if len(msgs) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in msgs[:2]]}
    return None


总结（SummarizationMiddleware）：见 8.2
。

10.3 在工具与中间件中访问状态

工具读写：见 7.3
、7.4

动态提示/模型/工具选择：见 12

11. 流式处理（Streaming）

三种模式：

"updates"：按节点级别（模型/工具）推送进展

"messages"：LLM token 粒度流式

"custom"：工具自定义流（通过 stream_writer）

from langchain.agents import create_agent

def get_weather(city: str) -> str:
    return f"It's always sunny in {city}!"

agent = create_agent(model="gpt-4o-mini", tools=[get_weather])

# 进度
for chunk in agent.stream({"messages": "Weather in SF?"}, stream_mode="updates"):
    print(chunk)

# Token 级
for token, meta in agent.stream({"messages": "Weather in SF?"}, stream_mode="messages"):
    print(meta["langgraph_node"], token.content_blocks)

# 自定义（来自工具）
for data in agent.stream({"messages": "Weather in SF?"}, stream_mode="custom"):
    print(data)


多模式同时：

for mode, chunk in agent.stream({"messages": "Weather in SF?"}, stream_mode=["updates", "custom"]):
    print("mode:", mode, "content:", chunk)

12. 代理中的上下文工程（Context Engineering）

动态系统提示：

from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def state_aware_prompt(req: ModelRequest) -> str:
    base = "You are a helpful assistant."
    if len(req.messages) > 10:
        base += "\nThis is a long conversation - be extra concise."
    return base


注入文件上下文：

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def inject_file_context(req: ModelRequest, handler) -> ModelResponse:
    files = req.state.get("uploaded_files", [])
    if files:
        desc = "\n".join([f"- {f['name']} ({f['type']}): {f['summary']}" for f in files])
        req = req.override(messages=[*req.messages, {"role": "user", "content": f"Files:\n{desc}"}])
    return handler(req)


根据状态选择工具/模型/输出格式：

@wrap_model_call
def state_based_tools(req, handler):
    is_auth = req.state.get("authenticated", False)
    if not is_auth:
        req = req.override(tools=[t for t in req.tools if t.name.startswith("public_")])
    return handler(req)

@wrap_model_call
def state_based_model(req, handler):
    req = req.override(model=init_chat_model("gpt-4o" if len(req.messages) > 10 else "gpt-4o-mini"))
    return handler(req)

from pydantic import BaseModel, Field
class SimpleResp(BaseModel): answer: str = Field(description="Brief")
class DetailedResp(BaseModel): answer: str; reasoning: str; confidence: float

@wrap_model_call
def state_based_output(req, handler):
    req = req.override(response_format=DetailedResp if len(req.messages) >= 3 else SimpleResp)
    return handler(req)

13. 人工干预（HITL）

配置中断：见 8.2 / HumanInTheLoopMiddleware
。

执行与恢复：

from langgraph.types import Command
config = {"configurable": {"thread_id": "my-thread"}}

result = agent.invoke({"messages": [{"role": "user", "content": "Delete old records"}]}, config=config)

if "__interrupt__" in result:
    # 展示待审操作（action_requests）并收集决策
    decisions = [{"type": "approve"}]  # 或 "edit", "reject"
    agent.invoke(Command(resume={"decisions": decisions}), config=config)

14. 多代理（Multi‑Agent）模式

两类模式：

工具调用：主管 Agent 将子 Agent 作为工具调用（集中式编排）

交接：当前 Agent 将控制权转移给另一 Agent（去中心化对话）

子代理作为工具：

from langchain.tools import tool

subagent = create_agent(model="gpt-4o-mini", tools=[...])

@tool("subagent_exec", description="Delegate to sub-agent")
def call_subagent(query: str):
    r = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return r["messages"][-1].content

controller = create_agent(model="gpt-4o", tools=[call_subagent])


携带 ToolCallId 与扩展状态回传：

from typing import Annotated
from langchain.tools import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage

@tool("call_subagent_full")
def call_subagent_full(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    r = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return Command(update={
        "example_state_key": r.get("example_state_key"),
        "messages": [ToolMessage(content=r["messages"][-1].content, tool_call_id=tool_call_id)]
    })

15. 检索与 RAG

两步 RAG：固定“检索 → 生成”，简单高效，适合 FAQ/文档机器人。
代理式 RAG：代理在推理中动态决定何时检索，灵活但延迟可变。
混合 RAG：加入查询增强/检索验证/答案验证等步骤。

最小代理式 RAG 示例：

import requests
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def fetch_url(url: str) -> str:
    r = requests.get(url, timeout=10.0)
    r.raise_for_status()
    return r.text

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[fetch_url],
    system_prompt="Use fetch_url to retrieve web content; quote snippets."
)

16. 模型上下文协议（MCP）

多服务器客户端：

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({
    "math": {"transport": "stdio", "command": "python", "args": ["/abs/path/math_server.py"]},
    "weather": {"transport": "streamable_http", "url": "https://:8000/mcp"}
})

tools = await client.get_tools()
agent = create_agent("claude-sonnet-4-5-20250929", tools)


自定义服务器（示例）：

# math_server.py
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Math")

@mcp.tool() def add(a: int, b: int) -> int: return a + b
@mcp.tool() def multiply(a: int, b: int) -> int: return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")


有状态会话：

from langchain_mcp_adapters.tools import load_mcp_tools

async with client.session("math") as session:
    tools = await load_mcp_tools(session)

17. 长期记忆（Store）

存储/搜索：

from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 2.0] * len(texts)]  # 示例：替换为真实嵌入

store = InMemoryStore(index={"embed": embed, "dims": 2})

ns = ("user-1", "app")
store.put(ns, "mem-1", {"rules": ["short answers", "English only"], "tag": "pref"})
item = store.get(ns, "mem-1")
items = store.search(ns, filter={"tag": "pref"}, query="language preferences")


在工具中读写 Store：见 7.3
。

18. 生产上线清单

持久化：Checkpointer（如 Postgres）与长记忆 Store（数据库/向量库）

观测性：记录模型/工具调用、延迟、token 消耗、错误率

上下文控制：Summarization、消息截断、ContextEditing

安全合规：PIIMiddleware、HITL、权限控制、审计日志

可靠性：ModelFallback、ToolRetry、调用上限（Model/Tool Call Limit）

成本优化：LLMToolSelector（先选工具）、动态模型选择（小→大）、提示缓存

测试与沙箱：LLMToolEmulator、单元测试中间件、灰度发布

速率限制：外部 API 调用限速、指数退避

Secrets 管理：凭据/密钥不要写入状态/日志

19. 迁移指引

从 langgraph.prebuilt.create_react_agent → create_agent：接口更简洁，配合中间件定制所有环节。

从旧版 langchain → langchain-classic：非代理核心能力迁移至 langchain-classic，v1 的 langchain 专注代理/工具/消息/模型初始化/嵌入。

20. 常见问题 & 故障排查

工具入参不生效：确保使用了类型注解或 args_schema（Pydantic/JSON Schema）。

RemoveMessage 无效：确认 messages 字段使用 add_messages reducer（AgentState 默认具备）。

结构化输出不匹配：启用 ToolStrategy(..., handle_errors=True)；检查字段/范围（如评分 1–5）。

循环过长/成本高：启用 ModelCallLimitMiddleware、ToolCallLimitMiddleware、SummarizationMiddleware，并考虑动态模型选择。

隐私与合规：使用 PIIMiddleware、HITL；敏感工具调用前人工审批。

测试困难：用 LLMToolEmulator 在 CI 中模拟工具。

附：完整最小示例
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import (
    dynamic_prompt, wrap_model_call, SummarizationMiddleware, ToolRetryMiddleware
)

@dataclass
class Ctx: user_id: str

@tool
def hello(name: str, runtime: ToolRuntime[Ctx]) -> str:
    return f"Hello {name}! (from {runtime.context.user_id})"

@dynamic_prompt
def sys_prompt(req) -> str:
    return f"You are a helpful assistant for user {req.runtime.context.user_id}."

@wrap_model_call
def select_model(req, handler):
    req = req.override(model=("gpt-4o" if len(req.messages) > 10 else "gpt-4o-mini"))
    return handler(req)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[hello],
    middleware=[sys_prompt, select_model, SummarizationMiddleware("gpt-4o-mini", 4000, 20),
                ToolRetryMiddleware(max_retries=2)]
)

result = agent.invoke(
    {"messages": [HumanMessage("Say hello to Alice")]},
    context=Ctx(user_id="u1")
)
print(result["messages"][-1].content)


到此，你已具备在生产环境中搭建可靠 LangChain v1 代理的一整套“砖与瓦”。