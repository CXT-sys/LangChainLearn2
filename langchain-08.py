"""
Agent 架构与创建
========
第五课：LangChain 核心应用 - Agent 基础
模块：2.1
目标：掌握 Agent 架构和 create_agent API
知识点：
-- Agent 概念与架构
-- create_agent 详细用法
-- ReAct Agent 模式
-- Agent 状态管理
-- 工具调用（Tool Calls）机制详解
"""

from typing import Any
from typing_extensions import NotRequired
from datetime import datetime

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import AgentState, Runtime, before_model, after_model
from dotenv import load_dotenv

# 加载环境变量与 LangFuse 追踪器
load_dotenv()
langfuse_handler = CallbackHandler()


# ===================== 示例1：Agent架构深度解析 =====================
def example_1():
    """
    示例 1：Agent 架构深度解析
    
    目标：理解 Agent 的内部工作原理
    知识点：
    -- Agent 的执行流程
    -- ReAct 模式详解
    -- 消息流转过程
    """
    print("\n===== 示例 1: Agent 架构深度解析 =====")

    # 定义工具
    @tool
    def search(query: str) -> str:
        """搜索信息。"""
        return f"搜索结果：关于 '{query}' 的相关信息..."

    @tool
    def calculator(expression: str) -> str:
        """计算表达式。"""
        try:
            return str(eval(expression))
        except Exception:
            return "计算错误"

    # 初始化模型
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    # 创建 Agent
    agent = create_agent(
        model=model, tools=[search, calculator], system_prompt="你是一个助手，可以搜索信息和进行计算。"
    )

    # 使用 stream 查看每一步的执行
    print("\n--- Agent 执行流程（逐步显示） ---")
    for step in agent.stream(
        {"messages": [("user", "搜索 Python 的最新版本，然后计算 2+3")]},
        stream_mode="updates",
        version="v2",
        config={"callbacks": [langfuse_handler]},
    ):
        #
        # 代码解释：agent.stream() 与 stream_mode="updates"
        # =============================================
        # agent.stream() 以流式方式逐步返回 Agent 的执行过程。
        #
        # stream_mode="updates" 表示每次只返回最新变化的数据，
        # 而不是返回完整状态。这样可以看到 Agent 每一步做了什么。
        #
        # step 的结构：
        # step = {
        #     "data": {                  # 各节点的数据
        #         "model": {             # 节点名 (model, tools, agent 等)
        #             "messages": [AIMessage(...)]  # 该节点输出的消息
        #         },
        #         "tools": {
        #             "messages": [ToolMessage(...)]
        #         }
        #     }
        # }
        #
        # 遍历 step["data"].items() 可以获取每个节点的输出，
        # data["messages"][-1] 取最新消息（最近一条）。
        #
        for node, data in step["data"].items():
            if data.get("messages"):
                msg = data["messages"][-1]
                print(f"\n节点: {node}")
                print(f"类型: {msg.type}")
                print(f"内容: {msg.content[:150]}...")

    print("\n--- 最终结果 ---")
    result = agent.invoke({"messages": [("user", "计算 10 乘以 5")]}, config={"callbacks": [langfuse_handler]})
    print(f"AI: {result['messages'][-1].content}")


# ===================== 示例2：create_agent参数详解 =====================
def example_2():
    """
    示例 2: create_agent 参数详解

    目标：掌握 create_agent 的所有关键参数
    知识点：
    - model: 模型配置
    - tools: 工具列表
    - system_prompt: 系统提示
    - middleware: 中间件
    - name: Agent 名称
    """
    print("\n===== 示例 2: create_agent 参数详解 =====")

    # 工具
    @tool
    def get_time() -> str:
        """获取当前时间。"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 中间件
    @before_model
    def add_timestamp(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """在每条消息前添加时间戳。"""
        print(f"[中间件] 处理消息，当前状态消息数：{len(state['messages'])}")
        return None

    # 创建带所有参数的 Agent
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.5)

    agent = create_agent(
        model=model,  # 模型实例
        tools=[get_time],  # 工具列表
        system_prompt="你是一个时间助手。",  # 系统提示
        middleware=[add_timestamp],  # 中间件
        name="time_assistant",  # Agent 名称（用于多 Agent 系统）
    )
    #
    # 代码解释：name="time_assistant" 参数
    # =============================================
    # name 是 Agent 的标识符，主要用于多 Agent 系统中区分不同 Agent。
    #
    # 作用：
    # 1. 日志和调试时标识是哪个 Agent 在执行
    # 2. 在多 Agent 编排中，用于路由和引用
    # 3. LangFuse 等追踪工具会用 name 标记 trace
    #
    # 如果只有一个 Agent，name 可以省略。
    #
    print("\n--- 测试：完整参数 Agent ---")
    result = agent.invoke({"messages": [("user", "现在几点了?")]}, config={"callbacks": [langfuse_handler]})
    print(f"AI: {result['messages'][-1].content}")


# ===================== 示例3：ReAct模式深度理解 =====================
def example_3():
    """
    示例 3: ReAct 模式深度理解
    
    目标：深入理解 Reasoning + Acting 模式
    知识点：
    - Reasoning: 推理步骤
    - Acting: 行动（工具调用）
    - Observation: 观察结果
    - 循环直到完成
    """
    print("\n===== 示例 3: ReAct 模式深度理解 =====")

    #
    # ReAct 模式（Reasoning + Acting）图示
    # =============================================
    #
    # 用户提问: "广州现在的天气舒适吗?"
    #
    #     ┌──────────────────────────────────────┐
    #     │           用户输入问题               │
    #     └──────────────────────────────────────┘
    #                      ↓
    #     ┌──────────────────────────────────────┐
    #     │ ① Reasoning (推理)                   │
    #     │ AI 思考: 要判断天气是否舒适,          │
    #     │ 我需要知道温度和湿度。                │
    #     │ → 决定调用 get_temperature("广州")    │
    #     └──────────────────────────────────────┘
    #                      ↓
    #     ┌──────────────────────────────────────┐
    #     │ ② Acting (行动)                      │
    #     │ 执行工具调用: get_temperature("广州")│
    #     │ → 返回: "28"                         │
    #     └──────────────────────────────────────┘
    #                      ↓
    #     ┌──────────────────────────────────────┐
    #     │ ③ Observation (观察)                 │
    #     │ AI 收到工具结果: 温度 = 28℃          │
    #     │ AI 思考: 还需要湿度才能综合判断       │
    #     │ → 决定调用 get_humidity("广州")      │
    #     └──────────────────────────────────────┘
    #                      ↓
    #     ┌──────────────────────────────────────┐
    #     │ ② Acting (行动)                      │
    #     │ 执行工具调用: get_humidity("广州")   │
    #     │ → 返回: "80"                         │
    #     └──────────────────────────────────────┘
    #                      ↓
    #     ┌──────────────────────────────────────┐
    #     │ ③ Observation (观察)                 │
    #     │ AI 收到工具结果: 湿度 = 80%          │
    #     │ AI 思考: 温度 28℃ + 湿度 80% → 闷热  │
    #     └──────────────────────────────────────┘
    #                      ↓
    #     ┌──────────────────────────────────────┐
    #     │ ④ 最终回答 (Answer)                  │
    #     │ "广州目前温度 28℃, 湿度 80%, 属于闷热天气,
    #     │ 建议减少户外活动, 注意防暑降温。"      │
    #     └──────────────────────────────────────┘
    #
    # 简单循环表示:
    #
    #     Reasoning → Acting → Observation ──┐
    #               ↑                        │
    #               │  需要更多数据?         │
    #               └─────────── 是 ──────────┘
    #                           │
    #                           否
    #                           │
    #                           ↓
    #                      最终回答 (Answer)
    #
    #
    # 创建需要多步推理的场景
    @tool
    def get_temperature(city: str) -> str:
        """获取城市温度"""
        temps = {"北京": 25, "上海": 22, "广州": 28}
        return str(temps.get(city, 20))

    @tool
    def get_humidity(city: str) -> str:
        """获取城市湿度。"""
        humidity = {"北京": 40, "上海": 65, "广州": 80}
        return str(humidity.get(city, 50))

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[get_temperature, get_humidity],
        system_prompt="""你是一个天气分析师。
使用工具获取数据后，综合分析天气情况。
如果温度>25 且湿度>70 说明闷热
如果温度<15 说明寒冷
否则说明舒适。""",
    )

    print("\n--- ReAct 多步推理 ---")
    print("用户: 广州现在的天气舒适吗?")
    print("\n执行流程:")

    for step in agent.stream(
        {"messages": [("user", "广州现在的天气舒适吗? 需要综合温度和湿度分析")]},
        stream_mode="updates",
        version="v2",
        config={"callbacks": [langfuse_handler]},
    ):
        for node, data in step["data"].items():
            if data.get("messages"):
                msg = data["messages"][-1]
                print(f"\n步骤 [{node}]:")
                if msg.type == "ai":
                    #
                    # 代码解释：msg.tool_calls
                    # =============================================
                    # tool_calls 是 AIMessage 上的属性，是工具调用列表。
                    # 每个元素是一个字典：
                    # {
                    #     "name": "get_weather",  # 工具名
                    #     "args": {"city": "北京"},  # 工具参数
                    #     "id": "call_abc123",  # 调用ID
                    #     "type": "tool_call",  # 类型标识
                    # }
                    #
                    # 判断模型想调用工具还是直接回答:
                    # if msg.tool_calls: --> 想调用工具
                    # else:              --> 想直接回答
                    #
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f" → 调用工具: {msg.tool_calls[0]['name']}")
                    else:
                        print(f" → 最终答案: {msg.content[:100]}...")
                elif msg.type == "tool":
                    #
                    # 代码解释：msg.type == "ai" 和 msg.type == "tool"
                    # =============================================
                    # msg.type 是消息类型标识，常见值：
                    #
                    # "ai" --> AIMessage, 模型生成的消息
                    #         可能包含 tool_calls (想调用工具)
                    #         或 content (直接回答)
                    #
                    # "tool" --> ToolMessage, 工具执行结果
                    #           content 字段包含工具返回值
                    #
                    # "human" --> HumanMessage, 用户输入
                    # "system" --> SystemMessage, 系统提示
                    #
                    # 在 stream 中通过 msg.type 可以区分当前是
                    # 模型输出阶段还是工具返回阶段。
                    #
                    print(f" ← 工具返回: {msg.content}")


# ===================== 示例4：Agent状态管理 =====================
def example_4():
    """
    示例 4: Agent 状态管理
    
    目标：掌握 Agent 状态的读取和修改
    知识点：
    - 访问 Agent 内部状态
    - 自定义状态字段
    - 状态持久化基础
    """
    print("\n===== 示例 4: Agent 状态管理 =====")

    from langchain.agents.middleware import after_model

    #
    # 代码解释：CustomAgentState(AgentState) 与 NotRequired
    # =============================================
    # AgentState 是 LangGraph 的内置状态类型，
    # 默认包含 messages 字段（消息历史）。
    #
    # 通过继承 AgentState 可以添加自定义字段：
    # class CustomAgentState(AgentState):
    #     tool_call_count: NotRequired[int]
    #
    # NotRequired 的含义：
    # -- 来自 typing_extensions，类似 TypedDict 中的 Optional
    # -- 表示这个字段不是必须存在的，可以不存在于状态中
    # -- 访问时需要用 state.get("tool_call_count", 0) 而非 state["tool_call_count"]
    # -- 这样避免 KeyError，因为 Agent 初始化时可能没有这个字段
    #
    class CustomAgentState(AgentState):
        tool_call_count: NotRequired[int]
        user_preference: NotRequired[str]

    # 中间件：统计工具调用次数
    #
    # 代码解释：@after_model(state_schema=CustomAgentState)
    # =============================================
    # @after_model 是中间件装饰器，在模型生成消息后执行。
    #
    # 执行时机：
    # 模型生成消息 -> @after_model 中间件 -> 进入下一步
    #
    # state_schema 参数：
    # 指定中间件使用的状态类型。
    # 传入 CustomAgentState 后，state 参数就会有
    # tool_call_count 和 user_preference 的类型提示。
    #
    # 返回值：
    # 返回 dict -> 更新状态中的对应字段
    # 返回 None -> 不修改状态
    #
    # 其他中间件装饰器：
    # @before_model -> 模型调用前执行
    # @after_model -> 模型调用后执行
    # @before_tool -> 工具调用前执行
    # @after_tool -> 工具调用后执行
    #
    @after_model(state_schema=CustomAgentState)
    def track_tool_usage(state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
        """统计工具调用次数。"""
        # 计算本轮新增的工具调用
        tool_calls = 0
        for msg in state["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls += len(msg.tool_calls)

        current_count = state.get("tool_call_count", 0)
        if tool_calls > current_count:
            return {"tool_call_count": tool_calls}
        return None

    # 工具
    @tool
    def search_info(query: str) -> str:
        """搜索信息。"""
        return f"关于 '{query}' 的信息..."

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)
    agent = create_agent(
        model=model, tools=[search_info], system_prompt="你是一个信息查询助手。", middleware=[track_tool_usage]
    )

    print("\n--- 测试：状态追踪 ---")
    result = agent.invoke(
        {"messages": [("user", "搜索 Python"), ("user", "再搜索一下 Java")]},
        config={"callbacks": [langfuse_handler]}
    )
    print(f"工具调用次数: {result.get('tool_call_count', 0)}")
    print(f"消息总数: {len(result['messages'])}")


# ===================== 示例5：综合实战 - 研究助手Agent =====================
def example_5():
    """
    示例 5: 综合实战 - 研究助手 Agent

    目标：创建完整的研究助手 Agent
    知识点：
    - 多工具协作
    - 状态管理
    - 错误处理
    - 完整的 ReAct 流程
    """
    print("\n===== 示例 5: 综合实战 - 研究助手 Agent =====")

    # 工具 1: 文献搜索
    @tool
    def search_papers(topic: str, year: int = 2024) -> str:
        """
        搜索学术论文。

        参数:
            topic: 研究主题
            year: 发表年份
        """
        return f"已找到 {year} 年关于 '{topic}' 的10篇相关论文，示例：《XXX在AI领域的应用研究》..."

    # 工具 2: 数据分析
    @tool
    def analyze_data(metric: str, data: str) -> str:
        """
        分析数据指标。

        参数:
            metric: 分析指标（如 "平均值"、"最大值"）
            data: 数据（逗号分隔的数字）

        返回:
            分析结果
        """
        try:
            numbers = [float(x.strip()) for x in data.split(",")]
            if metric == "平均值":
                return f"平均值: {sum(numbers)/len(numbers):.2f}"
            elif metric == "最大值":
                return f"最大值: {max(numbers)}"
            elif metric == "最小值":
                return f"最小值: {min(numbers)}"
            else:
                return f"不支持的指标: {metric}"
        except Exception:
            return "数据格式错误"

    # 工具 3: 生成摘要
    @tool
    def generate_summary(text: str, max_length: int = 100) -> str:
        """
        生成文本摘要。

        参数:
            text: 原文内容
            max_length: 最大长度

        返回:
            摘要文本
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3, max_tokens=800)
    agent = create_agent(
        model=model,
        tools=[search_papers, analyze_data, generate_summary],
        system_prompt="""你是一个学术研究助手，可以：
- 搜索学术论文
- 分析实验数据
- 生成研究摘要

请根据用户需求，选择合适的工具完成研究任务。""",
    )

    # 测试场景 1: 文献搜索
    print("\n--- 测试 1: 文献搜索 ---")
    result1 = agent.invoke(
        {"messages": [("user", "搜索一下 2024 年关于 AI 的论文")]},
        config={"callbacks": [langfuse_handler]}
    )
    print(f"AI: {result1['messages'][-1].content}")

    # 测试场景 2: 数据分析
    print("\n--- 测试 2: 数据分析 ---")
    result2 = agent.invoke(
        {"messages": [("user", "帮我分析这组数据的平均值: 10, 20, 30, 40, 50")]},
        config={"callbacks": [langfuse_handler]}
    )
    print(f"AI: {result2['messages'][-1].content}")


def main(example_number: int):
    """运行指定的示例。"""
    print("=" * 60)
    print("第五课: Agent 架构与创建")
    print("=" * 60)

    examples = {1: example_1, 2: example_2, 3: example_3, 4: example_4, 5: example_5}

    if example_number in examples:
        examples[example_number]()
    else:
        print(f"错误: 示例编号 {example_number} 不存在")


if __name__ == "__main__":
    main(5)