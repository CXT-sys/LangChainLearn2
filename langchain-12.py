# 短期记忆管理
# ==========
# 第九课：记忆系统 - 短期记忆
# 模块：3.1
# 目标：掌握短期记忆（会话内记忆）的实现
# 知识点：
# -- 对话历史管理
# -- Checkpointer 基础
# -- 会话状态维护
# -- 记忆生命周期
# -- thread_id 详解
# -- checkpointer= 参数详解
# ......

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()
langfuse_handler = CallbackHandler()


def example_1():
    """
    示例 1：基础记忆 - 多轮对话
    目标：理解短期记忆的基本用法
    知识点：
    -- InMemorySaver 的使用
    -- thread_id 的作用
    -- 对话连续性
    """
    print("\n===== 示例 1：基础记忆 - 多轮对话 =====")

    # 创建记忆管理器
    checkpointer = InMemorySaver()
    # ————————————————————————————————
    # 代码解释：InMemorySaver
    # ————————————————————————————————
    # LangGraph 的内存检查点保存器。
    # 在内存中保存 Agent 的状态图，使多轮对话
    # 之间能保持上下文连续。
    #
    # 注意：数据存在内存中，进程退出后丢失。
    # 适合短期记忆（会话内），不适合长期记忆。
    # 生产环境应使用 AsyncSqliteStore 或 PostgresSaver。
    #

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.5)

    # 创建带记忆的 Agent
    agent = create_agent(model=model, system_prompt="你是一个聊天助手，记住我们聊过的内容。", checkpointer=checkpointer)
    #                                                         L checkpointer= 参数
    # ————————————————————————————————
    # 代码解释：checkpointer= 参数
    # ————————————————————————————————
    # create_agent 的可选参数，用于启用状态持久化。
    #
    # 没有 checkpointer 时：
    # → 每次 invoke() 都是独立的，Agent 不记得上次说了什么
    # → Agent 的状态在 invoke() 返回后完全丢失
    #
    # 有 checkpointer 时：
    # → Agent 的状态被保存到一个"检查点"中
    # → 下次用同一个 thread_id 调用时，从上次停止的地方继续
    # → 多轮对话之间 Agent 能记住上下文
    #

    # 使用相同的 thread_id 保持对话连续
    config = {"configurable": {"thread_id": "chat_001"}}
    #                             L thread_id
    # ————————————————————————————————
    # 代码解释：thread_id
    # ————————————————————————————————
    # 会话的"身份证号"。
    # 同一个 thread_id → 同一会话 → 共享状态
    # 不同 thread_id → 不同会话 → 状态隔离
    #
    # 为什么是 {"configurable": {"thread_id": ...}} 这种嵌套结构？
    # → config 是 LangGraph 的统一运行时配置
    # → "configurable" 是一个命名空间，里面可以放各种运行时参数
    # → 除了 thread_id，还可以有其他自定义参数
    # ————————————————————————————————

    print("\n--- 第一轮对话 ---")
    result1 = agent.invoke(
        {"messages": [("user", "我叫张三，今年 28 岁")]},
        config={"configurable": {"thread_id": "chat_001", "callbacks": [langfuse_handler]}},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 第二轮对话（AI 应该记住名字） ---")
    result2 = agent.invoke(
        {"messages": [("user", "你还记得我叫什么吗?")]},
        config={"configurable": {"thread_id": "chat_001", "callbacks": [langfuse_handler]}},
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_2():
    """
    示例 2：多会话隔离
    目标：掌握不同会话的记忆隔离
    知识点：
    -- thread_id 隔离
    -- 并发会话管理
    -- 会话切换
    """
    print("\n===== 示例 2：多会话隔离 =====")

    checkpointer = InMemorySaver()
    # ————————————————————————————————
    # 代码解释：InMemorySaver
    # LangGraph 的内存检查点保存器。
    # 在内存中保存 Agent 的状态图，使多轮对话
    # 之间能保持上下文连续。
    #
    # 短期记忆 vs 长期记忆：
    #   短期记忆（InMemorySaver）：
    #   - 会话期间记住用户说的话
    #   - 进程重启后丢失
    #   - 适合：多轮对话、任务跟踪
    #
    #   长期记忆（Store, 如 InMemoryStore/AsyncSqliteStore）：
    #   - 跨会话记住用户偏好
    #   - 可以持久化到磁盘
    #   - 适合：用户画像、历史记录、个性化
    # ————————————————————————————————

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.5)

    agent = create_agent(model=model, system_prompt="你是一个友好的助手。", checkpointer=checkpointer)

    # 会话 A：张三
    print("\n--- 会话 A：张三 ---")
    result_a1 = agent.invoke(
        {"messages": [("user", "我是张三，我喜欢编程")]},
        config={"configurable": {"thread_id": "user_zhangsan"}, "callbacks": [langfuse_handler]},
    )
    print(f"AI: {result_a1['messages'][-1].content}")

    # 会话 B：李四
    print("\n--- 会话 B：李四 ---")
    result_b1 = agent.invoke(
        {"messages": [("user", "我是李四，我喜欢音乐")]},
        config={"configurable": {"thread_id": "user_lisi"}, "callbacks": [langfuse_handler]},
    )
    print(f"AI: {result_b1['messages'][-1].content}")

    # 切换回会话 A（应该记得张三）
    print("\n--- 切换回会话 A：张三 ---")
    result_a2 = agent.invoke(
        {"messages": [("user", "我喜欢什么?")]},
        config={"configurable": {"thread_id": "user_zhangsan", "callbacks": [langfuse_handler]}},
    )
    print(f"AI: {result_a2['messages'][-1].content}")

    # 切换到会话 B（应该记得李四）
    print("\n--- 切换到会话 B：李四 ---")
    result_b2 = agent.invoke(
        {"messages": [("user", "我喜欢什么?")]},
        config={"configurable": {"thread_id": "user_lisi", "callbacks": [langfuse_handler]}},
    )
    print(f"AI: {result_b2['messages'][-1].content}")


def example_3():
    """
    示例 3：对话摘要与记忆优化
    目标：优化长对话的记忆管理
    知识点：
    -- 对话长度管理
    -- 消息压缩
    -- 关键信息提取
    """
    print("\n===== 示例 3：对话摘要与记忆优化 =====")

    from langchain.agents.middleware import before_model, AgentState
    from langgraph.runtime import Runtime
    from typing import Any

    # 中间件：限制消息数量，防止上下文过长
    @before_model(can_jump_to=["end"])
    # L can_jump_to= 参数
    # ————————————————————————————————
    # 代码解释：can_jump_to=["end"]
    # ————————————————————————————————
    # 指定中间件可以跳转到的节点列表。
    #
    # 当中间件返回 jump_to 时，Agent 图会跳过后续的
    # 节点，直接跳转到这里指定的节点。
    #
    # can_jump_to=["end"] 表示此中间件有权提前
    # 结束当前图的执行（例如：对话过长时拦截）。
    #
    # 如果不设置 can_jump_to，中间件即使返回 jump_to
    # 也不会生效，图会继续正常执行。
    # ————————————————————————————————
    def limit_conversation_length(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """当消息过多时，提醒用户开始新对话。"""
        from langchain.messages import AIMessage

        if len(state["messages"]) > 6:
            return {"messages": [AIMessage("对话历史较长，建议开始新的对话以保持上下文清晰。")], "jump_to": "end"}
        return None

    checkpointer = (
        InMemorySaver()
    )  # MemorySaver是老版本写法 LangGraph 在新版本中将 MemorySaver 重命名为 InMemorySaver，命名更清晰（InMemory 明确表达"内存中"的意思

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.5, max_tokens=300)

    agent = create_agent(
        model=model,
        system_prompt="你是一个聊天助手。如果对话很长，主动建议用户开启新对话。",
        checkpointer=checkpointer,
        middleware=[limit_conversation_length],
    )

    print("\n--- 多轮对话测试 ---")
    for i in range(4):
        result = agent.invoke(
            {"messages": [("user", f"这是第 {i+1} 轮对话")]},
            config={"configurable": {"thread_id": "long_chat_001"}, "callbacks": [langfuse_handler]},
        )
        print(f"轮次 {i+1}: {len(result['messages'])} 条消息,AI回复内容: {result['messages'][-1].content}")


def example_4():
    """
    示例 4：任务状态追踪
    目标：使用自定义 State + Command 追踪任务进度
    知识点：
    -- 自定义 AgentState 扩展
    -- 工具通过 Command 更新 State
    -- 中间件追踪状态变化
    -- 多轮对话中状态的持续
    """
    print("\n===== 示例 4：任务状态追踪 =====")

    from langchain.agents.middleware import AgentState, after_model
    from langchain.tools import ToolRuntime
    from langgraph.runtime import Runtime
    from langgraph.types import Command
    from langchain_core.messages import ToolMessage
    from typing import Any, NotRequired

    class TaskState(AgentState):
        """任务状态：追踪当前任务名和进度。"""
        current_step: NotRequired[int]
        task_name: NotRequired[str]

    # 中间件：每次模型响应后打印当前任务状态
    @after_model(state_schema=TaskState)
    def track_task_state(state: TaskState, runtime: Runtime) -> dict[str, Any] | None:
        task_name = state.get("task_name")
        current_step = state.get("current_step")
        if task_name:
            print(f"【状态】任务: {task_name} | 步骤: {current_step}")
        return None

    # 工具：开始任务（写入 State）
    @tool
    def start_task(task_name: str, runtime: ToolRuntime) -> Command:
        """
        开始一个新任务。
        参数:
            task_name: 任务名称
        """
        tool_call_id = runtime.tool_call_id
        return Command(
            update={
                "task_name": task_name,
                "current_step": 1,
                "messages": [
                    ToolMessage(
                        content=f"✅ 任务 '{task_name}' 已创建，当前进度: 第 1 步",
                        tool_call_id=tool_call_id
                    )
                ],
            }
        )

    # 工具：完成任务步骤（更新 State）
    @tool
    def complete_step(step_number: int, runtime: ToolRuntime) -> Command:
        """
        完成任务的某个步骤。
        参数:
            step_number: 步骤编号
        """
        state = runtime.state
        current_task = state.get("task_name", "未知任务")
        tool_call_id = runtime.tool_call_id
        next_step = step_number + 1
        return Command(
            update={
                "current_step": next_step,
                "messages": [
                    ToolMessage(
                        content=f"✅ 步骤 {step_number} 已完成, {current_task} 当前进度: 第 {next_step} 步",
                        tool_call_id=tool_call_id
                    )
                ],
            }
        )

    # 工具：查询当前进度（只读 State）
    @tool
    def get_progress(runtime: ToolRuntime) -> str:
        """获取当前任务进度。"""
        state = runtime.state
        task_name = state.get("task_name")
        current_step = state.get("current_step")
        if task_name:
            return f"当前任务: {task_name}, 已完成 {current_step} 步"
        return "暂无进行中的任务"

    checkpointer = InMemorySaver()

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    agent = create_agent(
        model=model,
        tools=[start_task, complete_step, get_progress],
        system_prompt="""你是一个任务管理助手。
帮助用户创建任务并追踪进度。
使用工具来更新任务状态。""",
        checkpointer=checkpointer,
        middleware=[track_task_state],
    )

    config = {"configurable": {"thread_id": "task_001"}}

    print("\n--- 第 1 轮: 创建任务 ---")
    result1 = agent.invoke(
        {"messages": [("user", "帮我创建一个'学习 Python'的任务")]},
        config={**config, "callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 第 2 轮: 完成第 1 步 ---")
    result2 = agent.invoke(
        {"messages": [("user", "完成第 1 步")]},
        config={**config, "callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")

    print("\n--- 第 3 轮: 查询进度 ---")
    result3 = agent.invoke(
        {"messages": [("user", "我现在进度如何?")]},
        config={**config, "callbacks": [langfuse_handler]},
    )
    print(f"AI: {result3['messages'][-1].content}")

    print("\n--- 第 4 轮: 完成第 2 步 ---")
    result4 = agent.invoke(
        {"messages": [("user", "完成第 2 步")]},
        config={**config, "callbacks": [langfuse_handler]},
    )
    print(f"AI: {result4['messages'][-1].content}")


def example_5():
    """
    示例 5：综合实战 - 预约管理系统
    目标：创建带记忆的预约管理 Agent
    知识点：
    -- 多轮对话收集信息
    -- 状态维护
    -- 完整的短期记忆应用
    """
    print("\n===== 示例 5：综合实战 - 预约管理系统 =====")

    checkpointer = InMemorySaver()

    # 预约存储
    appointments = {}
    checkpointer = InMemorySaver()

    # 预约存储
    appointments = {}

    @tool
    def create_appointment(date: str, time: str, purpose: str) -> str:
        """
        创建预约。
        参数:
            date: 日期 (如 "2024-01-15")
            time: 时间 (如 "14:00")
            purpose: 预约目的
        返回:
            预约结果
        """
        appointment_id = len(appointments) + 1
        appointments[appointment_id] = {"date": date, "time": time, "purpose": purpose}
        return f"✅ 预约已创建\nID: {appointment_id}\n时间: {date} {time}\n目的: {purpose}"

    @tool
    def list_appointments() -> str:
        """列出所有预约。"""
        if not appointments:
            return "暂无预约记录"

        output = "预约列表\n"
        for aid, apt in appointments.items():
            output += f"{aid}. {apt['date']} {apt['time']} -- {apt['purpose']}\n"
        return output

    @tool
    def cancel_appointment(appointment_id: int) -> str:
        """
        取消预约。
        参数:
            appointment_id: 预约 ID
        """
        if appointment_id in appointments:
            del appointments[appointment_id]
            return f"预约 {appointment_id} 已取消"
        return f"未找到预约 {appointment_id}"

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    agent = create_agent(
        model=model,
        tools=[create_appointment, list_appointments, cancel_appointment],
        system_prompt="""你是一个预约管理助手，可以:
    - 创建新的预约
    - 查看所有预约
    - 取消预约
    记住用户的预约信息，帮助管理日程。""",
        checkpointer=checkpointer,
    )

    print("\n--- 测试 1: 创建预约 ---")
    result1 = agent.invoke(
        {"messages": [("user", "帮我预约明天下午 3 点看医生")]},
        config={"configurable": {"thread_id": "appointment_001"}, "callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 查看预约 ---")
    result2 = agent.invoke(
        {"messages": [("user", "我有哪些预约?")]},
        config={"configurable": {"thread_id": "appointment_001"}, "callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_6():
    """
    示例 6: 持久化短期记忆（PostgresSaver）
    目标: 演示生产级 Checkpointer —— 进程重启后对话记忆不丢失
    知识点:
    -- PostgresSaver 初始化与 setup()
    -- 与 InMemorySaver 的对比
    -- thread_id 隔离 + 持久化
    -- 生产环境配置
    """
    print("\n===== 示例 6: 持久化短期记忆（PostgresSaver） =====")

    # ————————————————————————————————
    # 为什么需要 PostgresSaver?
    # ————————————————————————————————
    # InMemorySaver 的问题:
    # -- 数据在进程内存中，进程结束全部丢失
    # -- 无法跨部署、跨进程恢复对话
    # -- 只适合开发和测试
    #
    # PostgresSaver 的优势:
    # -- 数据持久化到 PostgreSQL，进程重启不丢失
    # -- 支持多进程/多实例共享状态
    # -- 适合生产环境的长期运行
    #
    # 但语义上它依然是"短期记忆"（Checkpointer）:
    # -- 按 thread_id 隔离不同会话
    # -- 存的是对话过程，不是用户画像
    # -- Agent 不会跨 thread_id "认识"用户
    # ————————————————————————————————

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.pool import ConnectionPool
    from psycopg.rows import dict_row

    DB_URI = "postgresql://xiaoming:123123@localhost:5433/ai_memory"

    try:
        # 直接连接（单线程、开发测试用）
        # conn = psycopg.connect(DB_URI, autocommit=True)
        # 1. 创建连接池（生产环境推荐）
        # 如果是本地单线程测试，也可以直接用 psycopg.connect() 替代
        connection_kwargs = {
            "autocommit": True,  # 【必须】关闭隐式事务。因为 setup() 需要执行 CREATE INDEX CONCURRENTLY，该命令不能在事务块内运行
            "prepare_threshold": 0,  # 【推荐】关闭预编译语句，防止连接池在多进程/重启后出现 "Cached plan must not change result type" 报错
            "row_factory": dict_row,  # 【推荐】将查询结果转为字典格式（如 `{"thread_id": "xxx"}`），方便 LangGraph 底层按字段名读取
        }
        pool = ConnectionPool(
            conninfo=DB_URI,
            max_size=5,  # 连接池最大连接数，限制并发请求数量，保护数据库连接资源
            kwargs=connection_kwargs,
        )

        # 2. 初始化 PostgresSaver
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()  # 首次运行必须调用，创建 checkpoint 表结构

        # ————————————————————————————————
        # setup() 做了什么?
        # ————————————————————————————————
        # 在 PostgreSQL 中创建 checkpoint 相关表:
        # -- checkpoints: 存储每个 thread 的检查点
        # -- checkpoint_writes: 存储待写入的更新
        # -- checkpoint_blobs: 存储大对象
        # 表已存在时不会重复创建（幂等操作）
        # ————————————————————————————————

        model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.5)

        agent = create_agent(
            model=model,
            system_prompt="你是一个聊天助手，记住我们聊过的内容。",
            checkpointer=checkpointer,
        )

        print("\n--- 第 1 轮: 自我介绍（写入 PostgreSQL） ---")
        result1 = agent.invoke(
            {"messages": [("user", "我叫张三，今年 28 岁，喜欢编程")]},
            config={"configurable": {"thread_id": "pg_chat_001"}, "callbacks": [langfuse_handler]},
        )
        print(f"AI: {result1['messages'][-1].content}")

        print("\n--- 第 2 轮: 测试记忆（从 PostgreSQL 读取） ---")
        result2 = agent.invoke(
            {"messages": [("user", "你还记得我叫什么、喜欢什么吗?")]},
            config={"configurable": {"thread_id": "pg_chat_001"}, "callbacks": [langfuse_handler]},
        )
        print(f"AI: {result2['messages'][-1].content}")

        print("\n✅ 对话状态已持久化到 PostgreSQL")
        print("  即使关闭脚本、重启机器，再次运行本示例")
        print("  用相同的 thread_id 调用，Agent 依然记得之前的对话!")

    except Exception as e:
        print(f"❌ 连接数据库失败: {e}")
        print("  请确保 PostgreSQL 正在运行: ")
        print(
            "  docker run -d --name langgraph-pg -e POSTGRES_USER=xiaoming "
            "-e POSTGRES_PASSWORD=123123 -e POSTGRES_DB=ai_memory -p 5433:5432 postgres"
        )


def main(example_number: int):
    """运行指定的示例。"""
    print("=" * 60)
    print("第九课: 短期记忆管理")
    print("=" * 60)

    examples = {
        1: example_1,
        2: example_2,
        3: example_3,
        4: example_4,
        5: example_5,
        6: example_6
    }

    if example_number in examples:
        examples[example_number]()
    else:
        print(f"错误: 示例编号 {example_number} 不存在")


if __name__ == "__main__":
    main(6)