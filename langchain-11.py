"""
上下文工程实战
==============
第八课 LangChain 核心应用 - 上下文工程
模块: 2.5
目标: 掌握上下文工程的核心概念和实践
知识点:
-- 静态运行时上下文 (State)
-- 动态运行时上下文 (State)
-- 跨对话上下文 (Store)
-- 可变性与生命周期管理
-- Checkpointer vs Store 架构对比
-- Store API 详解
"""

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from dataclasses import dataclass
from typing import Any, NotRequired
from dotenv import load_dotenv

load_dotenv()
langfuse_handler = CallbackHandler()


def example_1():
    """
    示例 1: 静态运行时上下文
    目标: 理解静态上下文的概念和用途
    知识点:
    -- 定义上下文 Schema
    -- 传递上下文给 Agent
    -- 工具中访问上下文
    """
    print("\n===== 示例 1: 静态运行时上下文 =====")

    @dataclass
    class AppContext:
        """应用上下文：包含用户信息和设置。"""
        user_id: str
        user_name: str
        language: str = "zh"
        timezone: str = "Asia/Shanghai"

    # 工具：访问上下文
    @tool
    def get_user_greeting(runtime: ToolRuntime[AppContext]) -> str:
        """获取用户个性化问候语。"""
        ctx = runtime.context
        return f"您好，{ctx.user_name}！您的用户 ID 是 {ctx.user_id}，时区：{ctx.timezone}"

    # 代码解释: ToolRuntime[AppContext] 泛型类型
    #
    # ToolRuntime 是一个泛型类，类型参数指定了
    # runtime.context 的具体类型。
    #
    # ToolRuntime[AppContext] 表示：
    # runtime.context 的类型是 AppContext
    # 这提供了类型安全——IDE 可以自动补全 user_id、
    # user_name 等属性，编译器也能检查类型错误。
    #
    # 如果不写泛型参数，runtime.context 的类型会是
    # Any 或 Optional，失去类型检查的好处。

    # AgentState / Runtime / ToolRuntime 对比
    #
    # 对比项         AgentState           Runtime          ToolRuntime
    # 含义           Agent 的内存状态     Agent 执行时的运行环境  工具执行时的运行环境
    #               (TypedDict 字典)    (对象)           (对象)
    # 生命周期       整个 Agent 执行期间  单次 invoke/stream   单次工具调用期间
    # 内置属性       1. messages (必需)   1. context         1. context
    #               2. jump_to (可选)    2. model           2. state
    #               3. structured_response                   3. stream_writer
    #
    # 使用场景       中间件读写 Agent 状态  中间件访问上下文     工具内访问 Agent 状态
    # 创建方式       LangChain 自动创建   LangChain 自动创建   LangChain 自动注入
    # 使用示例       @before_model       @before_model       @tool
    #               def mw(state, rt):   def mw(state, rt):   def tool(x, rt):
    #                   msgs = state["messages"]  ctx = rt.context  ctx = rt.context
    #                                                       msgs = rt.state["messages"]

    @tool
    def get_user_settings(runtime: ToolRuntime[AppContext]) -> str:
        """获取用户当前设置。"""
        # 打印 runtime 所有键名和对应的值
        print("\n--- runtime 所有键名和值 ---")
        print(f"runtime 类型: {type(runtime).__name__}")
        for key, value in runtime.__dict__.items():
            print(f"\n{key}: {value}")
        print("-" * 50)

        ctx = runtime.context
        return f"用户设置:\n- 语言: {ctx.language}\n- 时区: {ctx.timezone}"

    # 中间件：追踪状态变化
    from langchain.agents.middleware import AgentState, before_model
    from langgraph.runtime import Runtime

    @before_model
    def print_state(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """记录状态变化。"""
        # 打印 state 和 runtime 所有字段和值
        print("\n--- state 所有字段和值 ---")
        print(f"state 类型: {type(state).__name__}")
        for key, value in state.items():
            print(f"\n{key}: {value}")

        print("\n--- runtime 所有字段和值 ---")
        print(f"runtime 类型: {type(runtime).__name__}")
        print(f"runtime 完整内容: {runtime}")
        # 动态获取 Runtime 的所有字段 (Runtime 是 Pydantic 模型)
        print(f"{runtime.context}")
        print(f"{runtime.context.user_id}")
        return None

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[get_user_greeting, get_user_settings],
        system_prompt="你是一个个性化助手，可以根据用户设置提供定制化服务。",
        context_schema=AppContext,
        middleware=[print_state],
    )

    # 代码解释: context_schema 参数
    #
    # context_schema 告诉 create_agent:
    # 1. Agent 的所有工具都可以通过 runtime.context
    #    访问这个类型的实例
    # 2. 在 agent.invoke() 时，需要用 context=
    #    参数传入该类型的实例
    #
    # context_schema 相当于给每个工具函数提供一个
    # "环境配置"，比如当前用户是谁、语言偏好等。

    # 调用时传入上下文
    result = agent.invoke(
        {"messages": [("user", "我的设置是什么?")]},
        config={"callbacks": [langfuse_handler]},
        context=AppContext(user_id="U001", user_name="张三", language="zh", timezone="Asia/Shanghai"),
    )
    print(f"AI: {result['messages'][-1].content}")


def example_2():
    """
    示例 2: 动态运行时上下文 (State)
    目标: 掌握 State 的使用
    知识点:
    -- 读取 State
    -- 更新 State
    -- State 的生命周期
    """
    print("\n===== 示例 2: 动态运行时上下文 (State) =====")

    from langchain.agents.middleware import AgentState, after_model
    from langgraph.runtime import Runtime
    from langgraph.types import Command
    from typing import Any, NotRequired

    # 代码解释: Command from langgraph.types
    #
    # Command 是 LangGraph 用于从工具内部向图发送
    # 指令的返回类型。常见用途:
    #
    # Command(update={"key": value})
    #   → 更新 State 中指定字段
    # Command(goto="node_name")
    #   → 跳转到指定节点
    #
    # 工具返回 Command 后，LangGraph 会:
    # 1. 执行 update 中指定的状态更新
    # 2. 继续图的执行流程
    #
    # 这是工具修改 AgentState 的标准方式——
    # 工具不能直接修改 state，必须通过 Command
    # 声明更新意图，由 LangGraph 框架执行。

    class AppState(AgentState):
        """自定义应用状态。"""
        conversation_topic: NotRequired[str]
        user_preferences: NotRequired[dict]

    # 工具：访问 State
    @tool
    def get_state_info(runtime: ToolRuntime) -> str:
        """获取当前对话状态信息。"""
        state = runtime.state
        topic = state.get("conversation_topic", "未设置")
        return f"当前话题: {topic}"

    # 工具：更新 State
    @tool
    def set_topic(new_topic: str, runtime: ToolRuntime) -> Command:
        """设置对话话题。

        参数:
            new_topic: 新的话题

        返回:
            状态更新命令
        """
        # Command 返回规范 (create_agent 场景)
        # 当工具需要同时更新状态和响应 LLM 时:
        # - "conversation_topic": 更新自定义状态字段
        # - "messages": 添加工具响应消息到对话历史
        # 这样 LangGraph 能正确处理两个更新。
        tool_call_id = runtime.tool_call_id
        return Command(
            update={
                "conversation_topic": new_topic,
                "messages": [
                    ToolMessage(
                        content=f"✅ 话题已设置为: {new_topic}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    # 中间件：追踪状态变化
    @after_model(state_schema=AppState)
    def track_state(state: AppState, runtime: Runtime) -> dict[str, Any] | None:
        """记录状态变化。"""
        topic = state.get("conversation_topic")
        if topic:
            print(f"[状态追踪] 当前话题: {topic}")
        return None

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[get_state_info, set_topic],
        system_prompt="你是一个对话管理助手，可以设置和查询当前对话话题。",
        middleware=[track_state],
    )

    print("\n--- 测试 1: 查询当前状态 ---")
    result1 = agent.invoke(
        {"messages": [("user", "当前的话题是什么?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 设置话题 ---")
    result2 = agent.invoke(
        {"messages": [("user", "把话题设置为 'Python 编程'")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_3():
    """
    示例 3: 跨对话上下文 (Store)
    目标: 掌握持久化存储的使用
    知识点:
    -- Store 的读写
    -- 跨会话数据共享
    -- 用户偏好存储
    """
    print("\n===== 示例 3: 跨对话上下文 (Store) =====")

    from langgraph.store.memory import InMemoryStore

    # 创建内存存储
    store = InMemoryStore()

    # 代码解释: InMemoryStore
    #
    # InMemoryStore 是 LangGraph 提供的内存存储实现。
    #
    # 重要警告:
    # - 数据仅存在于内存中，进程退出后全部丢失。
    # - 适合开发和测试，不适合生产环境。
    #
    # 生产环境应使用:
    # - AsyncSqliteStore (SQLite 文件持久化)
    # - PostgresStore (PostgreSQL 数据库)
    #
    # 但 InMemoryStore 的 API 与持久化 Store
    # 完全相同，所以学习成本可以迁移。

    # — Checkpointer vs Store 架构对比 —
    #
    # LangGraph 提供两种状态管理机制:
    #
    #               Checkpointer          Store
    # 代表类        InMemorySaver         InMemoryStore
    #               MemorySaver
    # 作用范围      单次会话内 (多轮对话)  跨会话 (长期记忆)
    # 存储内容      Agent 的完整状态图     任意业务数据
    #               (消息、节点状态)       (用户偏好、历史)
    # 标识方式      thread_id             namespace + key
    # 使用场景      "记住这次对话"         "记住用户长期偏好"
    #               "中断后继续"          "下次登录仍记得"
    # 传入方式      create_agent(         create_agent(
    #               checkpointer=)        store=)
    #
    # 简单类比:
    # - Checkpointer = 浏览器的 Session
    # - Store = 浏览器的 LocalStorage

    # 预存一些用户偏好
    store.put(("preferences",), "user_001", {"theme": "dark", "language": "zh", "notifications": True})
    store.put(("preferences",), "user_002", {"theme": "light", "language": "en", "notifications": False})

    # 代码解释: store.put()
    #
    # Store 是 LangGraph 的分层键值存储。
    #
    # put(namespace, key, value) 参数:
    # - namespace: 元组，类似文件夹路径
    # - key: 字符串，条目名称
    # - value: 任意 JSON 可序列化数据
    #
    # 示例中 ("preferences",) 是根文件夹，
    # key="user_001" 是文件名，
    # value 是实际的字典数据。
    #
    # 相同 namespace + key 会覆盖旧值，
    # 适合存储会更新的偏好设置/Profile。
    # 不同 key 则是新增，适合存历史记录。

    @dataclass
    class UserContext:
        user_id: str

    @tool
    def get_user_preferences(runtime: ToolRuntime[UserContext]) -> str:
        """获取用户偏好设置。
        从 Store 读取持久化数据。
        """
        user_id = runtime.context.user_id
        pref = store.get(("preferences",), user_id)
        print(pref)

        # 代码解释: store.get()
        #
        # store.get(namespace, key) 返回 StoreItem 对象，
        # 不是原始值！常见误区：直接打印 pref 会得到
        # 一个 StoreItem，而不是存入的字典。
        #
        # StoreItem 的属性:
        # .value → 存入的原始数据 (dict/list/str等)
        # .key → key 字符串
        # .namespace → namespace 元组
        # .updated_at → 最后更新时间戳
        # .created_at → 创建时间戳
        #
        # 如果 key 不存在，返回 None (需判空)。
        # 示例: pref.value.items() 才是真正的字典。

        if pref:
            return f"用户 {user_id} 的偏好:\n" + "\n".join(f"- {k}: {v}" for k, v in pref.value.items())
        return f"用户 {user_id} 暂无偏好设置"

    @tool
    def save_user_preference(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
        """保存用户偏好设置。

        参数:
            key: 偏好键名
            value: 偏好值

        返回:
            保存结果
        """
        user_id = runtime.context.user_id
        pref = store.get(("preferences",), user_id)
        preferences = pref.value if pref else {}
        preferences[key] = value
        store.put(("preferences",), user_id, preferences)
        return f"已保存: {key} = {value}"

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[get_user_preferences, save_user_preference],
        system_prompt="你是一个用户偏好管理助手。",
        context_schema=UserContext,
        store=store,
    )

    # 代码解释: store= 参数 in create_agent
    #
    # 将 Store 实例传给 create_agent 后:
    # 1. Agent 内部执行的工具函数可以访问
    #    同一个 store 实例 (通过闭包或注入)
    # 2. 确保所有工具操作的是同一份数据
    #
    # 与 checkpointer= 的区别:
    # checkpointer → 自动管理，框架自动读写
    #                对话状态，无需手动调用
    # store        → 手动管理，工具内需要
    #                手动调用 store.get/put
    #
    # 两者可以同时传入，互不干扰。

    print("\n--- 测试 1: 查询偏好 ---")
    result1 = agent.invoke(
        {"messages": [("user", "我的偏好设置是什么?")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_id="user_001"),
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 保存新偏好 ---")
    result2 = agent.invoke(
        {"messages": [("user", "帮我把字体大小设置为 '大'")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_id="user_001"),
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_4():
    """
    示例 4: 上下文生命周期管理
    目标: 理解不同上下文的生命周期
    知识点:
    -- Context: 单次调用生命周期
    -- State: 对话生命周期
    -- Store: 永久生命周期
    """
    print("\n===== 示例 4: 上下文生命周期管理 =====")

    @dataclass
    class SessionContext:
        """会话上下文：单次调用有效。"""
        session_id: str
        request_source: str  # "web", "mobile", "api"

    # Store 用于永久存储
    lifecycle_store = InMemoryStore()

    # 代码解释: InMemoryStore (再次使用)
    #
    # 同上，InMemoryStore 是非持久化的。
    # 这里命名为 lifecycle_store 是为了语义清晰，
    # 实际类型和行为与 example_3 中的 store 完全相同。

    @tool
    def log_request(runtime: ToolRuntime[SessionContext]) -> str:
        """记录请求日志。
        Context 在单次请求中有效。
        """
        ctx = runtime.context
        message = f"[{ctx.session_id}] 来源: {ctx.request_source}"
        # 存储到 Store (永久有效)
        lifecycle_store.put(("logs",), ctx.session_id, {"source": ctx.request_source})
        return f"请求已记录: {message}"

    @tool
    def get_request_history(runtime: ToolRuntime[SessionContext]) -> str:
        """获取请求历史 (从 Store)。"""
        # 代码解释: store.search()
        #
        # store.search(namespace_prefix) 按 namespace
        # 前缀搜索所有匹配的条目。
        #
        # 参数:
        # namespace_prefix: 元组，前缀匹配
        # ("logs",) 会匹配所有 ("logs",) 下的条目
        #
        # 返回:
        # List[StoreItem] - StoreItem 列表
        # 每个 item 同样有 .value、.key 等属性
        #
        # 与 store.get() 的区别:
        # get → 需要精确的 namespace + key
        # search → 按前缀批量获取，不需要知道 key
        #
        # 适用场景:
        # "获取某个用户的所有日志"
        # "列出所有某种类型的记录"

        print(runtime)
        print("\n=== =====")
        print(lifecycle_store)
        logs = lifecycle_store.search(("logs",))
        if not logs:
            return "暂无历史记录"
        return f"共 {len(logs)} 条记录"

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[log_request, get_request_history],
        system_prompt="你是一个请求日志管理助手。",
        context_schema=SessionContext,
        store=lifecycle_store,
    )

    print("\n--- 测试 1: 记录请求 ---")
    result1 = agent.invoke(
        {"messages": [("user", "记录这次请求")]},
        config={"callbacks": [langfuse_handler]},
        context=SessionContext(session_id="S001", request_source="web"),
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 查看历史 ---")
    result2 = agent.invoke(
        {"messages": [("user", "查看请求历史")]},
        config={"callbacks": [langfuse_handler]},
        context=SessionContext(session_id="S002", request_source="mobile"),
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_5():
    """
    示例 5: 综合实战 - 个性化推荐系统
    目标: 综合运用所有上下文概念
    知识点:
    -- 多层上下文整合
    -- 用户画像构建
    -- 个性化推荐
    -- 完整生产级实现
    """
    print("\n===== 示例 5: 综合实战 - 个性化推荐系统 =====")

    @dataclass
    class UserContext:
        """用户上下文。"""
        user_id: str
        user_level: str  # "new", "regular", "vip"

    # 用户数据存储
    user_store = InMemoryStore()

    # 商品数据库
    products_db = [
        {"id": 1, "name": "Python 入门教程", "category": "编程", "price": 99, "level": "new"},
        {"id": 2, "name": "Python 进阶实战", "category": "编程", "price": 199, "level": "regular"},
        {"id": 3, "name": "AI 深度学习", "category": "AI", "price": 299, "level": "regular"},
        {"id": 4, "name": "VIP 专属课程", "category": "精品", "price": 599, "level": "vip"},
        {"id": 5, "name": "机器学习实战", "category": "AI", "price": 249, "level": "regular"},
    ]

    @tool
    def view_history(runtime: ToolRuntime[UserContext]) -> str:
        """获取用户浏览历史。"""
        user_id = runtime.context.user_id
        history = user_store.get(("history",), user_id)
        if history:
            return f"浏览历史: {', '.join(history.value)}"
        return "暂无浏览历史"

    @tool
    def record_view(product_name: str, runtime: ToolRuntime[UserContext]) -> str:
        """记录浏览记录。

        参数:
            product_name: 商品名称
        """
        user_id = runtime.context.user_id
        history = user_store.get(("history",), user_id)
        views = history.value if history else []
        views.append(product_name)
        user_store.put(("history",), user_id, views)
        return f"已记录浏览: {product_name}"

    @tool
    def get_recommendations(runtime: ToolRuntime[UserContext]) -> str:
        """根据用户等级和历史获取推荐。"""
        ctx = runtime.context
        user_id = ctx.user_id
        user_level = ctx.user_level

        # 获取历史
        history = user_store.get(("history",), user_id)
        viewed = history.value if history else []

        # 根据等级过滤
        level_products = [p for p in products_db if p["level"] == user_level]

        # 排除已浏览的
        recommended = [p for p in level_products if p["name"] not in viewed]

        if not recommended:
            return "暂无推荐商品"

        output = f"为您推荐 ({user_level} 专属):\n"
        for p in recommended[:3]:
            output += f"- {p['name']}: ￥{p['price']} ({p['category']})\n"
        return output

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    agent = create_agent(
        model=model,
        tools=[view_history, record_view, get_recommendations],
        system_prompt="""你是一个个性化商品推荐助手。
- 根据用户等级推荐商品
- 记录用户浏览历史
- 推荐未浏览过的商品""",
        context_schema=UserContext,
        store=user_store,
    )

    print("\n--- 测试 1: 新用户体验 ---")
    result1 = agent.invoke(
        {"messages": [("user", "有什么推荐的?")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001", user_level="regular"),
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 浏览商品 ---")
    result2 = agent.invoke(
        {"messages": [("user", "我想看看 Python 进阶实战")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001", user_level="regular"),
    )
    print(f"AI: {result2['messages'][-1].content}")

    print("\n--- 测试 3: 问浏览历史 ---")
    result2 = agent.invoke(
        {"messages": [("user", "我看过哪些?")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001", user_level="regular"),
    )
    print(f"AI: {result2['messages'][-1].content}")

    print("\n--- 测试 4: 再次推荐 (排除已浏览) ---")
    result3 = agent.invoke(
        {"messages": [("user", "还有什么其他推荐?")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001", user_level="regular"),
    )
    print(f"AI: {result3['messages'][-1].content}")


def example_6():
    """
    示例 6: Store 持久化演示 (SqliteStore)
    目标: 演示真正的重启不丢失数据
    知识点:
    -- SqliteStore 初始化
    -- 数据持久化到硬盘
    -- 重启后数据依然存在
    """
    print("\n===== 示例 6: Store 持久化演示 (SqliteStore) =====")

    # SqliteStore vs InMemoryStore
    #
    # InMemoryStore:
    # - 数据存在内存中
    # - 进程结束，数据丢失
    # - 适合开发和测试
    #
    # SqliteStore:
    # - 数据存在本地硬盘文件 (.db)
    # - 进程重启，数据依然存在
    # - 适合生产环境的长期存储
    #
    # 两者的 API 完全相同 (put/get/search)，
    # 只需更换类名即可。

    from langgraph.store.sqlite import SqliteStore
    import sqlite3

    # 创建持久化 Store
    # SqliteStore 需要一个 sqlite3.Connection 对象
    db_path = "./lesson08_sqlite_store.db"
    # check_same_thread=False: 允许跨线程使用 (LangGraph 使用线程池)
    # isolation_level=None: 禁用隐式事务，由 SqliteStore 自行管理
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn=conn)

    @dataclass
    class UserProfile:
        """用户配置上下文。"""
        user_id: str
        username: str

    @tool
    def save_profile(name: str, age: int, runtime: ToolRuntime[UserProfile]) -> str:
        """保存用户资料。

        参数:
            name: 用户姓名
            age: 用户年龄
        """
        user_id = runtime.context.user_id
        profile = {"name": name, "age": age, "saved_at": "2026-04-11"}
        # 存入 SqliteStore
        store.put(("profiles",), user_id, profile)
        return f"✅ 已保存 {name} 的资料（持久化到硬盘）"

    @tool
    def load_profile(runtime: ToolRuntime[UserProfile]) -> str:
        """获取用户资料。"""
        user_id = runtime.context.user_id
        profile = store.get(("profiles",), user_id)
        if profile:
            data = profile.value
            return f"用户 {data['name']}, 年龄 {data['age']}, 保存时间: {data['saved_at']}"
        return "暂无资料，请先保存。"

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[save_profile, load_profile],
        system_prompt="你是一个用户资料管理助手。",
        context_schema=UserProfile,
        store=store,
    )

    print("\n--- 测试 1: 保存资料（写入硬盘） ---")
    result1 = agent.invoke(
        {"messages": [("user", "保存我的资料：姓名张三，年龄 28")]},
        config={"callbacks": [langfuse_handler]},
        context=UserProfile(user_id="U001", username="张三"),
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 读取资料（从硬盘读取） ---")
    result2 = agent.invoke(
        {"messages": [("user", "我的资料是什么?")]},
        config={"callbacks": [langfuse_handler]},
        context=UserProfile(user_id="U001", username="张三"),
    )
    print(f"AI: {result2['messages'][-1].content}")

    print("\n--- 验证持久化 ---")
    print("✅ 数据已保存在 ./lesson08_sqlite_store.db")
    print("  即使重启代码，再次运行本示例，数据依然存在!")


def example_7():
    """
    示例 7: Store 持久化演示 (PostgreSQL)
    目标: 演示使用生产级数据库作为 Store
    知识点:
    -- PostgresStore 初始化
    -- 生产环境配置
    -- 与 SqliteStore 的对比
    """
    print("\n===== 示例 7: Store 持久化演示 (PostgreSQL) =====")

    # 注意事项:
    # 1. PostgresStore 已内置在 langgraph 包中 (无需额外安装包)
    # 2. 只需要安装: pip install psycopg psycopg-pool
    # 3. 需要本地或远程 PostgreSQL 数据库
    #    docker run -d \
    #    --name langgraph-pg \
    #    -e POSTGRES_USER=xiaoming \
    #    -e POSTGRES_PASSWORD=123123 \
    #    -e POSTGRES_DB=ai_memory \
    #    -p 5433:5432 \
    #    postgres

    from langgraph.store.postgres import PostgresStore

    # PostgresStore vs SqliteStore
    #
    # SqliteStore:
    # - 单文件数据库，适合轻量级应用
    # - 并发写入性能较弱
    #
    # PostgresStore:
    # - 生产级关系型数据库
    # - 支持高并发读写，支持分布式部署
    # - 适合企业级 AI 应用

    # 配置你的 PostgreSQL 连接字符串
    # 注意: 端口 5433 映射到容器内的 5432
    # 用户名/密码/数据库名需与 docker run 时的 -e 参数一致
    DB_URI = "postgresql://xiaoming:123123@localhost:5433/ai_memory"

    try:
        # 1. 创建连接对象 (必须使用 psycopg.connect)
        # autocommit=True: 允许 setup() 执行 CREATE INDEX CONCURRENTLY
        # (该语句不能在事务块内运行)
        import psycopg
        conn = psycopg.connect(DB_URI, autocommit=True)

        # 2. 初始化 Store
        store = PostgresStore(conn=conn)

        # 3. 初始化表结构 (首次运行必须调用)
        # setup() 会创建 store 表 (prefix, key, value 等字段)
        # 如果表已存在, setup() 不会重复创建 (幂等操作)
        store.setup()

        @dataclass
        class UserProfile:
            user_id: str
            username: str

        @tool
        def save_profile_pg(name: str, runtime: ToolRuntime[UserProfile]) -> str:
            """保存用户资料到 PostgreSQL。

            参数:
                name: 用户姓名
            """
            user_id = runtime.context.user_id
            store.put(("profiles",), user_id, {"name": name})
            return f"✅ 已将 {name} 存入 PostgreSQL"

        @tool
        def load_profile_pg(runtime: ToolRuntime[UserProfile]) -> str:
            """从 PostgreSQL 读取用户资料。"""
            user_id = runtime.context.user_id
            profile = store.get(("profiles",), user_id)
            if profile:
                return f"用户: {profile.value['name']}"
            return "暂无记录"

        model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

        agent = create_agent(
            model=model,
            tools=[save_profile_pg, load_profile_pg],
            system_prompt="你是一个使用 PostgreSQL 作为记忆存储的助手。",
            context_schema=UserProfile,
            store=store,
        )

        print("\n--- 测试: 写入并读取 ---")
        result = agent.invoke(
            {"messages": [("user", "保存我叫王五，然后读取我的名字")]},
            config={"callbacks": [langfuse_handler]},
            context=UserProfile(user_id="U002", username="王五"),
        )
        print(f"AI: {result['messages'][-1].content}")
        print("\n✅ 数据已持久化到 PostgreSQL 数据库!")

    except Exception as e:
        print(f"❌ 连接数据库失败: {e}")
        print("  请检查数据库连接设置及依赖安装情况。")


def main(example_number: int):
    """运行指定的示例。"""
    print("=" * 60)
    print("第八课: 上下文工程实战")
    print("=" * 60)

    examples = {
        1: example_1,
        2: example_2,
        3: example_3,
        4: example_4,
        5: example_5,
        6: example_6,
        7: example_7,
    }

    if example_number in examples:
        examples[example_number]()
    else:
        print(f"错误: 示例编号 {example_number} 不存在")


if __name__ == "__main__":
    main(7)


# Store 确实是实现 Agent 长期记忆 (Long-Term Memory) 的关键组件。
# 不过，它和 checkpointer 实现的"记忆"在功能上有明确的分工:

# 1. Checkpointer: 短期记忆 ("工作记忆")
# 作用: 记住当前对话说了什么 (上下文)。
# 类比: 就像你和我正在聊天。我如果忘了上一句说了什么，就没法跟你交流了。
# 适用场景: 多轮对话、被打断后恢复。
# 生命周期: 一次对话结束，它的任务就完成了。

# 2. Store: 长期记忆 ("永久记忆")
# 作用: 记住跨对话的信息 (用户喜好、历史总结、特定事实)。
# 类比: 就像笔记本。哪怕我睡了一觉 (程序重启)，或者你过了一个月再来找我 (新的 thread_id)，只要翻开笔记本 (store)，我依然记得你叫"张三"，喜欢"黑色"。
# 适用场景:
# 用户画像: 记住用户的名字、职业、偏好。
# 跨会话总结: 把上一次聊天的结论总结后存入 Store，下次对话开始时读取，Agent 就能说: "上次我们聊到...，结论是..."。
# 事实存储: 用户告诉 Agent"我家有只狗叫旺财"，下次聊天时 Agent 仍能知道并回答关于旺财的问题。

# 总结
# 如果你想让 Agent "接得上话" (知道上下文)，用 Checkpointer。
# 如果你想让 Agent "记性好" (认识你是谁，记得你的喜好)，用 Store。
# 在高级的 Agent 架构中，通常是两者配合使用: 用 Checkpointer 维持当前对话流，用 Store 在对话结束时或中间过程中提取关键信息并长期保存。