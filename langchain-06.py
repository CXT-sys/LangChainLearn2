"""
工具定义与高级用法
第三课：核心组件 -- 工具(Tools)
模块: 1.3
目标: 掌握工具的高级定义技巧和运行时上下文访问
知识点:
--- 工具定义的最佳实践
--- 自定义工具名称和描述
--- 高级参数 schema (Pydantic)
--- 运行时上下文访问 (ToolRuntime)
--- 工具错误处理
"""

import warnings
import logging

# 过滤 Pydantic 和 LangGraph 的序列化警告（已知行为，不影响功能）
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*Pydantic serializer.*")
warnings.filterwarnings("ignore", message=".*Pre-structured output.*")
logging.getLogger("pydantic").setLevel(logging.ERROR)

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, Field
from typing import Literal
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载环境变量（从 .env 文件）
load_dotenv()

# 初始化 Langfuse 回调处理器（用于追踪和监控）
langfuse_handler = CallbackHandler()


def example_1():
    """
    示例 1: 工具定义的最佳实践
    目标：掌握工具定义的多种方式和最佳实践
    知识点：
    - 基础工具定义（@tool 装饰器）
    - 自定义工具名称
    - 自定义工具描述
    - 类型提示的重要性
    """
    print("\n===== 示例 1: 工具定义的最佳实践 =====")

    # 方式 1: 基础工具定义（推荐）
    # docstring 会被 AI 读取，用于理解工具功能
    @tool
    def search_news(keyword: str, limit: int = 5) -> str:
        """
        搜索最新新闻信息。
        这个函数用于检索与关键词相关的新闻。
        
        参数:
            keyword: 搜索关键词, 例如 "AI"、"科技"
            limit: 返回结果数量, 默认 5 条
        
        返回:
            新闻摘要列表
        """
        news_db = {
            "AI": ["AI 技术突破: GPT-5 发布", "AI 在医疗领域的应用", "AI 助手成为日常标配"],
            "科技": ["量子计算新进展", "5G 网络覆盖全球", "区块链技术成熟应用"],
        }
        results = news_db.get(keyword, ["暂无相关新闻"])
        return "\n".join(results[:limit])

    # 方式 2: 自定义工具名称
    # 当函数名不够清晰时，可以自定义工具名
    @tool("weather_query")  # 自定义名称
    # --------------------------------------------------
    # 代码解释: @tool("custom_name") 自定义工具名称
    #
    # @tool 装饰器接受一个字符串参数作为工具的自定义名称。
    # 默认情况下，工具名称就是函数名。当函数名不够语义化
    # 或需要与外部系统对齐时，可以显式指定名称。
    #
    # 示例:
    # @tool("weather_query")
    # def get_weather_info(city: str) -> str: ...
    #
    # 对 AI 可见的名称是 "weather_query", 而不是 "get_weather_info"。
    # 这在工具注册到 Agent 时非常重要。
    # --------------------------------------------------
    def get_weather_info(city: str) -> str:
        """
        查询城市天气。
        
        参数:
            city: 城市名称
        
        返回:
            天气描述
        """
        weather_db = {"北京": "晴朗, 25°C", "上海": "多云, 22°C", "广州": "阵雨, 28°C"}
        return weather_db.get(city, f"无法查询 {city} 的天气")

    # 方式 3: 自定义描述
    # 当 docstring 不够清晰时，可以覆盖描述
    @tool("calc", description="执行数学运算。适用于所有数学计算问题。")
    def calculate(expression: str) -> str:
        """计算数学表达式。"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算错误: {e}"

    # 初始化模型
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=[search_news, get_weather_info, calculate],
        system_prompt="你是一个多功能助手，可以查询新闻、天气和进行数学计算。",
    )

    # 测试 1: 新闻查询
    print("\n--- 测试 1: 新闻查询 ---")
    result1 = agent.invoke(
        {"messages": [("user", "最近有什么 AI 相关的新闻?")]}, config={"callbacks": [langfuse_handler]}
    )
    print(f"AI: {result1['messages'][-1].content}")

    # 测试 2: 天气查询
    print("\n--- 测试 2: 天气查询 ---")
    result2 = agent.invoke(
        {"messages": [("user", "北京天气怎么样?")]}, config={"callbacks": [langfuse_handler]}
    )
    print(f"AI: {result2['messages'][-1].content}")

    # 测试 3: 数学计算
    print("\n--- 测试 3: 数学计算 ---")
    result3 = agent.invoke(
        {"messages": [("user", "计算 (15 + 25) 乘以 3")]}, config={"callbacks": [langfuse_handler]}
    )
    print(f"AI: {result3['messages'][-1].content}")


def example_2():
    """
    示例 2: 高级参数 Schema (Pydantic)
    目标：使用 Pydantic 定义复杂的工具参数
    知识点：
    - Pydantic BaseModel 定义参数
    - Field 添加参数描述和默认值
    - Literal 类型限制输入选项
    """
    print("\n===== 示例 2: 高级参数 Schema (Pydantic) =====")

    # 定义输入 Schema
    class TravelQuery(BaseModel):
        """旅行查询的输入参数。"""
        origin: str = Field(description="出发城市")
        # --------------------------------------------------
        # 代码解释: Pydantic Field(description=...)
        #
        # Field(description=...) 为 Pydantic 模型的每个字段添加描述信息。
        # 这些描述会被 LangChain 提取并传递给 AI，帮助 AI 理解
        # 每个参数的含义，从而更准确地调用工具。
        #
        # 示例:
        # origin: str = Field(description="出发城市")
        #
        # AI 看到的是:
        # origin (string): 出发城市
        #
        # 没有 description 的字段，AI 只能看到字段名和类型，
        # 理解能力会大幅下降。始终为工具参数添加 description。
        # --------------------------------------------------
        destination: str = Field(description="目的地城市")
        date: str = Field(description="出行日期, 格式: YYYY-MM-DD")
        travel_class: Literal["经济舱", "商务舱", "头等舱"] = Field(default="经济舱", description="舱位等级")

    # 使用 args_schema 定义复杂参数
    @tool(args_schema=TravelQuery)
    # --------------------------------------------------
    # 代码解释: args_schema=TravelQuery 与 Pydantic
    #
    # args_schema 参数允许使用 Pydantic BaseModel 类来定义
    # 工具的输入参数结构。这样做的好处:
    # 1. 强类型验证: LangChain 会在调用前自动校验参数
    # 2. 完整的描述信息: Field(description=...) 会被传递给 AI
    # 3. IDE 智能提示: 开发时获得自动补全
    # 4. 默认值支持: Field(default=...) 设置参数默认值
    #
    # 使用方式:
    # class TravelQuery(BaseModel):
    #     origin: str = Field(description="出发城市")
    #     ...
    #
    # @tool(args_schema=TravelQuery)
    # def search_flights(origin: str, ...) -> str: ...
    #
    # 函数签名中的参数名必须与 BaseModel 中的字段名一致。
    # --------------------------------------------------
    def search_flights(origin: str, destination: str, date: str, travel_class: str = "经济舱") -> str:
        """
        查询航班信息。
        这个工具用于搜索两个城市之间的航班。
        """
        # 模拟航班数据
        flights_db = {
            ("北京", "上海"): [
                {"time": "08:00", "price": 1200, "airline": "国航"},
                {"time": "14:00", "price": 980, "airline": "东航"},
                {"time": "19:00", "price": 1100, "airline": "南航"},
            ],
            ("上海", "广州"): [
                {"time": "09:00", "price": 1500, "airline": "东航"},
                {"time": "15:00", "price": 1300, "airline": "南航"},
            ],
        }
        route = (origin, destination)
        flights = flights_db.get(route, [])

        if not flights:
            return f"暂无 {origin} 到 {destination} 的航班"

        # 根据舱位调整价格
        price_multiplier = {"经济舱": 1.0, "商务舱": 2.5, "头等舱": 4.0}
        multiplier = price_multiplier.get(travel_class, 1.0)

        results = []
        for f in flights:
            price = int(f["price"] * multiplier)
            results.append(f"{f['airline']} {f['time']} - ¥{price} ({travel_class})")

        return f"{origin} → {destination} ({date}):\n" + "\n".join(results)

    # 定义另一个复杂 Schema
    class HotelQuery(BaseModel):
        """酒店预订的输入参数。"""
        city: str = Field(description="城市名称")
        check_in: str = Field(description="入住日期, 格式: YYYY-MM-DD")
        check_out: str = Field(description="退房日期, 格式: YYYY-MM-DD")
        room_type: Literal["标准间", "大床房", "套房"] = Field(default="标准间", description="房间类型")

    @tool(args_schema=HotelQuery)
    def search_hotels(city: str, check_in: str, check_out: str, room_type: str = "标准间") -> str:
        """
        查询酒店信息。
        这个工具用于搜索指定城市的酒店。
        """
        hotels_db = {
            "北京": [
                {"name": "北京饭店", "price": 800, "rating": 4.8},
                {"name": "王府井酒店", "price": 600, "rating": 4.5},
                {"name": "国贸大酒店", "price": 1200, "rating": 4.9},
            ],
            "上海": [
                {"name": "和平饭店", "price": 1500, "rating": 4.9},
                {"name": "浦东香格里拉", "price": 1300, "rating": 4.8},
            ],
        }
        hotels = hotels_db.get(city, [])

        if not hotels:
            return f"暂无 {city} 的酒店信息"

        # 根据房型调整价格
        price_multiplier = {"标准间": 1.0, "大床房": 1.2, "套房": 2.5}
        multiplier = price_multiplier.get(room_type, 1.0)

        results = []
        for h in hotels:
            price = int(h["price"] * multiplier)
            results.append(f"{h['name']} - ¥{price}/晚 (评分: {h['rating']})")

        return f"{city} 酒店 ({room_type}):\n" + "\n".join(results)

    # 初始化模型
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    # 创建旅行助手 Agent
    agent = create_agent(
        model=model,
        tools=[search_flights, search_hotels],
        system_prompt="""你是一个专业的旅行规划助手，可以帮助用户:
- 查询航班信息
- 搜索酒店
请根据用户的需求, 提供详细的出行建议。""",
    )

    # 测试 1: 查询航班
    print("\n--- 测试 1: 查询航班 ---")
    result1 = agent.invoke(
        {"messages": [("user", "帮我查一下北京到上海的航班, 明天出发, 商务舱")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    # 测试 2: 查询酒店
    print("\n--- 测试 2: 查询酒店 ---")
    result2 = agent.invoke(
        {"messages": [("user", "上海有什么好酒店推荐? 大床房, 明天开始住两晚")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_3():
    """
    示例 3: 运行时上下文访问 (ToolRuntime)
    目标：掌握工具如何访问运行时信息
    知识点：
    - ToolRuntime 的使用
    - 访问对话历史 (state)
    - 访问用户上下文 (context)
    - 流式写入 (stream_writer)
    """
    print("\n===== 示例 3: 运行时上下文访问 =====")

    # 过滤 Pydantic 序列化警告 (LangGraph checkpointing 的已知行为)
    import warnings
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    # 定义用户上下文
    class UserContext(BaseModel):
        """用户上下文信息。"""
        user_id: str
        user_name: str

    # 工具 1: 访问对话历史
    @tool
    def get_conversation_history(runtime: ToolRuntime) -> str:
        """
        获取当前对话历史。
        这个工具可以查看之前的对话内容。
        runtime 参数是自动注入的, 不会暴露给 AI。
        """
        messages = runtime.state["messages"]
        # --------------------------------------------------
        # 代码解释: runtime.state["messages"]
        #
        # runtime.state 是 AgentState 类型（本质是一个字典），
        # 包含 Agent 的完整运行状态。
        #
        # state["messages"] 存储了完整的对话历史，
        # 包括用户消息、AI 回复、工具调用结果等。
        #
        # 每个消息对象具有:
        # msg.type → 消息类型 ("human", "ai", "tool" 等)
        # msg.content → 消息内容
        #
        # 示例用法:
        # messages = runtime.state["messages"]
        # for msg in messages:
        #     if msg.type == "human":
        #         print(msg.content)
        #
        # 注意: state 中还可能包含 checkpoints、metadata 等其他字段，
        # 但 messages 是最常用的。
        # --------------------------------------------------

        # 提取最近的 3 条用户消息
        user_messages = []
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_messages.append(msg.content)
                if len(user_messages) >= 3:
                    break

        if user_messages:
            return "最近的用户消息:\n" + "\n".join(f"- {msg}" for msg in reversed(user_messages))
        return "暂无历史消息"

    # 工具 2: 访问用户上下文
    @tool
    def get_user_profile(runtime: ToolRuntime[UserContext]) -> str:
        """
        获取当前模型的个人信息。
        通过 runtime.context 访问用户上下文。
        """
        # --------------------------------------------------
        # 代码解释: ToolRuntime[UserContext] 泛型类型
        #
        # ToolRuntime 支持泛型参数，用于指定上下文的类型。
        # ToolRuntime[UserContext] 表示:
        # - runtime.context 的类型是 UserContext
        # - 获得完整的 IDE 类型检查和自动补全
        #
        # 使用方式:
        # class UserContext(BaseModel):
        #     user_id: str
        #     user_name: str
        #
        # @tool
        # def my_tool(runtime: ToolRuntime[UserContext]) -> str:
        #     user_id = runtime.context.user_id  # IDE 有提示
        #     name = runtime.context.user_name  # 类型安全
        #
        # 不指定泛型参数时（ToolRuntime 不加 []），
        # runtime.context 的类型是 Any，没有类型检查。
        # 建议始终使用 ToolRuntime[YourContextClass] 获得类型安全。
        # --------------------------------------------------
        user_id = runtime.context.user_id
        user_name = runtime.context.user_name

        # --------------------------------------------------
        # 代码解释: runtime.context
        #
        # runtime.context 用于访问通过 agent.invoke() 的 context= 参数
        # 传入的用户自定义上下文数据。
        #
        # 数据流向:
        # 1. 定义上下文类: class UserContext(BaseModel): ...
        # 2. 创建 Agent 时声明: context_schema=UserContext
        # 3. 调用时传入数据: agent.invoke(..., context=UserContext(...))
        # 4. 工具中访问: runtime.context.user_id
        #
        # 注意:
        # - 如果调用时没有传入 context=，runtime.context 为 None
        # - 如果创建 Agent 时没有声明 context_schema，runtime.context 也为 None
        # - 建议始终声明 context_schema，获得类型安全和 IDE 提示
        #
        # 与 middleware 的关系:
        # runtime.context 不仅在工具中可以访问，在中间件（middleware）
        # 中同样可以使用，实现全局上下文共享。
        # --------------------------------------------------

        # 模拟用户数据库
        user_db = {
            "001": {"name": "张三", "level": "VIP", "points": 5000},
            "002": {"name": "李四", "level": "普通会员", "points": 1200},
            "003": {"name": "王五", "level": "黄金会员", "points": 8000},
        }

        user = user_db.get(user_id, {})
        if user:
            return (
                f"用户信息:\n"
                f" - 姓名: {user.get('name', user_name)}\n"
                f" - 等级: {user.get('level', '未知')}\n"
                f" - 积分: {user.get('points', 0)}"
            )
        return f"未找到用户 {user_id} 的信息"

    # 工具 3: 流式写入进度
    @tool
    def generate_report(topic: str, runtime: ToolRuntime) -> str:
        """
        生成主题报告。
        使用 stream_writer 实时显示进度。
        """
        writer = runtime.stream_writer
        # --------------------------------------------------
        # 代码解释: runtime.stream_writer
        #
        # runtime.stream_writer 是一个可调用对象（Callable[[str], None]），
        # 用于在工具执行过程中实时输出进度信息。
        #
        # 使用场景:
        # - 长时间运行的任务（如报告生成、数据分析）
        # - 需要向用户或开发者展示执行进度
        # - 流式响应场景中提供中间状态反馈
        #
        # 示例:
        # writer = runtime.stream_writer
        # writer("正在收集资料...")
        # writer("正在分析数据...")
        # writer("报告生成完成!")
        #
        # 输出会在 agent.stream() 的 "updates" 模式中捕获，
        # 配合 stream_mode="updates" 使用效果最佳。
        # --------------------------------------------------

        # 模拟报告生成过程
        writer(f"开始生成 '{topic}' 报告...")
        writer("正在收集资料...")
        writer("正在分析数据...")
        writer("正在撰写报告...")
        writer("报告生成完成!")

        return f"关于 '{topic}' 的报告已生成。"

    # 初始化模型
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=[get_conversation_history, get_user_profile, generate_report],
        system_prompt="""你是一个智能助手，可以:
- 查看对话历史
- 获取用户信息
- 生成报告
根据用户需求选择合适的工具。""",
        context_schema=UserContext,
    )
    # --------------------------------------------------
    # 代码解释: context_schema=UserContext 在 create_agent 中
    #
    # context_schema 在创建 Agent 时声明上下文的类型。
    #
    # 作用:
    # 1. 告诉 LangChain 后续会通过 context= 传入什么类型的数据
    # 2. 使 runtime.context 获得确定的类型（而非 None）
    # 3. 启用 IDE 类型检查和自动补全
    #
    # 配对使用流程:
    # context_schema=UserContext ← 创建时声明类型（这里）
    # ↓
    # agent.invoke(..., context=UserContext(...)) ← 调用时传入数据
    # ↓
    # runtime.context.user_id ← 工具/中间件中访问
    #
    # 注意:
    # - context_schema 是可选的，不声明时 runtime.context 为 None
    # - 建议始终声明 context_schema，获得类型安全
    # - 类型可以是 dataclass、BaseModel 或任意 Python 类
    # --------------------------------------------------

    # 测试 1: 获取用户信息
    print("\n--- 测试 1: 获取用户信息 ---")
    result1 = agent.invoke(
        {"messages": [("user", "我的个人信息是什么?")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_id="001", user_name="张三"),
    )
    # --------------------------------------------------
    # 代码解释: context=UserContext(...) 在 agent.invoke() 中
    #
    # context 参数在调用 agent.invoke() 时传入具体的上下文数据。
    #
    # 与 context_schema 的关系:
    # context_schema=UserContext ← 声明类型（create_agent 时）
    # context=UserContext(...)   ← 传入数据（invoke 时）
    #
    # 每次 invoke 都可以传入不同的 context，实现:
    # - 多用户隔离（不同用户有不同的 user_id）
    # - 动态上下文（同一次会话中可以切换上下文）
    # - 租户隔离（SaaS 场景中区分不同租户）
    #
    # 示例:
    # agent.invoke(
    #   {"messages": [("user", "我的信息是什么?")]},
    #   context=UserContext(user_id="001", user_name="张三"),
    # )
    #
    # context 是可选的，不传入时 runtime.context 为 None。
    # --------------------------------------------------
    print(f"AI: {result1['messages'][-1].content}")

    # 测试 2: 生成报告
    print("\n--- 测试 2: 生成报告 ---")
    for chunk in agent.stream(
        {"messages": [("user", "帮我生成一份关于 AI 发展的报告")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_id="001", user_name="张三"),
        stream_mode="updates",
    ):
        # --------------------------------------------------
        # 代码解释: stream_mode="updates"
        #
        # stream_mode 控制 agent.stream() 返回的数据格式。
        #
        # "updates" 模式:
        # - 只返回每个节点的增量更新（而不是完整状态）
        # - 返回格式: {"节点名": {"字段": 值}}
        # - 适合: 实时监控工具执行进度、中间结果
        #
        # 其他可选模式:
        # "values" → 返回完整的 AgentState（数据量大）
        # "messages" → 只返回消息更新
        # "custom" → 自定义流式输出
        #
        # 与 stream_writer 的配合:
        # 工具中使用 runtime.stream_writer("进度信息")
        # 输出会在 stream_mode="updates" 的 chunk 中捕获
        #
        # 示例:
        # for chunk in agent.stream(..., stream_mode="updates"):
        #     print(chunk)
        #     # 输出如: {"generate_report": {"messages": [...]}}
        # --------------------------------------------------
        print(f"AI: {chunk}")


def example_4():
    """
    示例 4: 工具错误处理
    目标：掌握工具错误处理的最佳实践
    知识点：
    - 工具内部异常处理
    - 返回错误信息给 AI
    - 使用 ToolMessage 处理错误
    """
    print("\n===== 示例 4: 工具错误处理 =====")

    # 工具 1: 带错误处理的 API 调用
    @tool
    def call_api(endpoint: str, params: str = "") -> str:
        """
        调用外部 API 获取数据。
        
        参数:
            endpoint: API 端点名称
            params: 查询参数
        
        返回:
            API 响应数据
        """
        # 模拟 API 调用
        api_db = {
            "users": '{"id": 1, "name": "张三", "email": "zhangsan@example.com"}',
            "products": '{"id": 101, "name": "iPhone", "price": 5999}',
        }

        try:
            if endpoint not in api_db:
                raise ValueError(f"未知的 API 端点: {endpoint}")

            # 模拟网络错误
            if endpoint == "error":
                raise ConnectionError("网络连接失败")

            return api_db[endpoint]
        except ValueError as e:
            # 参数错误 - 返回清晰的错误信息
            return f"参数错误: {e}"
        except ConnectionError as e:
            # 网络错误 - 返回可恢复的错误
            return f"网络错误: {e}, 请稍后重试"
        except Exception as e:
            # 其他错误
            return f"未知错误: {type(e).__name__} - {e}"

    # 工具 2: 带重试逻辑的工具
    @tool
    def fetch_data_with_retry(url: str, max_retries: int = 3) -> str:
        """
        从 URL 获取数据, 支持自动重试。
        
        参数:
            url: 目标 URL
            max_retries: 最大重试次数, 默认 3 次
        
        返回:
            获取的数据
        """
        import random

        # 模拟不稳定的网络连接
        for attempt in range(1, max_retries + 1):
            try:
                # 模拟 50% 的失败率
                if random.random() < 0.5 and attempt < max_retries:
                    raise ConnectionError(f"第 {attempt} 次尝试失败")

                # 模拟成功响应
                return f"成功获取数据（尝试次数: {attempt}）\n {{\n'status': 'ok', 'data': '示例数据'}}"
            except ConnectionError as e:
                if attempt == max_retries:
                    return f"错误: 达到最大重试次数（{max_retries}），最后错误: {e}"
                # 继续重试
                continue

        return "错误: 未知失败"

    # 初始化模型
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=[call_api, fetch_data_with_retry],
        system_prompt="""你是一个技术助手，可以调用 API 和获取数据。
如果遇到错误, 请向用户解释并建议解决方案。""",
    )

    # 测试 1: 正常调用
    print("\n--- 测试 1: 正常 API 调用 ---")
    result1 = agent.invoke({"messages": [("user", "帮我调用 users API")]}, config={"callbacks": [langfuse_handler]})
    print(f"AI: {result1['messages'][-1].content}")

    # 测试 2: 错误处理
    print("\n--- 测试 2: 错误处理 ---")
    result2 = agent.invoke({"messages": [("user", "帮我调用 error API")]}, config={"callbacks": [langfuse_handler]})
    print(f"AI: {result2['messages'][-1].content}")


def example_5():
    """
    示例 5: 综合实战 - 电商客服助手
    目标：综合运用所有工具知识，创建生产级客服 Agent
    知识点：
    - 复杂工具定义 (Pydantic Schema)
    - 运行时上下文访问
    - 错误处理
    - 多工具协作
    """
    print("\n===== 示例 5: 综合实战 - 电商客服助手 =====")

    # 定义用户上下文
    @dataclass
    class CustomerContext:
        """客户上下文信息。"""
        customer_id: str
        order_id: str

    # 定义查询 Schema
    class OrderQuery(BaseModel):
        """订单查询参数。"""
        order_id: str = Field(description="订单编号")

    class ProductQuery(BaseModel):
        """商品查询参数。"""
        product_name: str = Field(description="商品名称或关键词")
        max_price: float = Field(default=9999, description="最高价格")

    # 工具 1: 订单查询
    @tool(args_schema=OrderQuery)
    def query_order(order_id: str, runtime: ToolRuntime[CustomerContext]) -> str:
        """
        查询订单状态和物流信息。
        
        参数:
            order_id: 订单编号
        
        返回:
            订单详细信息
        """
        # 模拟订单数据库
        orders_db = {
            "ORD001": {
                "status": "已发货",
                "product": "iPhone 15 Pro",
                "price": 7999,
                "logistics": "顺丰快递 SF1234567890",
                "estimated_delivery": "2024-01-15",
            },
            "ORD002": {
                "status": "处理中",
                "product": "MacBook Air M3",
                "price": 9999,
                "logistics": "待发货",
                "estimated_delivery": "2024-01-18",
            },
        }

        order = orders_db.get(order_id)
        if not order:
            return f"未找到订单 {order_id}"

        return (
            f"订单 {order_id} 详情:\n"
            f" - 商品: {order['product']}\n"
            f" - 价格: ¥{order['price']}\n"
            f" - 状态: {order['status']}\n"
            f" - 物流: {order['logistics']}\n"
            f" - 预计送达: {order['estimated_delivery']}"
        )

    # 工具 2: 商品搜索
    @tool(args_schema=ProductQuery)
    def search_products(product_name: str, max_price: float = 9999) -> str:
        """
        搜索商品。
        
        参数:
            product_name: 商品名称或关键词
            max_price: 最高价格限制
        
        返回:
            商品列表
        """
        products_db = [
            {"name": "iPhone 15 Pro", "price": 7999, "stock": 100, "rating": 4.8},
            {"name": "iPhone 15", "price": 5999, "stock": 200, "rating": 4.7},
            {"name": "MacBook Air M3", "price": 9999, "stock": 50, "rating": 4.9},
            {"name": "AirPods Pro", "price": 1999, "stock": 500, "rating": 4.6},
            {"name": "iPad Air", "price": 4999, "stock": 150, "rating": 4.7},
        ]

        # 过滤商品
        results = [p for p in products_db if product_name.lower() in p["name"].lower() and p["price"] <= max_price]

        if not results:
            return f"未找到符合条件的商品"

        output = "搜索结果:\n"
        for p in results:
            output += f"- {p['name']}: ¥{p['price']}（库存: {p['stock']}），评分: {p['rating']}\n"

        return output

    # 工具 3: 售后服务
    @tool
    def process_return(order_id: str, reason: str, runtime: ToolRuntime[CustomerContext]) -> str:
        """
        处理退货申请。
        
        参数:
            order_id: 订单编号
            reason: 退货原因
        
        返回:
            退货处理结果
        """
        # 模拟退货政策检查
        return_policy = {
            "ORD001": {"eligible": True, "deadline": "2024-01-20"},
            "ORD002": {"eligible": False, "reason": "订单尚未发货"},
        }

        policy = return_policy.get(order_id)
        if not policy:
            return f"未找到订单 {order_id}"

        if policy["eligible"]:
            return (
                f"退货申请已受理! \n"
                f" - 订单: {order_id}\n"
                f" - 原因: {reason}\n"
                f" - 退货截止日期: {policy['deadline']}\n"
                f" - 退款将在收到商品后 3-5 个工作日内处理"
            )
        else:
            return f"抱歉, 该订单暂不符合退货条件。\n" f" - 原因: {policy['reason']}\n" f"如有问题, 请联系客服"

    # 初始化模型
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3, max_tokens=600)

    # 创建电商客服 Agent
    agent = create_agent(
        model=model,
        tools=[query_order, search_products, process_return],
        system_prompt="""你是一个专业的电商客服助手，可以帮助用户:
- 查询订单状态和物流信息
- 搜索商品并提供购买建议
- 处理退货申请
请礼貌、专业地回答用户问题, 并提供详细的解决方案。""",
        context_schema=CustomerContext,
    )

    # 测试场景 1: 查询订单
    print("\n--- 测试 1: 查询订单 ---")
    result1 = agent.invoke(
        {"messages": [("user", "帮我查一下订单 ORD001 的物流")]},
        config={"callbacks": [langfuse_handler]},
        context=CustomerContext(customer_id="C001", order_id="ORD001"),
    )
    print(f"用户: 帮我查一下订单 ORD001 的物流")
    print(f"AI: {result1['messages'][-1].content}")

    # 测试场景 2: 搜索商品
    print("\n--- 测试 2: 搜索商品 ---")
    result2 = agent.invoke(
        {"messages": [("user", "我想买 iPhone, 有什么推荐?")]},
        config={"callbacks": [langfuse_handler]},
        context=CustomerContext(customer_id="C001", order_id="ORD001"),
    )
    print(f"用户: 我想买 iPhone, 有什么推荐?")
    print(f"AI: {result2['messages'][-1].content}")

    # 测试场景 3: 退货处理
    print("\n--- 测试 3: 退货处理 ---")
    result3 = agent.invoke(
        {"messages": [("user", "我要退货, 订单 ORD001, 因为不想要了")]},
        config={"callbacks": [langfuse_handler]},
        context=CustomerContext(customer_id="C001", order_id="ORD001"),
    )
    print(f"用户: 我要退货, 订单 ORD001, 因为不想要了")
    print(f"AI: {result3['messages'][-1].content}")


def main(example_number: int):
    """
    运行指定的示例
    
    参数:
        example_number: 示例编号 (1-5)
    """
    print("=" * 60)
    print("第三课: 工具定义与高级用法")
    print("=" * 60)

    examples = {1: example_1, 2: example_2, 3: example_3, 4: example_4, 5: example_5}

    if example_number in examples:
        examples[example_number]()
    else:
        print(f"错误: 示例编号 {example_number} 不存在, 请选择 1-5")


if __name__ == "__main__":
    # 修改这里的数字来运行不同的示例 (1-5)
    main(4)