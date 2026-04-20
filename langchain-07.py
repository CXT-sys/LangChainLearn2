# 提示词工程与中间件
# =====================================
# 第四课：核心组件 - 提示词工程
# 模块：1.4
# 目标：掌握提示词设计和中间件系统
# 知识点：
# - 静态提示词模板
# - 动态提示词（@dynamic_prompt）
# - AgentState 和 Runtime（中间件的基础参数）
# - 中间件 7 种 hook
# - wrap_model_call 和 wrap_tool_call

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import (
    before_agent,
    before_model,
    after_model,
    after_agent,
    wrap_model_call,
    wrap_tool_call,
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
    AgentState,
)
from langgraph.runtime import Runtime
from typing import Any, Callable
from dotenv import load_dotenv

# 加载环境变量，初始化Langfuse回调
load_dotenv()
langfuse_handler = CallbackHandler()

# ==================================================
# 示例1：静态提示词设计最佳实践
# ==================================================
def example_1():
    """
    示例 1: 静态提示词设计最佳实践
    目标：掌握高质量的静态提示词设计
    知识点：
    - 角色定义
    - 行为准则
    - 工具使用指南
    """
    print("\n==== 示例 1: 静态提示词设计最佳实践 ====")

    # 静态系统提示词（角色+职责+约束）
    SYSTEM_PROMPT = """你是一个专业的编程助手，专注于回答编程相关问题。

你的职责：
- 提供准确、实用的代码示例
- 解释代码的工作原理
- 指出最佳实践和潜在问题

回答要求：
- 代码必须包含详细注释
- 优先使用 Python 3.10+ 语法

如果用户的问题超出编程范围，请礼貌地引导回编程话题。"""

    # 定义工具：搜索代码示例
    @tool
    def search_code_example(language: str, topic: str) -> str:
        """
        搜索代码示例。
        参数:
            language: 编程语言（如 "python", "javascript"）
            topic: 主题关键词
        返回:
            相关代码示例
        """
        examples = {
            ("python", "排序"): "sorted([3, 1, 2]) # 返回 [1, 2, 3]",
            ("python", "文件"): "with open('file.txt') as f: content = f.read()",
        }
        return examples.get((language, topic), f"暂无 {language} 的 {topic} 示例")

    # 初始化模型与Agent
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)
    agent = create_agent(
        model=model,
        tools=[search_code_example],
        system_prompt=SYSTEM_PROMPT,
    )

    # 测试静态提示词效果
    print("\n--- 测试: 编程问题 ---")
    result = agent.invoke(
        {"messages": [("user", "Python 中如何对列表排序? ")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result['messages'][-1].content}")

# ==================================================
# 示例2：动态提示词（@dynamic_prompt）
# ==================================================
def example_2():
    """
    示例 2: 动态提示词 - 根据上下文生成提示
    目标: 使用 @dynamic_prompt 装饰器创建动态提示词
    知识点:
    - @dynamic_prompt 装饰器
    - 根据运行时上下文生成提示
    """
    print("\n==== 示例 2: 动态提示词 ====")
    from dataclasses import dataclass

    # 定义用户上下文（传递给Agent的运行时参数）
    @dataclass
    class UserContext:
        """用户上下文。"""
        user_level: str # "beginner", "intermediate", "expert"

    # ------------------------------
    # @dynamic_prompt 装饰器说明
    # - 将普通函数转换为动态提示词中间件
    # - 每次模型调用前执行，返回字符串作为系统提示词
    # - 可通过 request.state/request.runtime 访问状态与上下文
    # ------------------------------
    @dynamic_prompt
    def generate_adaptive_prompt(request: ModelRequest) -> str:
        """根据用户水平生成适应性提示词。"""
        user_level = "beginner"
        # 从Runtime上下文获取用户等级
        if request.runtime and request.runtime.context:
            user_level = request.runtime.context.user_level

        base_prompt = "你是一个 Python 编程教师。"

        # 根据用户等级动态生成提示词
        if user_level == "beginner":
            return f"""{base_prompt}
用户水平: 初学者
- 使用简单易懂的语言, 避免专业术语
- 提供详细的逐步解释和类比"""
        elif user_level == "intermediate":
            return f"""{base_prompt}
用户水平: 中级
- 可以使用专业术语
- 关注最佳实践和设计模式"""
        else:
            return f"""{base_prompt}
用户水平: 专家
- 深入讨论底层原理和架构设计"""

    # 定义工具：解释编程概念
    @tool
    def explain_concept(concept: str) -> str:
        """解释编程概念。"""
        explanations = {
            "装饰器": "装饰器是 Python 中修改函数行为的高级功能。",
            "闭包": "闭包是函数和其引用环境的组合。",
        }
        return explanations.get(concept, f"暂无 {concept} 的解释")

    # 初始化模型与Agent（注册动态提示词中间件）
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.5)
    agent = create_agent(
        model=model,
        tools=[explain_concept],
        middleware=[generate_adaptive_prompt],
    )

    # 测试不同用户等级的动态提示词效果
    print("\n--- 测试 1: 初学者模式 ---")
    result1 = agent.invoke(
        {"messages": [("user", "什么是装饰器? ")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_level="beginner"),
    )
    print(f"AI: {result1['messages'][-1].content[:200]}...")

    print("\n--- 测试 2: 专家模式 ---")
    result2 = agent.invoke(
        {"messages": [("user", "什么是装饰器? ")]},
        config={"callbacks": [langfuse_handler]},
        context=UserContext(user_level="expert"),
    )
    print(f"AI: {result2['messages'][-1].content[:200]}...")

# ==================================================
# 示例3：AgentState 与 Runtime 详解
# ==================================================
def example_3():
    """
    示例 3: AgentState 和 Runtime 详解
    """
    print("\n==== 示例 3: AgentState 和 Runtime 详解 ====")

    # ------------------------------
    # 一、AgentState：Agent的“内存”
    # - 是一个TypedDict，在Agent整个生命周期中存在
    # - 内置字段：messages（必需）、jump_to（可选）、structured_response（可选）
    # - 支持扩展自定义字段
    # ------------------------------
    # 源码定义（来自LangChain）：
    # class AgentState(TypedDict, Generic[ResponseT]):
    #     messages: Required[Annotated[list[AnyMessage], add_messages]]
    #     jump_to: NotRequired[Annotated[JumpTo | None, ...]]
    #     structured_response: NotRequired[Annotated[ResponseT, ...]]

    # 扩展AgentState，添加自定义字段
    from typing_extensions import NotRequired
    from dataclasses import dataclass

    class DemoState(AgentState):
        call_count: NotRequired[int] # 记录模型调用次数（可选字段）

    # 访问方式：
    # state["messages"]          # 读取内置字段
    # state.get("call_count", 0) # 安全访问自定义字段

    # ------------------------------
    # 二、Runtime：Agent执行时的“运行时环境”
    # - 每次agent.invoke()/agent.stream()调用时创建
    # - 核心属性：runtime.context（用户自定义上下文）、runtime.model（当前模型实例）
    # ------------------------------

    # ------------------------------
    # 三、AgentState / Runtime / ToolRuntime 对比
    # | 对比项       | AgentState               | Runtime                | ToolRuntime             |
    # |--------------|--------------------------|------------------------|-------------------------|
    # | 含义         | Agent的内存状态（TypedDict） | Agent执行时的运行环境（对象） | 工具执行时的运行环境（对象） |
    # | 生命周期     | 整个Agent执行期间        | 单次invoke/stream调用期间 | 单次工具调用期间        |
    # | 内置属性     | messages、jump_to、structured_response | context、model | context、state、stream_writer |
    # | 使用场景     | 中间件读写Agent状态      | 中间件访问上下文       | 工具内访问Agent状态     |
    # ------------------------------

    # 定义用户上下文类（传递给Runtime）
    @dataclass
    class DemoContext:
        user_id: str
        verbose: bool = True

    # ------------------------------
    # 中间件Hook示例：@before_model / @after_model / @before_agent / @after_agent
    # ------------------------------
    # @before_model：模型调用前执行，可读写AgentState，返回值可修改状态或终止Agent
    @before_model(state_schema=DemoState)
    def count_calls(state: DemoState, runtime: Runtime) -> dict[str, Any] | None:
        """统计模型调用次数。"""
        msg_count = len(state["messages"])
        current = state.get("call_count", 0)
        user_id = runtime.context.user_id if runtime.context else "unknown"

        if runtime.context and runtime.context.verbose:
            print(f"[count_calls] 用户: {user_id}, 消息数: {msg_count}, 调用次数: {current}")
        # 返回字典会合并到AgentState中
        return {"call_count": current + 1}

    # @after_model：模型调用后执行，用于日志、统计等后处理，按声明反序执行
    @after_model(state_schema=DemoState)
    def show_result(state: DemoState, runtime: Runtime) -> dict[str, Any] | None:
        """显示模型返回结果。"""
        last_msg = state["messages"][-1]
        content = last_msg.content if hasattr(last_msg, "content") else ""
        total = state.get("call_count", 0)
        print(f"[show_result] 回复长度: {len(content)} 字符, 累计调用: {total} 次")
        return None

    # @before_agent：Agent执行前执行，整个生命周期仅一次
    @before_agent(state_schema=DemoState)
    def init_state(state: DemoState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent启动时初始化。"""
        print("[before_agent] Agent 开始执行")
        return None

    # @after_agent：Agent执行后执行，整个生命周期仅一次
    @after_agent(state_schema=DemoState)
    def cleanup_state(state: DemoState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent完成时清理。"""
        total = state.get("call_count", 0)
        print(f"[after_agent] Agent 执行完成, 总调用次数: {total}")
        return None

    # 定义演示工具
    @tool
    def demo_tool(query: str) -> str:
        """演示工具。"""
        return f"演示结果: {query}"

    # 初始化模型与Agent（注册所有中间件）
    model = init_chat_model(
        "doubao-seed-2.0-lite",
        model_provider="openai",
        temperature=0.5,
    )
    agent = create_agent(
        model=model,
        tools=[demo_tool],
        system_prompt="你是一个演示助手。",
        middleware=[init_state, count_calls, show_result, cleanup_state],
        context_schema=DemoContext,
    )

    # 测试AgentState与Runtime
    print("\n--- 测试: AgentState 和 Runtime 演示 ---")
    result = agent.invoke(
        {"messages": [("user", "你好")]},
        config={"callbacks": [langfuse_handler]},
        context=DemoContext(user_id="user_001", verbose=True),
    )

    print(f"\n最终 call_count: {result.get('call_count', 0)}")
    print(f"AI: {result['messages'][-1].content}")

# ==================================================
# 示例4：中间件 - 7种Hook完整生命周期
# ==================================================
def example_4():
    """
    示例 4: 中间件 - 7 种 hook 完整生命周期
    目标: 掌握 LangChain 中间件的所有 hook 类型
    知识点:
    - 7 种装饰器的执行时机和返回值
    - wrap_model_call 的责任链模式
    - wrap_tool_call 的工具包装
    """
    print("\n==== 示例 4: 中间件 - 7 种 hook 完整生命周期 ====")

    # ------------------------------
    # LangChain 中间件完整生命周期
    # Agent 执行流程：
    # before_agent (整个生命周期只执行一次)
    # → ReAct 循环（可能多次）:
    #   wrap_model_call (包装整个模型调用，洋葱模型)
    #   → before_model (每次模型调用前)
    #   → 模型调用
    #   → after_model (每次模型调用后)
    #   → 如果需要工具: wrap_tool_call → 工具执行
    # → after_agent (整个生命周期只执行一次)
    # ------------------------------

    # 7 种装饰器：
    # 1. @before_agent → Agent 执行前（只一次）
    # 2. @before_model → 每次模型调用前
    # 3. @after_model → 每次模型调用后
    # 4. @after_agent → Agent 执行后（只一次）
    # 5. @wrap_model_call → 包装整个模型调用链
    # 6. @wrap_tool_call → 包装整个工具调用链
    # 7. @dynamic_prompt → 动态生成提示词

    # 7 种装饰器对比：
    # | 装饰器          | 执行时机               | 函数签名                  | 返回值       |
    # |-----------------|------------------------|---------------------------|--------------|
    # | @before_agent   | Agent 执行前(只一次)   | def mw(state, runtime)     | dict/None    |
    # | @before_model   | 每次模型调用前         | def mw(state, runtime)     | dict/None    |
    # | @after_model    | 每次模型调用后         | def mw(state, runtime)     | dict/None    |
    # | @after_agent    | Agent 执行后(只一次)   | def mw(state, runtime)     | dict/None    |
    # | @wrap_model_call| 包装整个模型调用链     | def mw(request, handler)   | ModelResponse|
    # | @wrap_tool_call | 包装整个工具调用链     | def mw(request, handler)   | Any          |
    # | @dynamic_prompt | 动态生成提示词         | def mw(request)            | str          |

    # 特殊参数：
    # - can_jump_to=["end"] → @before_agent 和 @before_model 可声明跳转目标
    # - state_schema=CustomState → 指定自定义状态类型
    # - tools=[] → 注册额外工具

    # 多中间件执行顺序 (middleware=[a, b, c])：
    # a.before → b.before → c.before → 模型 → c.after → b.after → a.after

    # 1. @before_agent - Agent启动初始化
    @before_agent
    def init_logging(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent启动前的初始化。"""
        print(f"[before_agent] Agent 开始, 初始消息数: {len(state['messages'])}")
        return None

    # 2. @before_model - 每次模型调用前
    @before_model
    def log_before_call(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """每次模型调用前记录日志。"""
        print(f"[before_model] 即将调用模型, 消息数: {len(state['messages'])}")
        return None

    # 3. @after_model - 每次模型调用后
    @after_model
    def log_after_call(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """每次模型调用后记录日志。"""
        last_message = state["messages"][-1]
        content_len = len(last_message.content) if hasattr(last_message, "content") else 0
        print(f"[after_model] 模型返回, 内容长度: {content_len}")
        return None

    # 4. @after_agent - Agent执行完成后
    @after_agent
    def final_cleanup(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent完成后的清理工作。"""
        print(f"[after_agent] Agent 执行完成, 总消息数: {len(state['messages'])}")
        return None

    # 工具：编程技巧工具
    @tool
    def get_tip(topic: str) -> str:
        """获取编程小技巧。"""
        tips = {
            "python": "Python 中使用 walrus operator (:=) 可以简化代码。",
            "git": "使用 git rebase -i 可以整理提交历史。",
        }
        return tips.get(topic.lower(), f"暂无 {topic} 的技巧")

    # 初始化模型与Agent
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.5)
    agent = create_agent(
        model=model,
        tools=[get_tip],
        system_prompt="你是一个编程技巧分享助手。",
        middleware=[init_logging, log_before_call, log_after_call, final_cleanup],
    )

    # 测试
    print("\n--- 测试: 中间件完整生命周期 ---")
    result = agent.invoke(
        {"messages": [("user", "分享一个 Python 小技巧")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"\nAI: {result['messages'][-1].content}")

# ==================================================
# 示例5：wrap_model_call - 模型调用包装
# ==================================================
def example_5():
    """
    示例 5: wrap_model_call - 模型调用包装
    目标: 掌握 wrap_model_call 中间件
    知识点:
    - 拦截模型调用
    - 实现重试逻辑
    - 动态模型选择
    """
    print("\n==== 示例 5: wrap_model_call - 模型调用包装 ====")

    # ------------------------------
    # @wrap_model_call 装饰器说明
    # - 完全控制模型调用过程的中间件
    # - 接收 ModelRequest 和 handler 可调用对象
    # - 必须调用 handler(request)，否则模型不会执行
    # ------------------------------
    # handler 的作用（责任链模式）：
    # - 单个 wrap_model_call 时，handler 指向实际模型调用
    # - 多个时，handler 指向下一个中间件，形成链式调用
    # - A(handler(B(handler(C(handler(request))))))
    # ------------------------------
    # request.override(model=...) 创建新请求副本，替换指定字段
    # ------------------------------

    # 1. 重试中间件
    @wrap_model_call
    def retry_on_failure(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """模型调用失败时自动重试。"""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[重试中间件] 第 {attempt} 次尝试")
                return handler(request)
            except Exception as e:
                if attempt == max_retries:
                    print(f"[重试中间件] 达到最大重试次数: {e}")
                    raise
                print(f"[重试中间件] 失败, 准备重试: {e}")

    # 初始化两个模型
    simple_model = init_chat_model(
        "doubao-seed-2.0-lite",
        model_provider="openai",
        temperature=0.3,
    )
    advanced_model = init_chat_model(
        "glm4.7",
        model_provider="openai",
        temperature=0.7,
    )

    # 2. 动态模型选择中间件
    @wrap_model_call
    def dynamic_model_selector(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """根据对话长度选择模型。"""
        if len(request.state["messages"]) > 10:
            print("[动态模型] 使用高级模型 (长对话)")
            selected_model = advanced_model
        else:
            print("[动态模型] 使用简单模型 (短对话)")
            selected_model = simple_model

        return handler(request.override(model=selected_model))

    # 工具：评估问题复杂度
    @tool
    def calculate_complexity(problem: str) -> str:
        """评估问题复杂度。"""
        return f"问题 '{problem}' 的复杂度: 中等"

    # 初始化Agent
    agent = create_agent(
        model=simple_model,
        tools=[calculate_complexity],
        system_prompt="你是一个问题分析助手。",
        middleware=[retry_on_failure, dynamic_model_selector],
    )

    # 测试
    print("\n--- 测试: 动态模型选择 ---")
    result = agent.invoke(
        {"messages": [("user", "帮我分析一下: 如何设计一个分布式系统? ")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result['messages'][-1].content}")

# ==================================================
# 示例6：综合实战 - 智能客服系统
# ==================================================
def example_6():
    """
    示例 6: 综合实战 - 智能客服系统
    目标: 综合运用提示词和中间件
    知识点:
    - @dynamic_prompt 动态提示词
    - @wrap_tool_call 工具监控
    - @before_model 消息限流
    - 完整的实战项目
    """
    print("\n==== 示例 6: 综合实战 - 智能客服系统 ====")

    from dataclasses import dataclass

    # ------------------------------
    # @wrap_tool_call 装饰器说明
    # - 包装工具调用的中间件，在每个工具执行前后运行
    # - request 参数：
    #   - request.tool_call: 工具调用信息（dict）
    #     包含 "name"（工具名）、"args"（参数）、"id"（调用ID）
    # - handler(request): 执行实际工具调用，必须调用，否则工具不会执行
    # ------------------------------

    # 客户上下文
    @dataclass
    class CustomerContext:
        """客户上下文。"""
        customer_type: str # "vip", "regular"
        language: str = "zh"

    # 1. 动态客服提示词
    @dynamic_prompt
    def customer_service_prompt(request: ModelRequest) -> str:
        """生成客服提示词。"""
        customer_type = "regular"
        if request.runtime and request.runtime.context:
            customer_type = request.runtime.context.customer_type

        base_prompt = "你是一个专业的电商客服助手。"

        if customer_type == "vip":
            return f"""{base_prompt}
客户类型: VIP 客户
- 提供最优质的服务体验
- 使用尊敬的称呼 (如"尊敬的 VIP 客户")"""
        else:
            return f"""{base_prompt}
客户类型: 普通客户
- 友好、专业地回答问题"""

    # 2. 工具调用监控
    @wrap_tool_call
    def monitor_tool_usage(request, handler: Callable[[Any], Any]) -> Any:
        """监控工具调用并记录日志。"""
        tool_name = request.tool_call.get("name", "unknown")
        print(f"[工具监控] 开始执行: {tool_name}")
        try:
            result = handler(request)
            print(f"[工具监控] 执行成功: {tool_name}")
            return result
        except Exception as e:
            print(f"[工具监控] 执行失败: {tool_name} - {e}")
            from langchain.messages import ToolMessage
            return ToolMessage(
                content=f"工具执行失败: {e}",
                tool_call_id=request.tool_call.get("id", ""),
            )

    # 3. 消息限流中间件
    @before_model
    def rate_limiter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """简单的消息限流。"""
        from langchain.messages import AIMessage
        if len(state["messages"]) > 15:
            return {
                "messages": [AIMessage("您的消息过于频繁, 请稍后再试。")],
                "jump_to": "end",
            }
        return None

    # 客服工具：查询订单状态
    @tool
    def query_order_status(order_id: str) -> str:
        """
        查询订单状态。
        参数: order_id - 订单编号
        返回: 订单状态信息
        """
        orders = {
            "ORD001": {"status": "已发货", "logistics": "顺丰 SF123456"},
            "ORD002": {"status": "处理中", "logistics": "待发货"},
        }
        order = orders.get(order_id)
        if not order:
            return f"未找到订单 {order_id}"
        return f"订单 {order_id}: {order['status']}, 物流: {order['logistics']}"

    # 客服工具：查询优惠活动
    @tool
    def get_discounts(customer_type: str = "regular") -> str:
        """
        查询当前优惠活动。
        参数: customer_type - 客户类型
        返回: 优惠活动列表
        """
        if customer_type == "vip":
            return "VIP 专属优惠: 全场 8 折 + 免费包邮"
        return "当前活动: 新用户注册立减 50 元, 满 200 包邮"

    # 初始化模型与Agent
    model = init_chat_model(
        "doubao-seed-2.0-lite",
        model_provider="openai",
        temperature=0.5,
    )
    agent = create_agent(
        model=model,
        tools=[query_order_status, get_discounts],
        middleware=[customer_service_prompt, monitor_tool_usage, rate_limiter],
        context_schema=CustomerContext,
    )

    # 测试场景1: VIP客户查询订单
    print("\n--- 测试 1: VIP 客户查询订单 ---")
    result1 = agent.invoke(
        {"messages": [("user", "帮我查一下订单 ORD001")]},
        config={"callbacks": [langfuse_handler]},
        context=CustomerContext(customer_type="vip"),
    )
    print(f"AI: {result1['messages'][-1].content}")

    # 测试场景2: 普通客户咨询优惠
    print("\n--- 测试 2: 普通客户咨询优惠 ---")
    result2 = agent.invoke(
        {"messages": [("user", "有什么优惠活动吗? ")]},
        config={"callbacks": [langfuse_handler]},
        context=CustomerContext(customer_type="regular"),
    )
    print(f"AI: {result2['messages'][-1].content}")

# ==================================================
# 主函数：运行指定示例
# ==================================================
def main(example_number: int):
    """运行指定的示例。"""
    print("=" * 60)
    print("第四课: 提示词工程与中间件")
    print("=" * 60)

    examples = {
        1: example_1, # 静态提示词设计
        2: example_2, # 动态提示词 (@dynamic_prompt)
        3: example_3, # AgentState 和 Runtime 详解
        4: example_4, # 中间件 - 7 种 hook 完整生命周期
        5: example_5, # wrap_model_call
        6: example_6, # 综合实战 - 智能客服
    }

    if example_number in examples:
        examples[example_number]()
    else:
        print(f"错误: 示例编号 {example_number} 不存在, 请选择 1-6")

if __name__ == "__main__":
    main(5) # 修改为1-6运行不同示例