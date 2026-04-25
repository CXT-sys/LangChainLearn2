"""
长期记忆实现
==========
第十课：记忆系统 - 长期记忆
模块：3.2
目标：掌握跨会话的长期记忆实现
知识点：
-- 持久化存储
-- 用户画像构建
-- 记忆检索与更新
-- Profile vs Collection 模式
"""

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.postgres import PostgresStore
import psycopg
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
langfuse_handler = CallbackHandler()

DB_URI = "postgresql://xiaoming:123123@localhost:5433/ai_memory"

def get_postgres_store():
    """获取 PostgreSQL 存储实例（所有示例共用）。
    直接使用 psycopg.Connection + PostgresStore 构造函数。
    PostgresStore 内部会自动管理 JSONB 序列化。
    注意：PostgresStore 要求所有 value 必须为 dict 类型。
    """
    from psycopg_pool import ConnectionPool
    from psycopg.rows import dict_row

    connection_kwargs = {
        "autocommit": True,  # 【必须】关闭隐式事务。因为 setup() 需要执行 CREATE INDEX CONCURRENTLY，该命令不能在事务块内运行
        "prepare_threshold": 0,  # 【推荐】关闭预编译语句。防止连接池在多进程/重启后出现 "Cached plan must not change result type" 报错
        "row_factory": dict_row,  # 【推荐】将查询结果转为字典格式（如 `{"thread_id": "xxx"}`），方便 LangGraph 底层按字段名读取
    }
    pool = ConnectionPool(
        conninfo=DB_URI,
        max_size=5,  # 连接池最大连接数，限制并发请求数量，保护数据库连接资源
        kwargs=connection_kwargs,
    )
    store = PostgresStore(pool)
    store.setup()  # 自动创建/迁移 store 表
    return store


# ————————————————————————————————
# Store 实现对比：InMemoryStore vs PostgresStore
# ————————————————————————————————
# 本课程统一使用 PostgresStore 演示生产环境标准做法。
#
# InMemoryStore（内存存储）：
# -- 数据存在进程内存，重启丢失。
# -- 优点：零配置，适合快速原型验证。
# -- 缺点：无法跨进程/跨部署共享数据。
# -- 用法：store = InMemoryStore()
#
# PostgresStore（PostgreSQL 存储）：
# -- 数据持久化到数据库，重启不丢，支持高并发。
# -- 优点：生产级，数据可靠，支持分布式。
# -- 缺点：需运行 PostgreSQL 服务。
# -- 用法：见 get_postgres_store()
#
# 两者 API 完全一致（put/get/search），迁移只需改一行代码。
# ————————————————————————————————

def example_1():
    """
    示例 1：基础长期记忆 - 用户信息存储
    目标：理解长期记忆的基本概念
    知识点：
    -- Store 的持久化
    -- 跨会话数据保留
    -- 用户信息管理
    """
    print("\n===== 示例 1：基础长期记忆 - 用户信息存储 =====")

    # 创建持久化存储（生产环境使用 PostgresStore）
    store = get_postgres_store()

    @dataclass
    class UserContext:
        user_id: str

    @tool
    def save_profile(name: str, age: int, occupation: str, runtime: ToolRuntime[UserContext]) -> str:
        """
        保存用户个人信息。
        信息将长期保存，跨会话可用。
        参数:
            name: 姓名
            age: 年龄
            occupation: 职业
        """
        user_id = runtime.context.user_id
        profile = {"name": name, "age": age, "occupation": occupation}
        store.put(("profiles",), user_id, profile)
        return f"✅ 个人信息已保存\n姓名: {name}\n年龄: {age}\n职业: {occupation}"

    @tool
    def get_profile(runtime: ToolRuntime[UserContext]) -> str:
        """获取保存的个人信息。"""
        user_id = runtime.context.user_id
        profile = store.get(("profiles",), user_id)
        if profile:
            p = profile.value
            return f"个人信息:\n- 姓名: {p['name']}\n- 年龄: {p['age']}\n- 职业: {p['occupation']}"
        return "暂无个人信息，请先保存资料"

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[save_profile, get_profile],
        system_prompt="你是一个个人信息管理助手。",
        context_schema=UserContext,
        store=store,
    )

    print("\n--- 第一次会话：保存信息 ---")
    result1 = agent.invoke(
        {"messages": [("user", "我叫张三，28 岁，是一名程序员")]},
        config={"configurable": {"thread_id": "session_001"}, "callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 第二次会话（新的 thread_id）：查询信息 ---")
    result2 = agent.invoke(
        {"messages": [("user", "我的个人信息是什么?")]},
        config={"configurable": {"thread_id": "session_002"}, "callbacks": [langfuse_handler]},  # 不同的会话
        context=UserContext(user_id="U001"),
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_2():
    """
    示例 2：Checkpointer + Store 组合使用
    目标：展示短期记忆与长期记忆如何配合
    知识点：
    -- Checkpointer 维持当前对话流（短期）
    -- Store 保存用户长期画像（长期）
    -- 两者同时传入 create_agent
    """
    print("\n===== 示例 2：Checkpointer + Store 组合使用 =====")

    store = get_postgres_store()

    # 创建 Checkpointer（短期记忆）
    from langgraph.checkpoint.postgres import PostgresSaver

    check_conn = psycopg.connect(DB_URI, autocommit=True, prepare_threshold=0)
    checkpointer = PostgresSaver(check_conn)
    checkpointer.setup()

    @dataclass
    class UserContext:
        user_id: str

    @tool
    def add_interest_tag(tag: str, runtime: ToolRuntime[UserContext]) -> str:
        """
        添加用户兴趣标签到 Store（长期记忆）。
        参数:
            tag: 兴趣标签
        """
        user_id = runtime.context.user_id
        data = store.get(("interests",), user_id)
        tags = data.value.get("tags", []) if data else []
        if tag not in tags:
            tags.append(tag)
        store.put(("interests",), user_id, {"tags": tags})
        return f"已添加兴趣标签: {tag}"

    @tool
    def get_user_persona(runtime: ToolRuntime[UserContext]) -> str:
        """从 Store 读取用户画像（长期记忆）。"""
        user_id = runtime.context.user_id
        interests = store.get(("interests",), user_id)
        tags = interests.value.get("tags", []) if interests else []

        persona = f"用户画像 (ID: {user_id}):\n"
        persona += f"兴趣标签: {', '.join(tags) if tags else '暂无'}\n"
        if "编程" in tags and "AI" in tags:
            persona += "分析: 技术型人才，对 AI 和编程感兴趣"
        elif "编程" in tags:
            persona += "分析: 编程爱好者"
        return persona

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    agent = create_agent(
        model=model,
        tools=[add_interest_tag, get_user_persona],
        system_prompt="你是一个用户画像分析助手。",
        context_schema=UserContext,
        store=store,
        checkpointer=checkpointer,  # 同时传入 checkpointer 和 store
    )

    # ————————————————————————————————
    # 测试 1：同一个 thread_id 内多轮对话
    # Checkpointer 让 AI 记得刚才说了什么
    # Store 让 AI 记得"长期兴趣是什么"
    # ————————————————————————————————
    print("\n--- 第 1 轮：添加第一个兴趣 ---")
    agent.invoke(
        {"messages": [("user", "我对编程感兴趣")]},
        config={"configurable": {"thread_id": "persona_001"}, "callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001"),
    )

    print("\n--- 第 2 轮：Checkpointer 记住上一轮（AI 知道刚才聊过编程） ---")
    agent.invoke(
        {"messages": [("user", "我还喜欢 AI")]},
        config={"configurable": {"thread_id": "persona_001"}, "callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001"),
    )

    print("\n--- 第 3 轮：读取画像（Store 提供了长期数据） ---")
    result = agent.invoke(
        {"messages": [("user", "生成我的用户画像")]},
        config={"configurable": {"thread_id": "persona_001"}, "callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI: {result['messages'][-1].content}")

    # ————————————————————————————————
    # 测试 2：不同 thread_id，但同一个 user_id
    # Checkpointer 是空的（新会话），但 Store 数据还在
    # ————————————————————————————————
    print("\n--- 新会话：Checkpointer 重置，但 Store 数据仍在 ---")
    result2 = agent.invoke(
        {"messages": [("user", "生成我的用户画像")]},
        config={"configurable": {"thread_id": "persona_002"}, "callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI: {result2['messages'][-1].content}")
    print("\n✅ 说明: 新会话虽然不记得上次对话内容，但 Store 记住了你的长期兴趣!")


def example_3():
    """
    示例 3：记忆检索与搜索
    目标：掌握长期记忆的检索能力
    知识点：
    -- 全文搜索
    -- 按条件过滤
    -- 相关性排序
    """
    print("\n===== 示例 3：记忆检索与搜索 =====")

    store = get_postgres_store()

    store.put(
        ("notes", "datas"),
        "note_1",
        {"title": "Python 装饰器", "content": "装饰器用于修改函数行为", "tags": ["python", "编程"]},
    )
    store.put(
        ("notes", "datas"),
        "note_2",
        {"title": "机器学习基础", "content": "机器学习是 AI 的核心", "tags": ["AI", "机器学习"]},
    )
    store.put(
        ("notes", "datas"),
        "note_3",
        {"title": "Git 使用技巧", "content": "Git 是版本控制工具", "tags": ["git", "编程"]},
    )

    @tool
    def search_notes(keyword: str) -> str:
        """
        搜索笔记。
        参数:
            keyword: 搜索关键词
        """
        # 使用 filter 过滤（演示参数用法）
        # notes = store.search(("notes", "datas"), filter={"tags": ["python"]})
        # 这里用 search() 获取所有笔记，再手动过滤关键词
        notes = store.search(("notes", "datas"))  # 获取 namespace 下所有条目
        # ————————————————————————————————
        # store.search() 常用参数详解
        # ————————————————————————————————
        # search(namespace_prefix, query=None, filter=None, limit=10, offset=0)
        #
        # namespace_prefix: tuple[str, ...] 【必需】
        # 命名空间前缀匹配。例: ("notes",) 会匹配 ("notes", "datas") 下的所有条目。
        # 类似文件系统的目录搜索: search("docs/") → 返回 docs/ 下的所有文件。
        #
        # filter: dict 【常用】
        # 按 value 中的字段值过滤。例: filter={"type": "article"} 只返回 value 中包含 `{"type": "article"}` 的条目。
        #
        # limit: int 【常用】
        # 最多返回多少条结果，默认 10。分页时用 offset + limit 控制。
        #
        # offset: int 【分页用】
        # 跳过前 N 条结果，用于翻页。例: limit=5, offset=5 → 返回第 6-10 条。
        #
        # query: str 【语义搜索】
        # 自然语言查询（需要 Store 配置了 embedding 向量模型）。
        # 例: query="机器学习应用" → 返回语义相近的条目。
        #
        # 注意: PostgresStore 默认不支持向量搜索，需额外配置。
        #
        # 返回值: list[SearchItem]，每个 item 有 .key、.value、.namespace 等属性。
        # ————————————————————————————————

        # 预存笔记
        results = []
        for note in notes:
            if (
                keyword.lower() in note.value.get("title", "").lower()
                or keyword.lower() in note.value.get("content", "").lower()
                or keyword.lower() in "-".join(note.value.get("tags", [])).lower()
            ):
                results.append(note.value)

        if not results:
            return f"未找到包含 '{keyword}' 的笔记"

        output = f"找到 {len(results)} 条笔记:\n"
        for i, note in enumerate(results, 1):
            output += f"{i}. {note['title']}\n  {note['content']}\n  标签: {', '.join(note['tags'])}"
        return output

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model, tools=[search_notes], system_prompt="你是一个笔记管理助手，帮助搜索和管理笔记。", store=store
    )

    print("\n--- 测试 1：搜索 Python 相关 ---")
    result1 = agent.invoke({"messages": [("user", "搜索关于 Python 的笔记")]}, config={"callbacks": [langfuse_handler]})
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2：搜索编程相关 ---")
    result2 = agent.invoke(
        {"messages": [("user", "搜索所有跟编程相关的内容")]}, config={"callbacks": [langfuse_handler]}
    )
    print(f"AI: {result2['messages'][-1].content}")

def example_4():
    """
    示例 4：Profile vs Collection 模式
    目标：理解两种存储模式的区别
    知识点：
    -- Profile：单用户单一档案
    -- Collection：多条目集合
    """
    print("\n===== 示例 4：Profile vs Collection 模式 =====")

    store = get_postgres_store()

    @dataclass
    class UserContext:
        user_id: str

    @tool
    def set_user_preference(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
        """
        设置用户偏好（Profile 模式）。
        每个用户只有一个配置。
        """
        user_id = runtime.context.user_id
        data = store.get(("prefs",), user_id)
        prefs = data.value if data else {}
        prefs[key] = value
        store.put(("prefs",), user_id, prefs)
        return f"已保存: {key} = {value}"

    @tool
    def add_bookmark(url: str, title: str, runtime: ToolRuntime[UserContext]) -> str:
        """
        添加书签（Collection 模式）。
        用户可以有多个书签。
        """
        user_id = runtime.context.user_id
        bookmarks = store.search(("bookmarks", user_id))
        bookmark_id = len(bookmarks) + 1
        store.put(("bookmarks", user_id), f"bm_{bookmark_id}", {"url": url, "title": title})
        return f"已添加书签: {title} ({url})"

    @tool
    def list_bookmarks(runtime: ToolRuntime[UserContext]) -> str:
        """列出所有书签。"""
        user_id = runtime.context.user_id
        bookmarks = store.search(("bookmarks", user_id))
        if not bookmarks:
            return "暂无书签"

        output = "书签列表:\n"
        for bm in bookmarks:
            output += f"- {bm.value['title']}: {bm.value['url']}\n"
        return output

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[set_user_preference, add_bookmark, list_bookmarks],
        system_prompt="你是一个个人助手，管理用户的偏好和书签。",
        context_schema=UserContext,
        store=store,
    )

    print("\n--- 测试 1: Profile 模式 ---")
    result1 = agent.invoke(
        {"messages": [("user", "把我的主题设置为暗色")]},
        config={"configurable": {"thread_id": "profile_demo"}, "callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: Collection 模式 ---")
    result2 = agent.invoke(
        {"messages": [("user", "添加一个书签: GitHub，网址 github.com")]},
        config={"configurable": {"thread_id": "profile_demo"}, "callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_5():
    """
    示例 5：综合实战 - 个人学习助手
    目标：创建带长期记忆的学习助手
    知识点：
    -- 学习进度追踪
    -- 知识点掌握情况
    -- 个性化推荐
    -- 完整实现
    """
    print("\n===== 示例 5：综合实战 - 个人学习助手 =====")

    store = get_postgres_store()

    @dataclass
    class UserContext:
        user_id: str

    # 课程数据库
    courses = {
        "python_base": {"name": "Python 基础", "difficulty": "入门", "duration": "10 小时"},
        "python_advanced": {"name": "Python 进阶", "difficulty": "高级", "duration": "20 小时"},
        "ai_intro": {"name": "AI 入门", "difficulty": "入门", "duration": "8 小时"},
        "ml_practice": {"name": "机器学习实战", "difficulty": "高级", "duration": "30 小时"},
    }

    @tool
    def record_progress(course_id: str, progress: int, runtime: ToolRuntime[UserContext]) -> str:
        """
        记录学习进度。
        参数:
            course_id: 课程 ID
            progress: 进度百分比 (0-100)
        """
        user_id = runtime.context.user_id
        store.put(("progress", user_id), course_id, {"progress": progress})

        course = courses.get(course_id, {"name": course_id})
        return f"已记录: {course['name']} - 进度 {progress}%"

    @tool
    def get_learning_report(runtime: ToolRuntime[UserContext]) -> str:
        """生成学习报告。"""
        user_id = runtime.context.user_id
        progress_data = store.search(("progress", user_id))

        if not progress_data:
            return "暂无学习记录"

        report = "📊 学习报告:\n"
        for item in progress_data:
            course_id = item.key
            course = courses.get(course_id, {"name": course_id})
            progress = item.value.get("progress", 0)
            status = "✅ 完成" if progress >= 100 else f"🔄 {progress}%"
            report += f"\n- {course['name']}: {status}"
        return report

    @tool
    def recommend_courses(runtime: ToolRuntime[UserContext]) -> str:
        """推荐课程。"""
        user_id = runtime.context.user_id
        progress_data = store.search(("progress", user_id))
        completed = [item.key for item in progress_data if item.value.get("progress", 0) >= 100]

        recommendations = []
        if "python_base" in completed:
            recommendations.append("python_advanced")
        if "ai_intro" in completed:
            recommendations.append("ml_practice")

        if not recommendations:
            return "推荐课程: Python 基础、AI 入门"

        rec_courses = [courses[cid]["name"] for cid in recommendations]
        return f"推荐课程:\n" + "\n".join(f"- {name}" for name in rec_courses)

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    agent = create_agent(
        model=model,
        tools=[record_progress, get_learning_report, recommend_courses],
        system_prompt="""你是一个个人学习助手，可以:
- 记录学习进度
- 生成学习报告
- 推荐适合的课程
长期保存用户的学习数据。""",
        context_schema=UserContext,
        store=store,
    )

    print("\n--- 测试 1: 记录进度 ---")
    result1 = agent.invoke(
        {"messages": [("user", "我完成了 Python 基础课程的 50%")]},
        config={"configurable": {"thread_id": "learning_001"}, "callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 学习报告 ---")
    result2 = agent.invoke(
        {"messages": [("user", "生成我的学习报告")]},
        config={"configurable": {"thread_id": "learning_001"}, "callbacks": [langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI: {result2['messages'][-1].content}")


def main(example_number: int):
    """运行指定的示例。"""
    print("=" * 60)
    print("第十课: 长期记忆实现")
    print("=" * 60)

    examples = {1: example_1, 2: example_2, 3: example_3, 4: example_4, 5: example_5}

    if example_number in examples:
        examples[example_number]()
    else:
        print(f"错误: 示例编号 {example_number} 不存在")


if __name__ == "__main__":
    main(5)

# ————————————————————————————————
# 核心总结：LangGraph 上下文工程全景对比
# ————————————————————————————————

# 对比项              静态上下文(Static Context)   动态上下文(Dynamic Context)   短期记忆(Short-term)        长期记忆(Long-term)
# ----------------------------------------------------------------------------------------------------------------------
# 核心组件            context_schema                AgentState / Runtime.state    Checkpointer                Store
# ----------------------------------------------------------------------------------------------------------------------
# 数据流向            外部 → Agent                   节点间流转                    DB <--> Agent（自动）       DB <--> Agent（手动）
# ----------------------------------------------------------------------------------------------------------------------
# 生命周期            单次 invoke()                  单次图执行                    跨轮次（同一 thread_id）    永久（跨 session）
# ----------------------------------------------------------------------------------------------------------------------
# 隔离方式            每次调用独立传参               图运行结束即清空              thread_id 隔离              namespace + key 隔离
# ----------------------------------------------------------------------------------------------------------------------
# 访问 API            runtime.context                runtime.state                 自动加载/保存               store.get / store.put
# ----------------------------------------------------------------------------------------------------------------------
# 典型场景            "我是谁?"                      "我现在在干什么?"             "我们刚才说了什么?"         "我上次来喜欢什么?"
# （用户 ID、语言设置）                              （任务进度、中间变量）        （多轮对话）                （用户画像、收藏夹）
# ----------------------------------------------------------------------------------------------------------------------
# 生产实现            代码定义 Dataclass             代码定义 TypedDict            PostgresSaver               PostgresStore
# ----------------------------------------------------------------------------------------------------------------------

# 总结：
# 1. 静态上下文 是"入场券"：告诉 Agent 当前环境参数（如当前用户）。
# 2. 动态上下文 是"草稿纸"：Agent 干活时临时记一下进度。
# 3. 短期记忆 是"聊天记录"：保证对话不断片，AI 能接话。
# 4. 长期记忆 是"档案柜"：建立用户画像，实现个性化服务。
#
# 高级 Agent 通常是四者组合使用：
# agent = create_agent(
#    context_schema=AppContext,           <-- 1. 静态
#    checkpointer=PostgresSaver(...),     <-- 3. 短期
#    store=PostgresStore(...),            <-- 4. 长期
#    middleware=[track_state]             <-- 2. 动态（读写 state）
# )

# ————————————————————————————————
# 补充：Runtime 家族与上下文访问对比
# ————————————————————————————————
# 上述四种上下文，在代码中是如何被访问的？
# 核心桥梁就是 Runtime 和 ToolRuntime。
#
# 类型                  AgentState                  Runtime                      ToolRuntime
# 角色                  Agent 的数据结构            Agent 的执行环境              工具的专属环境
# 作用域                全图生命周期                中间件生命周期               单次工具调用周期
#
# 静态上下文(Context)    ×（只存动态状态）           ✓（runtime.context）         ✓（runtime.context）
# 动态上下文(State)      ✓（就是它本身）             ✓（runtime.state）           ✓（runtime.state）
# 短期记忆(Checkpoint)   ×（只负责存）               ×（框架自动处理）            ×（框架自动处理）
# 长期记忆(Store)        ×（只负责调用）              ×（需手动传入 store）        ✓（需手动调用 store）
#
# 典型用法              定义 Schema                  中间件中读取 context         工具中读取 state/context

# ————————————————————————————————
# 重点理解 ToolRuntime 的泛型写法：
# @tool
# def my_tool(runtime: ToolRuntime[AppContext]):
#    ctx = runtime.context          # 静态上下文（AppContext）
#    state = runtime.state          # 动态上下文（AgentState）
#    store.put(...)                 # 长期记忆
#
# ToolRuntime 就像一个"万能插座"：
# - 左边插着 AppContext（静态）
# - 右边插着 AgentState（动态）
# - 中间连着 Store（长期）

# ————————————————————————————————
# 概念回顾：上下文工程 vs 记忆
# ————————————————————————————————
# 很多初学者容易把"上下文工程"和"记忆"混为一谈。
# 实际上，上下文工程是一个更大的框架，记忆只是其中的一部分。
# LangGraph 的上下文工程包含 4 个层次：
#
# 1. 静态上下文 (context_schema)
#    生命周期：单次 invoke() 调用
#    作用：传递环境参数（用户 ID、语言、时区等）
#    类比：进门时出示的身份证
#
# 2. 动态上下文 (AgentState / Command)
#    生命周期：单次图的执行过程
#    作用：Agent 在执行中读写的状态（话题、进度等）
#    类比：执行任务时的草稿纸
#
# 3. 短期记忆 (Checkpointer)
#    生命周期：跨轮次（同一 thread_id 内）
#    作用：记住这次会话说了什么，保持对话连续
#    类比：本次聊天的聊天记录
#
# 4. 长期记忆 (Store) ← 本课重点
#    生命周期：永久（跨会话、跨天）
#    作用：记住用户长期偏好、画像、历史结论
#    类比：档案柜，下次来还能翻出来
#
# 总结：
#    短期/长期记忆解决"记住"的问题。
#    上下文工程解决"如何把正确的信息在正确的时间给到 Agent"。
#    好 Agent 通常是 4 层组合使用。

# ————————————————————————————————
# AgentState / Runtime / ToolRuntime 对比
# ————————————————————————————————
# 对比项                AgentState                  Runtime                      ToolRuntime
# 含义                  Agent 的内存状态             Agent 执行时的运行环境        工具执行时的运行环境
#                       （TypedDict 字典）           （对象）                     （对象）
# 生命周期              整个 Agent 执行期间          单次 invoke/stream           单次工具调用期间
#
# 内置属性              1. messages（必需）          1. context                  1. context
#                       2. jump_to（可选）           2. model                    2. state
#                       3. structured_response                                   3. stream_writer
#
# 使用场景              中间件读写 Agent 状态        中间件访问上下文             工具内访问 Agent 状态
# 创建方式              LangChain 自动创建           LangChain 自动创建           LangChain 自动注入
#
# 使用示例              @before_model                @before_model               @tool
#                       def mw(state, rt):            def mw(state, rt):          def tool(x, rt):
#                          msgs = state["messages"]      ctx = rt.context           ctx = rt.context
#                                                             msgs = rt.state["messages"]
