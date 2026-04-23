"""
SQL Agent 实战
==============
第七课 LangChain 核心应用 - 数据库交互
模块: 2.3
目标: 掌握 SQL Agent 的构建和使用
知识点:
-- SQLDatabase 组件
-- SQLDatabaseToolkit
-- 数据库连接与安全
-- Human-in-the-loop 审核
-- 真实的 SQL Agent 实现
"""

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
import sqlite3

load_dotenv()
langfuse_handler = CallbackHandler()


def _create_sample_db(db_path: str = "./lesson07_sample.db"):
    """创建示例 SQLite 数据库。"""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()  # 用于执行 SQL 命令并获取结果的指针

    # 创建用户表
    cursor.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            city TEXT,
            salary REAL
        )
        """
    )

    # 创建订单表
    cursor.execute(
        """
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product TEXT,
            amount REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # 插入测试数据
    users = [
        (1, "张三", 28, "北京", 15000),
        (2, "李四", 32, "上海", 20000),
        (3, "王五", 25, "广州", 12000),
        (4, "赵六", 35, "深圳", 25000),
        (5, "钱七", 30, "北京", 18000),
    ]

    # execute: 一条一条地执行 SQL (效率低)。
    # executemany: 一次性批量执行多条结构相同的 SQL (效率高)。
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?)", users)

    orders = [
        (1001, 1, "iPhone", 5999),
        (1002, 2, "MacBook", 12999),
        (1003, 1, "AirPods", 1999),
        (1004, 3, "iPad", 4799),
        (1005, 4, "MacBook Pro", 18999),
    ]
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)

    conn.commit()
    conn.close()
    print(f"✅ 示例数据库已创建: {db_path}")


def example_1():
    """
    示例 1: 真实 SQLDatabase 与基础查询
    目标: 理解 LangChain SQLDatabase 组件
    知识点:
    -- SQLDatabase 初始化
    -- table_info 查看表结构
    -- 安全执行 SQL 查询
    """
    print("\n===== 示例 1: 真实 SQLDatabase 与基础查询 =====")

    # 1. 创建示例数据库
    _create_sample_db()

    # ================================================
    # SQLDatabase 组件详解
    # ================================================
    # SQLDatabase 是 LangChain 提供的数据库抽象层。
    # 它封装了 SQLAlchemy 引擎, 提供安全的 SQL 交互接口。
    #
    # 支持的数据库 (任何 SQLAlchemy 兼容的数据库):
    # - SQLite: "sqlite:///./example.db"
    # - PostgreSQL: "postgresql://user:pass@localhost/dbname"
    # - MySQL: "mysql://user:pass@localhost/dbname"
    # - Oracle: "oracle://user:pass@localhost/dbname"
    #
    # 核心功能:
    # 1. get_table_names(): 列出所有表
    # 2. get_table_info(): 查看表结构 (DDL 语句)
    # 3. run(sql): 执行 SQL 查询并返回结果
    # 4. get_context(): 获取数据库上下文信息 (用于 Agent)
    # ================================================

    # 2. 初始化 SQLDatabase
    db = SQLDatabase.from_uri("sqlite:///./lesson07_sample.db")

    # 3. 查看数据库结构
    print("\n--- 数据库结构信息 ---")
    print(f"可用表: {db.get_table_names()}")
    print(f"\n表结构 (table_info):\n{db.table_info}")

    # 4. 测试直接查询
    print("\n--- 测试直接查询 ---")
    result = db.run("SELECT * FROM users LIMIT 2")
    print(f"查询结果:\n{result}")

    # ================================================
    # 5. 使用 SQLDatabaseToolkit 创建 Agent
    # ================================================
    # SQLDatabaseToolkit 是 LangChain 提供的 SQL 工具包。
    # 它会自动生成多个工具供 Agent 使用:
    #
    # -- sql_db_query: 执行 SQL 查询
    # -- sql_db_schema: 查询表结构
    # -- sql_db_list_tables: 列出所有表
    # -- sql_db_query_checker: 检查 SQL 语法
    #
    # Agent 会自动决定使用哪个工具:
    # 1. 先列出可用表 (sql_db_list_tables)
    # 2. 查看表结构 (sql_db_schema)
    # 3. 生成并执行 SQL (sql_db_query)
    # 4. 如有错误, 检查语法并重试 (sql_db_query_checker)
    # ================================================

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    # 创建 SQL 工具包
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    print(f"\n--- Toolkit 提供的工具 ---")
    for t in tools:
        print(f" - {t.name}: {t.description}")

    # 创建 SQL Agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""你是一个数据库助手, 帮助用户查询数据。
请使用提供的工具来探索数据库结构并执行查询。
如果查询失败, 请分析错误原因并尝试修正。""",
    )

    print("\n--- 测试 1: 查询所有用户 ---")
    result1 = agent.invoke(
        {"messages": [("user", "列出所有用户的姓名和城市")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 条件查询 ---")
    result2 = agent.invoke(
        {"messages": [("user", "查询薪资超过 15000 的用户")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_2():
    """
    示例 2: 手动封装 SQL 工具与权限控制
    目标: 掌握如何自定义 SQL 工具
    知识点:
    -- 手动创建查询工具
    -- SQL 注入防护
    -- 只读访问控制
    -- 查询结果格式化
    """
    print("\n===== 示例 2: 手动封装 SQL 工具与权限控制 =====")

    # 1. 初始化数据库 (确保数据库存在)
    _create_sample_db()
    db = SQLDatabase.from_uri("sqlite:///./lesson07_sample.db")

    # ================================================
    # 为什么需要手动封装 SQL 工具?
    # ================================================
    # 虽然 SQLDatabaseToolkit 提供了开箱即用的工具,
    # 但在生产环境中, 我们通常需要:
    # 1. 限制查询类型: 只允许 SELECT, 禁止 INSERT/UPDATE/DELETE
    # 2. 限制访问表: 只允许访问特定表
    # 3. 限制返回行数: 防止全表扫描拖慢系统
    # 4. 自定义错误处理: 友好的错误提示
    # 5. 查询日志: 记录所有执行的 SQL
    #
    # 下面演示如何手动封装一个安全的 SQL 查询工具。
    # ================================================

    @tool
    def safe_sql_query(sql: str) -> str:
        """
        安全执行 SQL 查询 (只读模式)。

        参数:
            sql: SQL 查询语句

        返回:
            查询结果 (最多返回 10 行)
        """
        # 安全检查 1: 只允许 SELECT
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith("SELECT"):
            return "❌ 错误: 只允许执行 SELECT 查询"

        # 安全检查 2: 阻止危险关键字
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
        for keyword in dangerous:
            if keyword in sql_stripped:
                return f"❌ 错误: 禁止使用 {keyword} 操作"

        # 安全检查 3: 限制返回行数 (自动追加 LIMIT)
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + " LIMIT 10"

        # 执行查询
        try:
            result = db.run(sql)
            if not result or result.strip() == "":
                return "查询结果为空"
            return f"✅ 查询成功:\n{result}"
        except Exception as e:
            return f"❌ 查询执行失败: {str(e)}"

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[safe_sql_query],
        system_prompt="""你是一个数据库助手, 帮助用户安全地查询数据。
- 只执行 SELECT 查询
- 将 SQL 结果格式化为易读的格式
- 如果查询失败, 解释错误原因""",
    )

    print("\n--- 测试 1: 正常查询 ---")
    result1 = agent.invoke(
        {"messages": [("user", "查询 users 表中所有用户的姓名和薪资")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 尝试危险操作 (应被拦截) ---")
    result2 = agent.invoke(
        {"messages": [("user", "DELETE FROM users WHERE id = 1")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_3():
    """
    示例 3: 数据分析 Agent
    目标: 创建能进行数据分析的 SQL Agent
    知识点:
    -- SQL 聚合查询 (COUNT, SUM, AVG, GROUP BY)
    -- 多表 JOIN 查询
    -- 统计分析
    """
    print("\n===== 示例 3: 数据分析 Agent =====")

    # 1. 初始化数据库
    _create_sample_db()
    db = SQLDatabase.from_uri("sqlite:///./lesson07_sample.db")

    @tool
    def analyze_data(query_type: str) -> str:
        """
        执行数据分析查询。

        参数:
            query_type: 分析类型
                - "user_stats": 用户统计 (平均薪资、最高薪资等)
                - "order_stats": 订单统计 (总订单数、总金额等)
                - "user_orders": 用户订单分析 JOIN 查询

        返回:
            分析结果
        """
        try:
            if query_type == "user_stats":
                sql = """
                    SELECT
                        COUNT(*) as 用户总数,
                        ROUND(AVG(salary), 0) as 平均薪资,
                        MAX(salary) as 最高薪资,
                        MIN(salary) as 最低薪资,
                        ROUND(AVG(age), 1) as 平均年龄
                    FROM users
                """
                result = db.run(sql)
                return f"📊 用户统计分析:\n{result}"

            elif query_type == "order_stats":
                sql = """
                    SELECT
                        COUNT(*) as 订单总数,
                        SUM(amount) as 订单总金额,
                        ROUND(AVG(amount), 0) as 平均订单金额,
                        MAX(amount) as 最高订单金额
                    FROM orders
                """
                result = db.run(sql)
                return f"📊 订单统计分析:\n{result}"

            elif query_type == "user_orders":
                sql = """
                    SELECT
                        u.name as 用户姓名,
                        COUNT(o.order_id) as 订单数,
                        SUM(o.amount) as 消费总额
                    FROM users u
                    LEFT JOIN orders o ON u.id = o.user_id
                    GROUP BY u.name
                    ORDER BY 消费总额 DESC
                """
                result = db.run(sql)
                return f"📊 用户订单分析:\n{result}"
            else:
                return f"❌ 未知的分析类型: {query_type}"
        except Exception as e:
            return f"❌ 查询执行失败: {str(e)}"

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    agent = create_agent(
        model=model,
        tools=[analyze_data],
        system_prompt="""你是一个数据分析专家，擅长：
- 用户数据分析
- 订单数据统计
- 用户消费行为分析
请根据数据给出专业的分析和建议。""",
    )

    print("\n--- 测试 1: 用户统计 ---")
    result1 = agent.invoke(
        {"messages": [("user", "帮我统计一下用户的基本数据")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 用户消费分析 ---")
    result2 = agent.invoke(
        {"messages": [("user", "分析一下用户的订单和消费情况")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_4():
    """
    示例 4: 多数据源整合 - BI Agent
    目标: 创建多数据库整合的商业智能 Agent
    知识点:
    -- 多个 SQLDatabase 实例
    -- 跨库查询
    -- 业务指标计算
    """
    print("\n===== 示例 4: 多数据源整合 - BI Agent =====")

    # 创建销售数据库
    sales_db_path = "./lesson07_sales.db"
    if os.path.exists(sales_db_path):
        os.remove(sales_db_path)

    conn = sqlite3.connect(sales_db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            region TEXT,
            product TEXT,
            revenue REAL,
            month TEXT
        )
        """
    )

    cursor.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?, ?)",
        [
            (1, "华东", "产品A", 50000, "1月"),
            (2, "华东", "产品B", 40000, "1月"),
            (3, "华南", "产品A", 35000, "1月"),
            (4, "华南", "产品C", 30000, "1月"),
            (5, "华北", "产品B", 45000, "1月"),
            (6, "华北", "产品A", 55000, "2月"),
            (7, "华东", "产品C", 25000, "2月"),
        ],
    )
    conn.commit()
    conn.close()

    # 初始化销售数据库
    sales_db = SQLDatabase.from_uri(f"sqlite:///{sales_db_path}")

    # 创建客户数据库
    customer_db_path = "./lesson07_customers.db"
    if os.path.exists(customer_db_path):
        os.remove(customer_db_path)

    conn = sqlite3.connect(customer_db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            region TEXT,
            tier TEXT
        )
        """
    )

    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)",
        [
            (1, "客户A", "华东", "VIP"),
            (2, "客户B", "华南", "普通"),
            (3, "客户C", "华北", "VIP"),
            (4, "客户D", "华东", "普通"),
        ],
    )
    conn.commit()
    conn.close()

    # 初始化客户数据库
    customer_db = SQLDatabase.from_uri(f"sqlite:///{customer_db_path}")

    @tool
    def query_sales(sql: str) -> str:
        """查询销售数据库。只允许 SELECT。"""
        if not sql.strip().upper().startswith("SELECT"):
            return "❌ 错误: 只允许 SELECT 查询"
        try:
            return f"销售数据:\n{sales_db.run(sql)}"
        except Exception as e:
            return f"❌ 查询失败: {str(e)}"

    @tool
    def query_customers(sql: str) -> str:
        """查询客户数据库。只允许 SELECT。"""
        if not sql.strip().upper().startswith("SELECT"):
            return "❌ 错误: 只允许 SELECT 查询"
        try:
            return f"客户数据:\n{customer_db.run(sql)}"
        except Exception as e:
            return f"❌ 查询失败: {str(e)}"

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    agent = create_agent(
        model=model,
        tools=[query_sales, query_customers],
        system_prompt="""你是一个商业智能分析助手。
你有两个数据源:
1. 销售数据库 (sales 表: region, product, revenue, month)
2. 客户数据库 (customers 表: name, region, tier)
请使用 query_sales 或 query_customers 工具来查询数据并回答用户问题。""",
    )

    print("\n--- 测试 1: 销售分析 ---")
    result1 = agent.invoke(
        {"messages": [("user", "各区域的销售额分别是多少?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 客户分析 ---")
    result2 = agent.invoke(
        {"messages": [("user", "有多少个 VIP 客户?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_5():
    """
    示例 5: Human-in-the-loop 审核 (真实 LangGraph 实现)
    目标: 使用 LangGraph interrupt 机制实现人工审核
    知识点:
    -- interrupt_before 机制
    -- 暂停/继续执行流程
    -- 人工审核决策 (真实 input 交互)
    -- 审核拒绝时返回 AIMessage
    """
    print("\n===== 示例 5: Human-in-the-loop 审核 (真实 LangGraph) =====")

    # 1. 初始化数据库
    _create_sample_db()
    db = SQLDatabase.from_uri("sqlite:///./lesson07_sample.db")

    # 2. 创建安全的 SQL 查询工具
    @tool
    def execute_sql(sql: str) -> str:
        """执行 SQL 查询 (只允许 SELECT)。"""
        if not sql.strip().upper().startswith("SELECT"):
            return "❌ 错误: 只允许执行 SELECT 查询"
        try:
            result = db.run(sql)
            if not result or result.strip() == "":
                return "查询结果为空"
            return f"✅ 查询成功:\n{result}"
        except Exception as e:
            return f"❌ 查询失败: {str(e)}"

    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    # ================================================
    # 3. 使用 LangGraph 的 interrupt_before 机制
    # ================================================
    # interrupt_before=["tools"] 让 Agent 在执行工具前暂停。
    # 此时可以检查 Agent 准备调用的工具参数 (如 SQL),
    # 人工决定是否允许执行。
    # ================================================

    agent = create_agent(
        model,
        tools=[execute_sql],
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
    )

    # 场景 1: 简单查询 (人工审核通过)
    print("\n--- 场景 1: 简单查询 (人工审核) ---")
    config1 = {
        "configurable": {"thread_id": "sql_review_1"},
        "callbacks": [langfuse_handler],
    }

    result1 = agent.invoke(
        {"messages": [("user", "查询所有用户的姓名和城市")]},
        config=config1,
    )

    # 检查 Agent 是否生成了工具调用
    ai_msg = result1["messages"][-1]
    if ai_msg.tool_calls:
        sql_to_exec = ai_msg.tool_calls[0]["args"]["sql"]
        print(f"\n🔴 Agent 已暂停! 它准备执行以下 SQL:")
        print(f"🔍 SQL: {sql_to_exec}")

        confirm = input("\n是否批准执行? (y/n): ")
        if confirm.lower() == "y":
            print("✅ 审核通过, 继续执行...")
            final_result = agent.invoke(None, config=config1)
            print(f"最终结果: {final_result['messages'][-1].content}")
        else:
            print("❌ 审核拒绝, 操作已取消。")
            # 审核拒绝: 构造 AIMessage 返回给用户
            reject_msg = AIMessage(
                content="⚠️ 您的查询需要人工审核, 但已被管理员拒绝。\n请简化您的查询或联系管理员获取更高权限。"
            )
            # 将拒绝消息添加到对话状态中
            agent.update_state(config1, {"messages": [reject_msg]})
            print(" → 已返回拒绝消息给用户。")
            # 获取更新后的状态
            updated_state = agent.get_state(config1)
            print(f" → AI 回复: {updated_state.values['messages'][-1].content}")
    else:
        print("\nℹ️ Agent 没有调用工具, 直接回复了: ")
        print(ai_msg.content)

    # 场景 2: 复杂查询 (人工审核拒绝)
    print("\n--- 场景 2: 复杂查询 (人工审核拒绝) ---")
    config2 = {
        "configurable": {"thread_id": "sql_review_2"},
        "callbacks": [langfuse_handler],
    }

    result2 = agent.invoke(
        {"messages": [("user", "查询每个用户的订单总数, 按订单数降序排列")]},
        config=config2,
    )

    ai_msg2 = result2["messages"][-1]
    if ai_msg2.tool_calls:
        sql_to_exec2 = ai_msg2.tool_calls[0]["args"]["sql"]
        print(f"\n🔴 Agent 已暂停! 它准备执行以下 SQL:")
        print(f"🔍 SQL: {sql_to_exec2}")

        confirm2 = input("\n是否批准执行? (y/n): ")
        if confirm2.lower() == "y":
            print("✅ 审核通过, 继续执行...")
            final_result2 = agent.invoke(None, config=config2)
            print(f"最终结果: {final_result2['messages'][-1].content}")
        else:
            print("❌ 审核拒绝, 操作已取消。")
            reject_msg2 = AIMessage(
                content="⚠️ 该查询包含多表 JOIN 和聚合操作, 暂不允许执行。\n请联系 DBA 获取更高权限, 或拆分为多个简单查询。"
            )
            agent.update_state(config2, {"messages": [reject_msg2]})
            print(" → 已返回拒绝消息给用户。")
            updated_state2 = agent.get_state(config2)
            print(f" → AI 回复: {updated_state2.values['messages'][-1].content}")
    else:
        print("\nℹ️ Agent 没有调用工具, 直接回复了: ")
        print(ai_msg2.content)

    print("\n--- Human-in-the-loop 流程总结 ---")
    print(
        """
✅ 使用 LangGraph interrupt_before 实现真实的人工审核:
1. 创建 Agent 时指定 interrupt_before=["tools"]
2. 必须提供 checkpointer (保存暂停状态)
3. 第一次 invoke: Agent 执行到工具前暂停
4. 人工检查 Agent 准备调用的工具参数 (如 SQL 语句)
5. 审核通过: invoke(None, config) 继续执行
   审核拒绝: agent.update_state(config, {"messages": [AIMessage(...)]})
             返回拒绝消息给用户

优势:
- 完全控制 Agent 的每一步操作
- 可以在工具执行前拦截和修改参数
- 审核拒绝时可以返回友好的 AIMessage 给用户
- 适合高风险操作 (SQL、API 调用、文件操作)
"""
    )


def main(example_number: int):
    """运行指定的示例。"""
    print("=" * 60)
    print("第七课: SQL Agent 实战")
    print("=" * 60)

    examples = {1: example_1, 2: example_2, 3: example_3, 4: example_4, 5: example_5}

    if example_number in examples:
        examples[example_number]()
    else:
        print(f"错误: 示例编号 {example_number} 不存在")


if __name__ == "__main__":
    main(3)