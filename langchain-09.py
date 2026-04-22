"""
RAG 系统构建实战
==============
第六课 LangChain 核心应用 - RAG 系统构建
模块: 2.2
目标: 掌握检索增强生成(RAG)的完整流程
知识点:
--- 文档加载与分块
--- 嵌入模型与向量存储
--- 检索器设计
--- RAG Agent 实现
--- 真实 VectorStore 与模拟检索的区别
"""

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

load_dotenv()
langfuse_handler = CallbackHandler()

# 全局向量存储 (模拟)
vector_store = {}


def example_1():
    """
    示例 1: 文档加载与分块
    目标: 掌握文档处理的基础流程
    知识点:
    -- 创建文档对象
    -- 文本分块策略
    -- chunk_size 和 chunk_overlap 的影响
    """
    print("\n==== 示例 1: 文档加载与分块 =====")

    # 模拟文档内容
    raw_documents = [
        Document(
            page_content="""
LangChain 是一个用于构建大语言模型应用的开源框架。
它提供了丰富的组件和工具, 帮助开发者快速构建 AI 应用。
LangChain 的核心理念是将 LLM 与外部数据源和工具集成。
主要组件包括: 模型接口、提示词模板、索引组件、链式调用和代理。
            """,
            metadata={"source": "langchain_intro.txt", "section": "简介"},
        ),
        Document(
            page_content="""
RAG(检索增强生成)是一种结合信息检索和文本生成的技术。
它通过检索相关知识库, 然后将检索结果作为上下文提供给 LLM。
这样可以减少 LLM 的幻觉, 提高回答的准确性。
RAG 的典型应用场景包括: 智能客服、知识库问答、文档摘要等。
            """,
            metadata={"source": "rag_guide.txt", "section": "RAG 概述"},
        ),
    ]

    # 代码解释: Document 类
    # LangChain 中文档的标准表示, 用于包装文本及其元数据。
    # 属性:
    # page_content: str -- 文档的实际文本内容
    # metadata: dict -- 文档的附加信息 (来源、章节、作者等)
    #
    # 用途:
    # 在 RAG 流程中贯穿始终: 分块、嵌入、检索、返回结果
    # 都是 Document 对象, 保证元数据不丢失
    # ================================================

    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20  # 每个 chunk 的字符数 # chunk 之间的重叠字符数
    )
    # ================================================
    # RecursiveCharacterTextSplitter 参数详解
    # ================================================
    # separators=["\n\n", "\n", ".", "!", "?"] (默认)
    # 按顺序尝试的分隔符列表, 从左到右依次尝试:
    # 1. "\n\n" → 段落级分割
    # 2. "\n" → 行级分割
    # 3. ". " → 单词级分割
    # 4. "" → 字符级分割 (兜底)
    # 如果前面的分隔符不能产生合适大小的块,
    # 就尝试下一个。
    #
    # keep_separator=True (默认)
    # True:  分隔符保留在 chunk 中
    # False: 分隔符被移除
    # "start": 分隔符放在 chunk 开头
    # "end": 分隔符放在 chunk 结尾
    #
    # is_separator_regex=False (默认)
    # False: 分隔符系普通字符串 (字面匹配)
    # True: 分隔符是正则表达式
    # 例如: r"[.,!?\n]" - 匹配标点或换行
    # 例如: r"^\n#+" - 匹配 Markdown 标题
    #
    # chunk_size=4000 (默认)
    # 每个 chunk 的最大长度
    #
    # chunk_overlap=200 (默认)
    # 相邻 chunk 的重叠字符数
    # 作用: 防止关键信息被截断, 提高检索命中率
    #
    # 示例:
    # 文本: "A" * 50 + "B" * 50
    # chunk_size=30, chunk_overlap=10
    #
    # Chunk 1: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    # Chunk 2: ..........AAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBB
    #           ↑ 10 个 A 是重叠部分
    #
    # ================================================
    splits = text_splitter.split_documents(raw_documents)

    print(f"原文档: {len(raw_documents)}")
    print(f"分块后: {len(splits)}")
    print("\n分块结果:")
    for i, split in enumerate(splits):
        print(f"\nChunk {i+1}:")
        print(f"  内容: {split.page_content}")
        print(f"  元数据: {split.metadata}")


def example_2():
    """
    示例 2: 真实向量存储 - OpenAIEmbeddings + Chroma
    目标: 演示简洁的 RAG 流程
    知识点:
    -- 文档 → 嵌入 → 向量存储 → 检索 → Agent
    """
    print("\n==== 示例 2: 真实向量存储 - OpenAIEmbeddings + Chroma =====")

    # 1. 准备文档
    documents = [
        Document(
            page_content="Python 是一种流行的编程语言, 广泛用于 Web 开发和数据分析。",
            metadata={"topic": "编程"},
        ),
        Document(
            page_content="机器学习是 AI 的核心技术, 包括监督学习、无监督学习和强化学习。",
            metadata={"topic": "AI"},
        ),
        Document(
            page_content="Python 的 Flask 框架可以快速构建 REST API。",
            metadata={"topic": "编程"},
        ),
        Document(
            page_content="深度学习使用神经网络来处理复杂的数据模式。",
            metadata={"topic": "AI"},
        ),
        Document(
            page_content="数据库是存储和管理数据的系统, 常见的有 MySQL、PostgreSQL。",
            metadata={"topic": "数据库"},
        ),
    ]

    # 2. 创建嵌入模型
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False,
    )

    # 3. 创建 Chroma 向量存储 (文档 → 嵌入 → 存储 一步完成)
    vectorstore = Chroma.from_documents(documents, embeddings)

    # ================================================
    # similarity_search() 参数说明
    # ================================================
    # vectorstore.similarity_search(query, k=4, filter=None, where_document=None)
    #
    # query (str):      搜索关键词 (自动转为向量)
    # k (int):          返回最相关的 k 个文档 (默认 4)
    # filter (dict):    元数据过滤条件
    # 示例: {"topic": "编程"} → 只返回编程类
    # 示例: {"topic": "AI", "level": "入门"} → 多条件 AND
    #
    # where_document (dict): 文档内容过滤
    # 示例: {"$contains": "Python"} → 内容必须包含 Python
    #
    # 返回值: list[Document]
    # ================================================

    # 4. 基础相似度搜索
    print("\n---- 相似度搜索 (基础) ----")
    results = vectorstore.similarity_search("Python 编程", k=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. [{doc.metadata.get('topic', '未知')}] {doc.page_content}")

    # 5. filter 元数据过滤
    print("\n---- 元数据过滤 [filter] ----")
    filtered = vectorstore.similarity_search("Python", k=2, filter={"topic": "编程"})
    for doc in filtered:
        print(f"  [{doc.metadata['topic']}] {doc.page_content}")

    # 6. where_document 内容过滤
    print("\n---- 文档内容过滤 [where_document] ----")
    # $contains: 文档内容必须包含指定文本
    content_filtered = vectorstore.similarity_search(
        "编程",
        k=2,
        where_document={"$contains": "Flask"},
    )
    print("$contains 'Flask':")
    for doc in content_filtered:
        print(f"  [{doc.metadata['topic']}] {doc.page_content}")

    # $not_contains: 文档内容不能包含指定文本
    not_contains = vectorstore.similarity_search(
        "技术",
        k=2,
        where_document={"$not_contains": "数据库"},
    )
    print("$not_contains '数据库':")
    for doc in not_contains:
        print(f"  [{doc.metadata['topic']}] {doc.page_content}")

    # ================================================
    # as_retriever() 参数说明
    # ================================================
    # vectorstore.as_retriever(search_type="similarity", search_kwargs={...})
    # 将向量存储包装为 LangChain Retriever 对象。
    # Retriever 可以直接作为工具传给 Agent 使用。
    #
    # search_type (检索策略, 3 种):
    # "similarity" (默认)
    # → 普通向量相似度搜索, 返回最相关的 k 个文档
    # "mmr" (Maximal Marginal Relevance)
    # → 在相关性和多样性之间平衡, 避免返回相似内容
    # "similarity_score_threshold"
    # → 只返回相似度分数超过阈值的结果
    #
    # search_kwargs (按 search_type 不同而不同):
    # 通用参数:
    # "k": 2,                    # 返回文档数量 (默认 4)
    # "filter": {"topic": "编程"}, # 元数据过滤
    #
    # MMR 模式额外参数:
    # "fetch_k": 20,            # 初始候选数量 (默认 20)
    # "lambda_mult": 0.5,       # 多样性参数 (0-1)
    #                           # 0 = 最多样, 1 = 最相似, 默认 0.5
    #
    # score_threshold 模式额外参数:
    # "score_threshold": 0.5,   # 最低相似度阈值 (0-1)
    #                           # 只有相似度 > 0.5 的结果才返回
    # ================================================
    # 返回值: VectorStoreRetriever 对象
    # -- 有 invoke(query) 方法, 返回 list[Document]
    # -- 可直接作为 Agent 的工具使用
    # ================================================

    # 5. 创建检索器 (包装为 Retriever)
    # a. 基础模式 (similarity)
    print("\n---- 检索器 a: 基础模式 [similarity] ----")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("Python 编程")
    for doc in docs:
        print(f"  [{doc.metadata.get('topic')}] {doc.page_content}")

    # b. MMR 模式 (多样性)
    print("\n---- 检索器 b: MMR 模式 (fetch_k=5, lambda_mult=0.3) ----")
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.3},
    )
    docs = mmr_retriever.invoke("Python")
    for doc in docs:
        print(f"  [{doc.metadata.get('topic')}] {doc.page_content}")

    # c. score_threshold 模式 (最低相似度阈值)
    print("\n---- 检索器 c: score_threshold 模式 (阈值 0.6) ----")
    threshold_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 2, "score_threshold": 0.6},
    )
    docs = threshold_retriever.invoke("Python")
    if docs:
        for doc in docs:
            print(f"  [{doc.metadata.get('topic')}] {doc.page_content}")
    else:
        print("  无满足阈值的结果")

    # d. 带 filter 的检索器
    print("\n---- 检索器 d: 带元数据过滤 ----")
    filtered_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2, "filter": {"topic": "编程"}},
    )
    docs = filtered_retriever.invoke("AI 技术")
    print("  (即使搜索 'AI 技术', 也只返回编程类结果)")
    for doc in docs:
        print(f"  [{doc.metadata.get('topic')}] {doc.page_content}")

    # 6. 检索工具
    @tool
    def search_documents(query: str) -> str:
        """
        搜索相关文档。

        参数:
            query: 搜索关键词

        返回:
            相关文档内容
        """
        docs = retriever.invoke(query)

        if not docs:
            return f"未找到关于 '{query}' 的文档"

        output = f"找到 {len(docs)} 个相关文档:\n"
        for i, doc in enumerate(docs, 1):
            output += f"{i}. [{doc.metadata.get('topic', '未知')}] {doc.page_content}\n"
        return output

    # 7. 创建 Agent (集成检索工具)
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0)

    agent = create_agent(
        model=model,
        tools=[search_documents],
        system_prompt="""你是一个知识库助手, 基于检索到的文档回答问题。
如果文档中没有相关信息, 请说明"知识库中暂无相关信息"。""",
    )

    print("\n---- 测试 1: 语义检索 [Python 相关] ----")
    result1 = agent.invoke(
        {"messages": [("user", "Python 可以用来做什么?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n---- 测试 2: 语义检索 [AI 相关] ----")
    result2 = agent.invoke(
        {"messages": [("user", "机器学习包括哪些技术?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_3():
    """
    示例 3: Chroma 详解 - 持久化、多集合、常用方法
    目标: 深入学习 Chroma 向量数据库的核心功能
    知识点:
    -- 持久化存储
    -- 多集合管理
    -- 常用方法演示
    -- Chroma vs FAISS 对比
    """
    print("\n==== 示例 3: Chroma 详解 - 持久化、多集合、常用方法 ====")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False,
    )

    # ================================================
    # 一、持久化存储演示
    # ================================================
    print("\n--- 持久化存储演示 ---")
    print("""
Chroma 默认将数据存储在内存中, 程序退出后数据丢失。
要持久化保存, 只需传入 persist_directory 参数:
""")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        collection_name="my_collection",
        embedding_function=embeddings,
    )
    print("数据自动保存到 ./chroma_db/")
    print("""
磁盘存储结构:
./chroma_db/
├── chroma.sqlite3 ← SQLite 数据库 (元数据、向量)
└── *.bin         ← 向量索引文件
""")

    # 创建持久化的 Chroma 实例
    # 只需要传入 persist_directory 即可自动持久化
    persistent_store = Chroma(
        persist_directory="./chroma_db",
        collection_name="demo_persistent",
        embedding_function=embeddings,
    )

    # 添加文档并持久化
    demo_docs = [
        Document(page_content="持久化数据可以跨会话保存。", metadata={"type": "demo"}),
        Document(page_content="Chroma 使用 SQLite 存储元数据。", metadata={"type": "demo"}),
    ]
    persistent_store.add_documents(demo_docs)
    print(f"已添加 {len(demo_docs)} 条文档到持久化存储。")

    # 验证: 重新查询
    results = persistent_store.similarity_search("数据存储", k=1)
    print(f"查询结果: {results[0].page_content if results else '无结果'}")

    # ================================================
    # 二、多集合管理
    # ================================================
    print("""
--- 多集合管理 ---
一个项目可以有多个向量库, 共享同一个底层数据库,
但通过 collection_name 实现数据隔离。

典型场景:
- faq_store: FAQ 问答知识
- product_store: 产品文档
- support_store: 客服工单知识
""")

    faq_store = Chroma(
        persist_directory="./chroma_db",
        collection_name="faq",
        embedding_function=embeddings,
    )

    product_store = Chroma(
        persist_directory="./chroma_db",
        collection_name="products",
        embedding_function=embeddings,
    )

    # FAQ 集合添加文档
    faq_docs = [
        Document(page_content="如何重置密码? 请在设置页面点击「忘记密码」。", metadata={"category": "账户"}),
        Document(page_content="如何升级会员? 访问官网的订阅页面即可操作。", metadata={"category": "付费"}),
        Document(page_content="支持哪些支付方式? 支持支付宝、微信、信用卡。", metadata={"category": "付费"}),
    ]
    faq_store.add_documents(faq_docs)
    print(f"FAQ 集合: 已添加 {len(faq_docs)} 条文档")

    # 产品集合添加文档
    product_docs = [
        Document(page_content="Pro 版支持无限协作和高级分析功能。", metadata={"product": "Pro"}),
        Document(page_content="基础版免费使用, 适合个人开发者。", metadata={"product": "Basic"}),
    ]
    product_store.add_documents(product_docs)
    print(f"产品集合: 已添加 {len(product_docs)} 条文档")

    # 跨集合检索验证: 数据完全隔离
    print("\n--- FAQ 搜索「支付」 ---")
    faq_results = faq_store.similarity_search("支付", k=2)
    for doc in faq_results:
        print(f"  [{doc.metadata.get('category', '未知')}] {doc.page_content}")

    print("\n--- 产品搜索「免费版」 ---")
    product_results = product_store.similarity_search("免费版", k=2)
    for doc in product_results:
        print(f"  [{doc.metadata.get('product', '未知')}] {doc.page_content}")

    # ================================================
    # 三、常用方法演示
    # ================================================
    print("""
--- 常用方法演示 ---
Chroma 提供以下常用 API:
1. get()         → 查询全部数据
2. count()       → 查看文档总数
3. peek(n)       → 预览前 n 条
4. similarity_search(query, filter=...) → 带过滤的搜索
5. add_documents() → 添加新文档
6. delete(ids=...) → 删除指定文档
""")

    # 1. 查询全部
    print("1. get() - 查询全部数据")
    all_data = faq_store.get()
    print(f"  返回字段: {list(all_data.keys())}")

    # 2. 查看总数
    print("\n2. count() - 查看文档总数")
    count = faq_store._collection.count()
    print(f"  FAQ 集合共有 {count} 条文档")

    # 3. 预览
    print("\n3. peek() - 预览前 2 条")
    peek_result = faq_store._collection.peek(2)
    print(f"  预览文档数: {len(peek_result.get('documents', []))}")

    # 4. 带元数据过滤的搜索
    print("\n4. similarity_search with filter - 元数据过滤搜索")
    filtered_results = faq_store.similarity_search("方法", k=3, filter={"category": "付费"})
    print(f"  过滤条件: category=付费, 结果数: {len(filtered_results)}")
    for doc in filtered_results:
        print(f"  - {doc.page_content}")

    # 5. 添加新文档
    # ================================================
    # add_documents() 的 ids 参数说明
    # ================================================
    # ids 是文档在 Chroma 中的唯一标识符。
    #
    # 1. 不传 ids (默认):
    #    Chroma 会自动生成 UUID。
    #    即使内容完全相同, 也会当作新文档存储 (可能导致重复)。
    #
    # 2. 传入 ids (自定义):
    #    使用你提供的 ID 作为唯一标识。
    #    如果 ID 已存在, Chroma 会**覆盖**原内容 (Upsert 机制)。
    #    这是实现数据去重和更新旧数据的最有效方式。
    #
    # 3. 精确定位:
    #    后续可通过该 ID 快速删除 (delete(ids=[...])) 或更新文档。
    #
    # 示例:
    # faq_store.add_documents([new_doc])
    # # 未传 ids, 自动生成 UUID 并插入
    #
    # faq_store.add_documents([new_doc], ids=["custom_id_001"])
    # # 传入自定义 ID, 若 ID 已存在则覆盖, 若不存在则新增
    # ================================================
    print("\n5. add_documents() - 添加新文档")
    new_doc = Document(page_content="如何联系客服? 发送邮件到 support@example.com。", metadata={"category": "账户"})
    faq_store.add_documents([new_doc])  # 默认自动生成 UUID
    new_count = faq_store._collection.count()
    print(f"  添加后总数: {new_count} 条")

    # 6. 删除文档 (演示 - 创建临时集合)
    print("\n6. delete() - 删除文档")
    temp_store = Chroma(
        persist_directory="./chroma_db",
        collection_name="temp_demo",
        embedding_function=embeddings,
    )
    temp_store.add_documents(
        [
            Document(page_content="这是一条临时数据", metadata={"temp": True}),
        ]
    )
    print(f"  删除前: {temp_store._collection.count()} 条")
    temp_data = temp_store.get()
    if temp_data.get("ids"):
        temp_store.delete(ids=[temp_data["ids"][0]])
        print(f"  删除后: {temp_store._collection.count()} 条")

    # ================================================
    # 五、数据去重 (Deduplication)
    # ================================================
    print("""
--- 数据去重 (Deduplication) ---
Chroma 本身**不自动**根据内容去重。它通过 ID 识别文档:
- 如果添加相同 ID 的文档 → 会**覆盖**原有内容
- 如果添加不同 ID 但相同内容 → 会**重复存储**

去重策略有以下三种:
""")

    # 策略 1: 手动检查内容是否存在
    print("策略 1: 手动检查内容 (通过 get() 遍历对比) ")
    check_store = Chroma(
        persist_directory="./chroma_db1",
        collection_name="dedup_check",
        embedding_function=embeddings,
    )

    # 先添加一些数据
    check_store.add_documents(
        [
            Document(page_content="策略 1:Python 是一门编程语言", metadata={"source": "a"}),
        ]
    )

    # 添加前检查
    new_doc = Document(page_content="策略 1:Python 是一门编程语言", metadata={"source": "b"})
    existing = check_store.get()
    # get() 返回的 "documents" 是字符串列表, 直接对比内容
    is_dup = any(d == new_doc.page_content for d in existing.get("documents", []))
    if is_dup:
        print(f"  → 内容已存在, 跳过: {new_doc.page_content[:30]}...")
    else:
        check_store.add_documents([new_doc])
        print(f"  → 新文档, 已添加: {new_doc.page_content[:30]}...")
    print(f"  → 当前总数: {check_store._collection.count()} 条")

    # 策略 2: 使用内容哈希作为 ID
    print("\n策略 2: 内容哈希作为 ID (自动覆盖重复内容) ")
    import hashlib

    def make_doc_id(text: str) -> str:
        """根据内容生成唯一 ID (类似 Git 的原理) """
        return hashlib.md5(text.encode()).hexdigest()[:12]

    hash_store = Chroma(
        persist_directory="./chroma_db2",
        collection_name="dedup_hash",
        embedding_function=embeddings,
    )

    # 第一次添加
    doc1 = Document(page_content="策略 2:langchain是一个好框架")
    doc1_id = make_doc_id(doc1.page_content)
    hash_store.add_documents([doc1], ids=[doc1_id])
    print(f"  → 添加 doc1 (ID: {doc1_id}): {doc1.page_content[:30]}...")

    # 第二次添加相同内容 (ID 相同, 会覆盖)
    doc1_dup = Document(page_content="策略 2:langchain是一个好框架")
    doc1_dup_id = make_doc_id(doc1_dup.page_content)
    hash_store.add_documents([doc1_dup], ids=[doc1_dup_id])
    print(f"  → 添加重复 doc1_dup (ID: {doc1_dup_id}): 覆盖原数据")
    print(f"  → 当前总数: {hash_store._collection.count()} 条 (未增加)")

    # 添加不同内容
    doc2 = Document(page_content="策略 2:机器学习是 AI 的核心技术")
    doc2_id = make_doc_id(doc2.page_content)
    hash_store.add_documents([doc2], ids=[doc2_id])
    print(f"  → 添加 doc2 (ID: {doc2_id}): {doc2.page_content[:30]}...")
    print(f"  → 当前总数: {hash_store._collection.count()} 条")

    # 策略 3: 先删除后添加 (手动 Upsert)
    print("\n策略 3: 先删除后添加 (手动实现更新) ")
    upsert_store = Chroma(
        persist_directory="./chroma_db3",
        collection_name="dedup_upsert",
        embedding_function=embeddings,
    )

    # 1. 添加原始文档
    upsert_store.add_documents(
        [
            Document(page_content="原始版本内容", metadata={"version": 1}),
        ]
    )
    data = upsert_store.get()
    original_id = data["ids"][0]
    print(f"  → 原始内容: {data['documents'][0]} (ID: {original_id[:8]}...)")
    print(f"  → 总数: {upsert_store._collection.count()} 条")

    # 2. 删除旧数据
    upsert_store.delete(ids=[original_id])
    print(f"  → 执行 delete 后: {upsert_store._collection.count()} 条")

    # 3. 添加新数据
    upsert_store.add_documents(
        [
            Document(page_content="更新后的内容", metadata={"version": 2}),
        ]
    )
    print(f"  → 执行 add 后: {upsert_store._collection.count()} 条")

    # 4. 验证结果
    final_data = upsert_store.get()
    print(f"  → 最终内容: {final_data['documents'][0]}")

    # 总结
    print("""
去重策略总结:

策略               适用场景                          优缺点
------------------------------------------------------------------------
手动检查内容       数据量小, 需要精确控制             准确但性能差 (全量遍历)
内容哈希做 ID     中等数据量, 内容驱动去重           高效, 相同内容自动覆盖
upsert (先删后加) 已知 ID, 需要更新已有数据        适合增量更新场景

推荐: 使用「内容哈希做 ID」是最通用的去重方案。
""")

    # ================================================
    # 四、Chroma vs FAISS 对比
    # ================================================
    print("""
--- Chroma vs FAISS 对比 ---

对比项          Chroma                  FAISS
------------------------------------------------------------------------
持久化          ✅ 内置支持              ❌ 需手动 save/load
元数据过滤      ✅ 原生支持              ❌ 需自行实现
按 ID 查询      ✅ 内置                  ❌ 不支持
安装复杂度      简单 (pip install)       需编译 (某些平台复杂)
检索速度        快                      极快 (C++ 优化)
多集合管理      ✅ collection 隔离       ❌ 需创建多个实例
适用场景        开发/中小规模            大规模生产环境
开源协议        MIT                     MIT

总结:
- 开发和原型阶段: 推荐 Chroma (开箱即用)
- 大规模生产环境: 推荐 FAISS 或云端方案
- 需要元数据过滤/ID 查询: 选 Chroma
- 极致检索性能: 选 FAISS
""")


def example_4():
    """
    示例 4: RAG 对话系统 (集成 Chroma 与记忆)
    目标: 构建带记忆的 RAG 对话系统, 使用真实向量库
    知识点:
    -- 真实文档嵌入与存储 (Chroma)
    -- 多轮对话中的检索增强
    -- 上下文保持 (InMemorySaver)
    """
    print("\n==== 示例 4: RAG 对话系统 (真实 Chroma) ====")

    # 1. 准备真实的文档数据
    from langchain_core.documents import Document

    tech_docs = [
        Document(
            page_content="LangChain 是构建 LLM 应用的框架, 提供 Agent、Chain、Tool 等组件。",
            metadata={"topic": "LangChain"},
        ),
        Document(page_content="Agent 是能自主决策的 AI 程序, 结合 LLM 和工具完成任务。", metadata={"topic": "Agent"}),
        Document(page_content="RAG 是检索增强生成技术, 先检索相关知识再生成回答。", metadata={"topic": "RAG"}),
        Document(page_content="Tool 是 Agent 可调用的函数, 扩展 Agent 的能力边界。", metadata={"topic": "Tool"}),
        Document(page_content="Chain 是将多个步骤串联执行的 LLM 工作流。", metadata={"topic": "Chain"}),
    ]

    # 2. 初始化 Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False,
    )

    # 3. 创建 Chroma 向量库 (模拟持久化, 实际使用内存)
    vectorstore = Chroma.from_documents(tech_docs, embeddings, collection_name="tech_kb_demo")

    # 4. 定义检索工具 (真实检索)
    @tool
    def retrieve_context(query: str) -> str:
        """
        从技术知识库检索上下文。

        参数:
            query: 查询关键词

        返回:
            检索到的相关文档内容
        """
        results = vectorstore.similarity_search(query, k=2)
        if not results:
            return "未找到相关信息"
        # 格式化输出以便模型阅读
        return "\n".join([f"[{doc.metadata.get('topic')}] {doc.page_content}" for doc in results])

    # 5. 初始化模型
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    # ================================================
    # 6. InMemorySaver 详解: Agent 的"记忆"系统
    # ================================================
    # InMemorySaver 是 LangGraph 提供的一种内存级检查点机制 (Checkpointer)。
    # 它的核心作用是: 在多次 invoke() 调用之间, 保持对话状态的连续性。
    #
    # 工作原理:
    # 1. 状态保存 (Save State):
    #    每次 Agent 完成任务后, InMemorySaver 会将当前的状态 (包括所有消息历史)
    #    保存在内存字典中, 并使用 thread_id 作为 Key。
    #
    # 2. 状态加载 (Load State):
    #    下一次调用 invoke() 时, 如果提供了相同的 thread_id,
    #    InMemorySaver 会自动从内存中加载之前的状态。
    #    这意味着 Agent 可以"看到"之前的对话历史, 从而实现多轮对话。
    #
    # 为什么 RAG 需要它?
    # - 在 RAG 系统中, 用户经常基于上一步的检索结果进行追问 (例如: "它还有什么功能?")。
    # - 如果没有 InMemorySaver, 每次调用都是全新的, Agent 不知道"它"指的是什么。
    # - 有了它, Agent 就能结合历史记录 (Context) 和新检索到的知识 (Knowledge) 回答。
    #
    # 局限性:
    # - 数据存储在 Python 进程的内存中, 程序重启后数据丢失。
    # - 生产环境建议替换为 PostgresSaver 或 RedisSaver。
    # ================================================

    # 6. 创建带记忆的 RAG Agent
    # checkpointer 参数告诉 Agent 使用 InMemorySaver 来管理状态
    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt="""你是一个技术文档助手。
    使用检索工具获取准确信息后回答用户问题。
    如果检索不到相关信息, 请如实告知。""",
        checkpointer=InMemorySaver(),
    )

    # 7. 多轮对话测试
    # thread_id 用于标识特定的对话会话
    thread_config = {"configurable": {"thread_id": "rag_chat_real_1"}}

    print("\n--- 第一轮对话: 询问基础概念 ---")
    result1 = agent.invoke(
        {"messages": [("user", "什么是 LangChain?")]},
        config={**thread_config, "callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 第二轮对话: 基于上下文的追问 ---")
    # 这里用户问"它", 模型需要通过 InMemorySaver 知道指的是 LangChain
    result2 = agent.invoke(
        {"messages": [("user", "它的 Agent 是什么?")]},
        config=thread_config,
    )
    print(f"AI: {result2['messages'][-1].content}")


def example_5():
    """
    示例 5: 产品分析系统 (真实 Chroma 检索)
    目标: 创建基于向量检索的产品问答 Agent
    知识点:
    -- 产品文档向量化
    -- 语义检索 (不仅仅是关键词匹配)
    -- Agent 规划与多步检索
    """
    print("\n==== 示例 5: 产品分析系统 (真实 Chroma) ====")

    # 1. 准备产品文档
    product_docs = [
        Document(
            page_content="iPhone 15 搭载 A16 Bionic 芯片, 6.1 英寸屏幕, 4800 万像素相机, 售价 5999 元起。",
            metadata={"product": "iPhone"},
        ),
        Document(
            page_content="iPhone 15 Pro 搭载 A17 Pro 芯片, 钛金属机身, 5 倍长焦, 售价 7999 元起。",
            metadata={"product": "iPhone Pro"},
        ),
        Document(
            page_content="MacBook Air M3 搭载 M3 芯片, 18 小时续航, 无风扇设计, 售价 8999 元起。",
            metadata={"product": "MacBook"},
        ),
        Document(
            page_content="MacBook Pro M3 搭载 M3 Pro 芯片, 18GB 内存, XDR 屏幕, 售价 12999 元起。",
            metadata={"product": "MacBook Pro"},
        ),
        Document(
            page_content="iPad Air M2 搭载 M2 芯片, 11 英寸 Liquid Retina 屏幕, 支持 Apple Pencil Pro。",
            metadata={"product": "iPad"},
        ),
    ]

    # 2. 初始化 Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False,
    )

    # 3. 创建产品向量库
    product_store = Chroma.from_documents(product_docs, embeddings, collection_name="products_demo")

    # 4. 定义检索工具
    # 为了代码的通用性和稳定性, 我们手动将 Retriever 封装为 Tool。
    # 这样能更好地控制返回格式, 且不依赖特定版本。
    #
    # 首先创建检索器 (Retriever)
    retriever = product_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"k": 2, "score_threshold": 0.5}
    )  # 返回最相关的 2 个文档

    # 将 Retriever 包装为 Tool
    @tool
    def search_product_info(query: str) -> str:
        """
        查询苹果产品的规格、价格和特性信息。

        参数:
            query: 用户的问题 (如 "iPhone 多少钱", "MacBook 的芯片")

        返回:
            相关产品文档内容
        """
        # 使用 retriever.invoke 进行语义检索
        docs = retriever.invoke(query)
        if not docs:
            return "未找到相关产品。"
        return "\n".join([f"- {doc.page_content}" for doc in docs])

    # 5. 初始化模型
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai", temperature=0.3)

    # 6. 创建产品顾问 Agent
    agent = create_agent(
        model=model,
        tools=[search_product_info],
        system_prompt="""你是一个专业的苹果产品顾问。
    根据用户的问题, 使用 search_product_info 工具查询准确的产品参数和价格。
    基于检索到的信息回答用户, 不要编造数据。""",
    )

    # 7. 测试
    print("\n--- 测试 1: 价格查询 ---")
    result1 = agent.invoke(
        {"messages": [("user", "iPhone 15 Pro 的价格是多少?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n--- 测试 2: 跨产品对比 (语义理解) ---")
    # 测试语义匹配: 用户问"笔记本", 系统应该检索到 "MacBook"
    result2 = agent.invoke(
        {"messages": [("user", "推荐一款适合办公的轻薄笔记本, 价格 9000 左右")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")


def main(example_number: int):
    """运行指定的示例。"""
    print("=" * 60)
    print("第六课 RAG 系统构建实战")
    print("=" * 60)

    examples = {1: example_1, 2: example_2, 3: example_3, 4: example_4, 5: example_5}

    if example_number in examples:
        examples[example_number]()
    else:
        print(f"错误: 示例编号 {example_number} 不存在")


if __name__ == "__main__":
    main(5)
