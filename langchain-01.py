"""
第1课: LangChain 生态系统概览 - 完整演示代码
"""
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 加载环境变量（.env文件中配置OPENAI_API_KEY、OPENAI_API_BASE等）
load_dotenv()


# ==================================================
# 示例 1: 最简单的 Hello World
# ==================================================
print("=" * 60)
print("示例 1: 最简单的 Hello World")
print("=" * 60)

try:
    # 1. 初始化模型（使用OpenAI兼容接口）
    model = init_chat_model(
        "doubao-seed-2.0-lite",
        model_provider="openai"
    )

    # 2. 调用模型
    response = model.invoke("你好, LangChain!")

    # 3. 输出结果
    print("\n模型回复: ")
    print(response.content)
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 2: 多提供商切换演示
# ==================================================
print("=" * 60)
print("示例 2: 多提供商切换演示")
print("=" * 60)

def chat_with_model(model_name, question):
    """通用的模型聊天函数"""
    print(f"\n{'='*50}")
    print(f"使用模型: {model_name}")
    print(f"问题: {question}")
    print(f"{'='*50}")

    try:
        model = init_chat_model(model_name, model_provider="openai")
        response = model.invoke(question)
        print(f"回答: {response.content[:100]}...")
    except Exception as e:
        print(f"\n模型调用失败: {e}")

# 测试问题
question = "用一句话解释什么是 AI"

# 支持的模型列表（OpenAI兼容接口）
models = [
    "doubao-seed-code",
    "glm-4.7",
    "kimi-k2.5",
    "doubao-seed-2.0-lite"
]

# 循环调用不同模型
for model_name in models:
    chat_with_model(model_name, question)
print("\n")


# ==================================================
# 示例 3: 流式输出
# ==================================================
print("=" * 60)
print("示例 3: 流式输出")
print("=" * 60)

try:
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai")
    print("\n正在生成内容...\n")

    prompt = "写一首关于 AI 的短诗（4句以内）"

    # 记录开始时间
    start_time = time.time()

    # 流式输出
    collected_text = ""
    for chunk in model.stream(prompt):
        collected_text += chunk.text
        print(chunk.text, end="", flush=True)

    # 计算耗时
    elapsed = time.time() - start_time
    print(f"\n\n{'='*50}")
    print(f"生成完成! 耗时: {elapsed:.2f} 秒")
    print(f"总长度: {len(collected_text)} 字符")
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 4: 批处理多个请求
# ==================================================
print("=" * 60)
print("示例 4: 批处理多个请求")
print("=" * 60)

try:
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai")

    # 准备多个问题
    questions = [
        "什么是机器学习？（一句话）",
        "什么是深度学习？（一句话）",
        "什么是强化学习？（一句话）"
    ]

    print("\n=== 性能对比实验 ===")

    # 方式1：顺序调用（串行）
    print("1. 顺序调用...")
    start = time.time()
    responses_seq = []
    for q in questions:
        resp = model.invoke(q)
        responses_seq.append(resp)
    time_seq = time.time() - start
    print(f"   耗时: {time_seq:.2f} 秒\n")

    # 方式2：批量调用（并行）
    print("2. 批量调用...")
    start = time.time()
    responses_batch = model.batch(questions)
    time_batch = time.time() - start
    print(f"   耗时: {time_batch:.2f} 秒\n")

    # 结果对比
    print("="*50)
    print(f"顺序调用: {time_seq:.2f} 秒")
    print(f"批量调用: {time_batch:.2f} 秒")
    if time_batch > 0:
        print(f"加速比: {time_seq / time_batch:.1f}x")

    # 验证结果一致（前30字符）
    print("\n验证结果（前30字符）:")
    for i, (r1, r2) in enumerate(zip(responses_seq, responses_batch)):
        print(f"{i+1}. 顺序: {r1.content[:30]}...")
        print(f"   批量: {r2.content[:30]}...")
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 5: 产品选择辅助工具
# ==================================================
print("=" * 60)
print("示例 5: 产品选择辅助工具")
print("=" * 60)

def choose_product():
    """交互式LangChain产品选择助手"""
    print("\n" + "=" * 60)
    print("LangChain 产品选择助手")
    print("=" * 60)

    print("\n请回答以下问题 (y/n):")
    try:
        # 获取用户输入
        q1 = input("1. 你是否需要快速上手 (y/n)? ").lower() == 'y'
        q2 = input("2. 你是否需要底层控制 (y/n)? ").lower() == 'y'
        q3 = input("3. 任务是否复杂多步骤 (y/n)? ").lower() == 'y'
    except KeyboardInterrupt:
        print("\n\n已取消")
        return

    print("\n" + "=" * 60)
    print("推荐结果:")
    print("=" * 60)

    # 根据需求推荐工具
    if q3:
        print("\n🤖 推荐: Deep Agents SDK")
        print("   原因: 复杂多步骤任务需要规划和子代理能力")
        print("\n   核心特性:")
        print("   ✅ 规划和任务分解")
        print("   ✅ 虚拟文件系统")
        print("   ✅ 子代理生成")
        print("   ✅ 长期记忆")
    elif q2:
        print("\n🔧 推荐: LangGraph 编排框架")
        print("   原因: 需要底层控制和精细编排")
        print("\n   核心特性:")
        print("   ✅ 持久执行")
        print("   ✅ 人工介入")
        print("   ✅ 状态管理")
        print("   ✅ 生产级部署")
    elif q1:
        print("\n🚀 推荐: LangChain 框架")
        print("   原因: 快速上手, 简单直接")
        print("\n   核心特性:")
        print("   ✅ 标准模型接口")
        print("   ✅ 易用的代理")
        print("   ✅ 模型集成")
        print("   ✅ LangSmith 调试")
    else:
        print("\n💡 需要更多信息才能推荐")
        print("   建议从 LangChain 开始尝试!")

    print("\n" + "=" * 60)

# 运行选择工具
choose_product()

print("\n" + "=" * 60)
print("第1课演示代码完成!")
print("=" * 60)