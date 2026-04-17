"""
第2课: LLM 模型基础与调用 - 完整演示代码
"""
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from typing import Literal

# 加载环境变量（配置API Key）
load_dotenv()


# ==================================================
# 示例 1: 基础模型调用
# ==================================================
print("=" * 60)
print("示例 1: 基础模型调用")
print("=" * 60)

try:
    # 1. 初始化模型
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai")

    # 2. 调用模型
    response = model.invoke("你好, LangChain!")

    # 3. 输出结果
    print("\n模型回复: ")
    print(response.content)
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 2: temperature 参数对比
# ==================================================
print("=" * 60)
print("示例 2: temperature 参数对比")
print("=" * 60)

try:
    prompt = "给咖啡店起一个有创意的名字"
    print(f"\n提示词: {prompt}\n")

    # 测试不同temperature
    for temp in [0.0, 0.3, 0.6, 0.9]:
        print(f"\n--- temperature = {temp} ---")
        # 为每个温度初始化模型
        model_with_temp = init_chat_model(
            "doubao-seed-2.0-lite",
            model_provider="openai",
            temperature=temp
        )
        # 多次调用看多样性（仅显示2次节省时间）
        for i in range(2):
            response = model_with_temp.invoke(prompt)
            print(f"  第{i+1}次: {response.content.strip()}")
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 3: 流式输出完整演示
# ==================================================
print("=" * 60)
print("示例 3: 流式输出完整演示")
print("=" * 60)

try:
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai")
    print("\n正在生成内容...\n")

    prompt = """请写一篇关于人工智能的短文，包含以下部分：
1. AI 的定义
2. AI 的历史
3. AI 的应用
4. 未来展望
请控制在 200 字以内。"""

    # 记录开始时间
    start_time = time.time()

    # 流式输出
    collected_text = ""
    for chunk in model.stream(prompt):
        collected_text += chunk.content
        print(chunk.content, end="", flush=True)

    # 计算耗时
    elapsed = time.time() - start_time
    print(f"\n\n{'='*50}")
    print(f"生成完成! 耗时: {elapsed:.2f} 秒")
    print(f"总长度: {len(collected_text)} 字符")
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 4: 模型选择器工具
# ==================================================
print("=" * 60)
print("示例 4: 模型选择器工具")
print("=" * 60)

TaskType = Literal["coding", "creative", "analysis", "chat"]

@dataclass
class ModelConfig:
    """模型配置类，封装不同任务的模型参数"""
    model_name: str
    temperature: float
    max_tokens: int
    description: str

# 预定义不同任务的配置
TASK_CONFIGS = {
    "coding": ModelConfig(
        model_name="doubao-seed-2.0-lite",
        temperature=0.0,
        max_tokens=2000,
        description="代码生成: 低温度、高确定性"
    ),
    "creative": ModelConfig(
        model_name="doubao-seed-2.0-lite",
        temperature=0.9,
        max_tokens=1000,
        description="创意写作: 高温度、多样化"
    ),
    "analysis": ModelConfig(
        model_name="doubao-seed-2.0-lite",
        temperature=0.2,
        max_tokens=1500,
        description="数据分析: 中低温度、准确"
    ),
    "chat": ModelConfig(
        model_name="doubao-seed-2.0-lite",
        temperature=0.7,
        max_tokens=800,
        description="日常聊天: 中等温度、自然"
    )
}

def get_model_for_task(task_type: TaskType):
    """根据任务类型获取配置好的模型"""
    config = TASK_CONFIGS.get(task_type, TASK_CONFIGS["chat"])
    print(f"\n🔧 任务类型: {task_type}")
    print(f"📝 配置: {config.description}")
    print(f"⚙️  temperature={config.temperature}, max_tokens={config.max_tokens}")
    
    return init_chat_model(
        model=config.model_name,
        model_provider="openai",
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

# 使用示例
print("\n=== 模型选择器演示 ===\n")
try:
    # 1. 代码生成任务
    print("--- 任务 1: 代码生成 ---")
    model_coding = get_model_for_task("coding")
    resp = model_coding.invoke("写一个 Python 快速排序函数")
    print(f"结果预览: {resp.content[:80]}...\n")

    # 2. 创意写作任务
    print("--- 任务 2: 创意写作 ---")
    model_creative = get_model_for_task("creative")
    resp = model_creative.invoke("给科幻小说起个名字")
    print(f"结果: {resp.content}\n")

    # 3. 数据分析任务
    print("--- 任务 3: 数据分析 ---")
    model_analysis = get_model_for_task("analysis")
    resp = model_analysis.invoke("分析这个数据: [1,3,5,7,9]")
    print(f"结果预览: {resp.content[:80]}...\n")
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")

print("\n" + "=" * 60)
print("第2课演示代码完成!")
print("=" * 60)