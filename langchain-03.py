"""
第3课: 消息 (Messages) 与对话历史 - 完整演示代码
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain.chat_models import init_chat_model


# ==================================================
# 示例 1: 四种消息类型
# ==================================================
print("=" * 60)
print("示例 1: 四种消息类型")
print("=" * 60)

# 1. SystemMessage - 系统消息
print("\n--- SystemMessage ---")
system_msg = SystemMessage("你是一个 helpful 的编程助手，总是提供代码示例。")
print(f"完整: {system_msg}")
print(f"类型: {type(system_msg)}")
print(f"内容: {system_msg.content}")

# 2. HumanMessage - 用户消息
print("\n--- HumanMessage ---")
human_msg = HumanMessage("如何写一个 Python 快速排序? ")
print(f"完整: {human_msg}")
print(f"类型: {type(human_msg)}")
print(f"内容: {human_msg.content}")

# 带元数据的 HumanMessage
human_msg_with_meta = HumanMessage(
    content="你好! ",
    name="alice",
    id="msg_001"
)
print(f"\n带元数据的消息:")
print(f"  name: {human_msg_with_meta.name}")
print(f"  id: {human_msg_with_meta.id}")

# 3. AIMessage - AI 消息
print("\n--- AIMessage ---")
ai_msg = AIMessage("这是一个快速排序的示例...")
print(f"完整: {ai_msg}")
print(f"类型: {type(ai_msg)}")
print(f"内容: {ai_msg.content}")

# 4. ToolMessage - 工具消息
print("\n--- ToolMessage ---")
tool_msg = ToolMessage(
    content="执行结果: 排序完成",
    tool_call_id="call_123",
    name="sort_function"
)
print(f"完整: {tool_msg}")
print(f"类型: {type(tool_msg)}")
print(f"内容: {tool_msg.content}")
print(f"tool_call_id: {tool_msg.tool_call_id}")
print("\n")


# ==================================================
# 示例 2: 对话历史管理
# ==================================================
print("=" * 60)
print("示例 2: 对话历史管理")
print("=" * 60)

try:
    # 初始化模型
    model = init_chat_model("qwen3.5-plus", model_provider="openai")

    # 初始对话历史
    messages = [
        SystemMessage("你是一个 friendly 的助手，记住用户说的话。"),
        HumanMessage("我叫小明，今年 25 岁。"),
    ]

    print("\n--- 第一轮对话 ---")
    print(f"消息数: {len(messages)}")
    response1 = model.invoke(messages)
    print(f"AI 回复: {response1.content[:50]}...")

    # 把 AI 回复加入历史
    messages.append(response1)

    # 第二轮对话 - 问一个需要记忆的问题
    messages.append(HumanMessage("我叫什么名字? 今年多大? "))

    print("\n--- 第二轮对话 ---")
    print(f"消息数: {len(messages)}")
    response2 = model.invoke(messages)
    print(f"AI 回复: {response2.content}")

    print("\n✅ 模型记住了之前的对话! ")
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 3: 消息内容块 (多模态)
# ==================================================
print("=" * 60)
print("示例 3: 消息内容块 (多模态)")
print("=" * 60)

# 文本 + 图片的多模态消息
img_url = "https://pic.rmb.bdstatic.com/bjh/news/c95f5cf4c1ac5e0870fae0021d92501f.jpeg"

# 方式 1: 使用 content_blocks (推荐)
human_msg_multimodal = HumanMessage(content_blocks=[
    {"type": "text", "text": "请描述这张图片"},
    {"type": "image", "url": img_url},
])

print("使用 content_blocks:")
for block in human_msg_multimodal.content_blocks:
    print(f" - {block['type']}: {block.get('text', block.get('url', ''))}")

try:
    model = init_chat_model("qwen3.5-plus", model_provider="openai")
    collected_text = ""
    print("\n模型回复:")
    for chunk in model.stream(input=[human_msg_multimodal]):
        if isinstance(chunk.content, str):
            collected_text += chunk.content
            print(chunk.content, end="", flush=True)
    print()
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 4: Token 使用统计
# ==================================================
print("=" * 60)
print("示例 4: Token 使用统计")
print("=" * 60)

try:
    model = init_chat_model("qwen3.5-plus", model_provider="openai")
    messages = [
        SystemMessage("你是一个 helpful 的助手。"),
        HumanMessage("写一首关于秋天的短诗。"),
    ]
    response = model.invoke(messages)

    print("\n--- Token 使用统计 ---")
    if response.usage_metadata:
        usage = response.usage_metadata
        print(f"usage_metadata:\n{usage}\n")
        print(f"输入 Token: {usage.get('input_tokens', 'N/A')}")
        print(f"输出 Token: {usage.get('output_tokens', 'N/A')}")
        print(f"总共 Token: {usage.get('total_tokens', 'N/A')}")

        if 'input_token_details' in usage:
            print(f"\n输入详情: {usage['input_token_details']}")
        if 'output_token_details' in usage:
            print(f"输出详情: {usage['output_token_details']}")
    else:
        print("该模型不返回 usage_metadata")

    print(f"\nAI 回复:\n{response.content}")
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 5: 字典格式消息
# ==================================================
print("=" * 60)
print("示例 5: 字典格式消息")
print("=" * 60)

# 也可以直接用字典格式, 类似 OpenAI API
print("\n--- 字典格式消息 ---")
messages_dict = [
    {"role": "system", "content": "你是一个 helpful 的助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好! 有什么可以帮助你的? "},
    {"role": "user", "content": "刚才我们说了什么? "},
]

# 混合使用 Message 对象和字典
messages_dict2 = [
    SystemMessage("你是一个 helpful 的助手"),
    HumanMessage("你好"),
    AIMessage("你好! 有什么可以帮助你的? "),
    HumanMessage("我们说了什么了啊? "),
]

print("消息格式:")
for msg in messages_dict:
    print(f"  {msg['role']}: {msg['content'][:30]}...")

try:
    model = init_chat_model("qwen3.5-plus", model_provider="openai")
    response = model.invoke(messages_dict2)
    print(f"\nAI 回复: {response.content}")
except Exception as e:
    print(f"\n提示: 请先设置 API Key。错误: {e}")
print("\n")


# ==================================================
# 示例 6: 完整对话机器人
# ==================================================
print("=" * 60)
print("示例 6: 完整对话机器人")
print("=" * 60)

class ChatBot:
    """简单的对话机器人类"""
    def __init__(self, system_prompt="你是一个 helpful 的助手。"):
        self.messages = [SystemMessage(system_prompt)]
        try:
            self.model = init_chat_model("qwen3.5-plus", model_provider="openai")
        except Exception as e:
            self.model = None
            print(f"警告: 模型初始化失败: {e}")

    def chat(self, user_input):
        """发送消息并获取回复"""
        if not self.model:
            return "错误: 模型未初始化"

        # 添加用户消息
        self.messages.append(HumanMessage(user_input))

        # 调用模型 (流式输出)
        response = ""
        print("AI: ", end="", flush=True)
        for chunk in self.model.stream(self.messages):
            if isinstance(chunk.content, str) and chunk.content.strip() != "":
                response += chunk.content
                print(chunk.content, end="", flush=True)
        print()

        # 添加 AI 回复到历史
        self.messages.append(AIMessage(response))
        return response

    def get_history(self):
        """获取对话历史"""
        return self.messages

    def clear_history(self):
        """清空历史 (保留系统消息)"""
        self.messages = self.messages[:1]  # 只保留第一个系统消息

# 演示使用
print("\n--- 对话机器人演示 ---")
try:
    bot = ChatBot("你是一个喜欢用表情符号的 friendly 助手。")
    print("Bot 已启动! 输入 'quit' 退出, 'clear' 清空历史\n")

    # 模拟几轮对话
    test_inputs = [
        "你好!",
        "我喜欢编程",
        "我刚才说我喜欢什么? ",
    ]

    for user_input in test_inputs:
        print(f"\n👤 用户: {user_input}")
        bot.chat(user_input)

    print(f"\n对话历史长度: {len(bot.get_history())} 条消息")
except Exception as e:
    print(f"\n提示: 错误: {e}")

print("\n" + "=" * 60)
print("第3课演示代码完成!")
print("=" * 60)