# -*- coding: utf-8 -*-
"""
LangChain 环境搭建与快速入门
第一课：最简单的 Agent 与工具调用
"""

# 依赖安装
# pip install -U langchain langfuse python-dotenv

# .env 配置示例
# OPENAI_API_KEY=你的密钥
# OPENAI_API_BASE=https://ark.cn-beijing.volces.com/api/coding/v3
# LANGFUSE_SECRET_KEY="sk-lf-de3db398-f41f-46cd-9d39-08b1725e8dcd"
# LANGFUSE_PUBLIC_KEY="pk-lf-7d926b8b-a829-457e-9d87-1115d948c82a"
# LANGFUSE_BASE_URL="http://localhost:3000"

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from datetime import datetime

# 加载环境变量
load_dotenv()

# 初始化观测回调
langfuse_handler = CallbackHandler()


# ==============================================
# 示例 1：最简单的 Agent
# ==============================================
def example_1():
    print("\n==== 示例 1: 最简单的 Agent ====")

    model = init_chat_model(
        "doubao-seed-2.0-lite",
        model_provider="openai"
    )

    agent = create_agent(
        model=model,
        system_prompt="你是一个友好的助手，用简洁的方式回答问题。",
    )

    result = agent.invoke(
        {"messages": [("user", "你好，请介绍一下 LangChain")]},
        config={"callbacks": [langfuse_handler]},
    )

    print(f"AI 回复: {result['messages'][-1].content}")


# ==============================================
# 示例 2：带工具的 Agent - 天气查询
# ==============================================
def example_2():
    print("\n==== 示例 2: 带工具的 Agent ====")

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气情况。
        参数: city: 城市名称，例如 "北京"、"上海"
        返回: 天气情况的描述字符串
        """
        weather_db = {
            "北京": "晴朗, 25°C, 适宜出行",
            "上海": "多云, 22°C, 建议带伞",
            "广州": "阵雨, 28°C, 注意防雨",
            "深圳": "晴天, 30°C, 注意防晒",
        }
        return weather_db.get(city, f"抱歉，暂时无法查询 {city} 的天气")

    model = init_chat_model(
        "doubao-seed-2.0-lite",
        model_provider="openai"
    )

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是一个天气助手，帮助用户查询天气信息。",
    )

    result = agent.invoke(
        {"messages": [("user", "北京的天气怎么样?")]},
        config={"callbacks": [langfuse_handler]},
    )

    print(f"AI 回复: {result['messages'][-1].content}")


# ==============================================
# 示例 3：多工具协作 Agent
# ==============================================
def example_3():
    print("\n==== 示例 3: 多工具协作 Agent ====")

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式的结果。"""
        try:
            return f"计算结果: {eval(expression)}"
        except Exception as e:
            return f"计算错误: {e}"

    @tool
    def get_current_time() -> str:
        """获取当前时间。"""
        return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    @tool
    def convert_temperature(value: float, unit: str) -> str:
        """在摄氏度和华氏度之间转换。"""
        if unit.upper() == "C":
            return f"{value}°C = {value * 9 / 5 + 32:.1f}°F"
        elif unit.upper() == "F":
            return f"{value}°F = {(value - 32) * 5 / 9:.1f}°C"
        return "错误: 单位必须是 'C' 或 'F'"

    model = init_chat_model(
        "doubao-seed-2.0-lite",
        model_provider="openai"
    )

    agent = create_agent(
        model=model,
        tools=[calculator, get_current_time, convert_temperature],
        system_prompt="""你是一个实用助手，可以帮助用户:
- 进行数学计算
- 查询当前时间
- 转换温度单位
请根据用户的问题，选择合适的工具来帮忙。""",
    )

    # 测试 1: 数学计算
    print("\n--- 测试 1: 数学计算 ---")
    result1 = agent.invoke(
        {"messages": [("user", "计算 25 加 37 乘以 2 等于多少?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI 回复: {result1['messages'][-1].content}")

    # 测试 2: 时间查询
    print("\n--- 测试 2: 时间查询 ---")
    result2 = agent.invoke(
        {"messages": [("user", "现在几点了?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI 回复: {result2['messages'][-1].content}")

    # 测试 3: 温度转换
    print("\n--- 测试 3: 温度转换 ---")
    result3 = agent.invoke(
        {"messages": [("user", "100 华氏度是多少摄氏度?")]},
        config={"callbacks": [langfuse_handler]},
    )
    print(f"AI 回复: {result3['messages'][-1].content}")


# ==============================================
# 主程序入口
# ==============================================
if __name__ == "__main__":
    example_1()
    example_2()
    example_3()