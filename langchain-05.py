## tools - 第5课演示代码

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
import json
from pydantic import BaseModel, Field
from typing import Literal

# ==================================================
# 示例 1: 基础工具创建
# ==================================================
print("=" * 60)
print("示例 1: 基础工具创建")
print("=" * 60)

# 最简单的工具
@tool
def search_database(query: str, limit: int = 10) -> str:
    """在客户数据库中搜索匹配的记录。

    Args:
        query: 要搜索的关键词
        limit: 返回结果的最大数量
    """
    # 模拟数据库查询
    results = [
        {"id": 1, "name": "张三", "email": "zhang@example.com"},
        {"id": 2, "name": "李四", "email": "li@example.com"},
    ]
    return f"找到 {min(limit, len(results))} 条关于 '{query}' 的结果: {results}"

print("\n--- 工具信息 ---")
print(f"工具名: {search_database.name}")
print(f"工具描述: {search_database.description}")
print(f"工具参数 Schema:")
print(json.dumps(search_database.args_schema.model_json_schema(), indent=2, ensure_ascii=False))

# 调用工具 (和普通函数一样)
print("\n--- 调用工具 ---")
result = search_database.invoke({"query": "客户", "limit": 5})
print(f"结果: {result}")

print("\n")

# ==================================================
# 示例 2: 自定义工具名和描述
# ==================================================
print("=" * 60)
print("示例 2: 自定义工具名和描述")
print("=" * 60)

# 自定义工具名
@tool("web_search")
def search(query: str) -> str:
    """搜索网络获取信息。"""
    return f"关于 '{query}' 的网络搜索结果: ..."

print("\n--- 自定义工具名 ---")
print(f"函数名: {search}")
print(f"工具名: {search.name}")

# 自定义工具描述
@tool(
    "calculator",
    description="执行数学计算。任何数学问题（加减乘除）都用这个工具。"
)
def calc(expression: str) -> str:
    """计算数学表达式。"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

print("\n--- 自定义描述 ---")
print(f"工具名: {calc.name}")
print(f"描述: {calc.description}")

# 测试计算器
print("\n--- 测试计算器 ---")
print(calc.invoke({"expression": "2 + 2"}))
print(calc.invoke({"expression": "10 * 5"}))

print("\n")

# ==================================================
# 示例 3: Pydantic Schema 高级定义
# ==================================================
print("=" * 60)
print("示例 3: Pydantic Schema 高级定义")
print("=" * 60)

class WeatherInput(BaseModel):
    """天气查询的输入。"""
    location: str = Field(description="城市名或坐标, 如 '北京'、'上海'")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位: 摄氏度(celsius)或华氏度(fahrenheit)"
    )
    include_forecast: bool = Field(
        default=False,
        description="是否包含未来5天的天气预报"
    )

@tool(args_schema=WeatherInput)
def get_weather(
    location: str,
    units: str = "celsius",
    include_forecast: bool = False
) -> str:
    """获取当前天气和可选预报。"""
    # 模拟天气数据
    temp_c = 22
    temp = temp_c if units == "celsius" else (temp_c * 9/5 + 32)

    result = f"{location} 的天气: 晴天, {temp}°{units[0].upper()}"

    if include_forecast:
        result += "\n未来5天预报: 晴天、晴天、多云、晴天、晴天"

    return result

print("\n--- WeatherInput Schema ---")
print(json.dumps(WeatherInput.model_json_schema(), indent=2, ensure_ascii=False))

print("\n--- 调用天气工具 ---")
print(get_weather.invoke({
    "location": "北京",
    "units": "celsius",
    "include_forecast": True
}))

print("\n")

# ==================================================
# 示例 5: 实际案例 - 网络搜索工具
# ==================================================
print("=" * 60)
print("示例 5: 实际案例 - 网络搜索工具")
print("=" * 60)

@tool("web_search", description="搜索网络获取最新信息。需要最新新闻或事实时使用。")
def web_search(query: str) -> str:
    """搜索网络（模拟）。"""
    # 这里应该调用真实的搜索 API
    # 比如 SerpAPI、Google Search API 等
    mock_results = {
        "AI": "AI 最新进展: 大模型在多个领域取得突破...",
        "天气": "今日天气: 全国大部分地区晴天, 气温适宜...",
        "新闻": "今日要闻: 科技公司发布新产品...",
    }
    return mock_results.get(query, f"关于 '{query}' 的搜索结果: ...")

print("\n--- 网络搜索工具 ---")
print(f"工具名: {web_search.name}")
print(f"描述: {web_search.description}")
print("\n测试搜索:")
print(web_search.invoke({"query": "AI"}))

print("\n")

# ==================================================
# 示例 6: 实际案例 - 计算器工具
# ==================================================
print("=" * 60)
print("示例 6: 实际案例 - 计算器工具")
print("=" * 60)

@tool("calculator", description="执行数学计算。加减乘除、复杂表达式都用这个。")
def calculator(expression: str) -> str:
    """安全的计算器工具。"""
    try:
        # 简单的安全检查 (生产环境应该用更严格的)
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含非法字符"

        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except ZeroDivisionError:
        return "错误: 不能除以零"
    except Exception as e:
        return f"计算错误: {e}"

print("\n--- 计算器工具 ---")
test_cases = [
    "2 + 2",
    "10 * 5",
    "(10 + 5) * 3",
    "10 / 0",
]

for expr in test_cases:
    print(f"{expr} → {calculator.invoke({'expression': expr})}")

print("\n")

# ==================================================
# 示例 7: 多个工具组合
# ==================================================
print("=" * 60)
print("示例 7: 多个工具组合")
print("=" * 60)

@tool
def web_search(query: str) -> str:
    """搜索网络获取信息。"""
    return f"关于 '{query}' 的网络搜索结果: ..."

# 自定义工具描述
@tool(
    description="执行数学计算。任何数学问题（加减乘除）都用这个工具。"
)
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_weather(city: str) -> dict:
    """获取城市天气（返回结构化数据）。"""
    return {
        "city": city,
        "temperature": 22,
        "units": "celsius",
        "conditions": "晴天"
    }

# 定义多个工具
tools = [
    web_search,
    calculator,
    get_weather,
]

print("\n--- 工具列表 ---")
for i, tool_obj in enumerate(tools, 1):
    print(f"{i}. {tool_obj.name}")
    print(f"   描述: {tool_obj.description}")

# 也可以用 bind_tools 绑定到模型
print("\n--- 提示: 可以用 model.bind_tools(tools) 绑定到模型 ---")
print("(下节课讲代理时会详细演示)")

print("\n")

# ==================================================
# 总结
# ==================================================
print("=" * 60)
print("第5课演示代码完成!")
print("=" * 60)
print("\n本课要点:")
print("1. @tool 装饰器 - 把函数变成工具")
print("2. 自定义工具属性 - name、description")
print("3. Pydantic Schema - 定义复杂输入")
print("4. 保留参数名 - config 和 runtime 不能用")
print("5. 三种返回类型 - 字符串、对象、Command")