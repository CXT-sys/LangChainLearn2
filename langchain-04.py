"""
第4课 结构化输出（Structured Output）- 完整演示代码
"""
import os
import json
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List
from typing_extensions import TypedDict, Annotated

# 加载环境变量（配置API Key）
load_dotenv()

# 可选：Langfuse回调（无依赖可注释）
# from langfuse.langchain import CallbackHandler
# langfuse_handler = CallbackHandler()


# ==================================================
# 示例 1: Pydantic 模型基础
# ==================================================
print("=" * 60)
print("示例 1: Pydantic 模型基础")
print("=" * 60)

class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱地址")
    phone: str = Field(description="电话号码")
    extra_info: Dict[str, Any]

# 查看模型信息
print(f"模型名称: {ContactInfo.__name__}")
print(f"模型描述: {ContactInfo.__doc__}")
print("\n字段:")
for field_name, field_info in ContactInfo.model_fields.items():
    print(f" - {field_name}: {field_info.annotation}")
    print(f"   描述: {field_info.description}")

# 创建实例并序列化
contact = ContactInfo(
    name="张三",
    email="zhang@example.com",
    phone="13800138000",
    extra_info={"aa":"bb"}
)
print(f"\n实例: {contact}")
print(f"转字典: {contact.model_dump()}")
print(f"转JSON: {contact.model_dump_json()}")
print("\n")


# ==================================================
# 示例 2: with_structured_output() 使用
# ==================================================
print("=" * 60)
print("示例 2: with_structured_output() 使用")
print("=" * 60)

def chat():
    try:
        model = init_chat_model("doubao-seed-2.0-lite", 
                                model_provider="openai",
                                # callbacks=[langfuse_handler]  # 可选：启用Langfuse回调
                                )
        model_with_structure = model.with_structured_output(ContactInfo)

        text = "请从以下文本提取联系人信息: 李四, 邮箱 zhang@example.com, 电话 13800138000"
        print(f"输入文本: {text}")

        response = model_with_structure.invoke(text)
        print(f"\n输出类型: {type(response)}")
        print(f"输出: {response}")
        print(f"姓名: {response.name}")
        print(f"邮箱: {response.email}")
        print(f"字典格式: {response.model_dump()}")
    except Exception as e:
        print(f"错误: {e}")
chat()
print("\n")


# ==================================================
# 示例 3: 产品评论分析（带验证）
# ==================================================
print("=" * 60)
print("示例 3: 产品评论分析（带验证）")
print("=" * 60)

class ProductReview(BaseModel):
    """产品评论分析"""
    rating: int | None = Field(
        description="评分 (1-5)",
        ge=1, le=5
    )
    sentiment: Literal["positive", "negative"] = Field(
        description="情感: 正面/负面"
    )
    key_points: List[str] = Field(description="关键点列表")

print("字段验证规则:")
print(" - rating: 1-5 之间")
print(" - sentiment: 只能是 'positive' 或 'negative'")
print(" - key_points: 字符串列表")

try:
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai")
    model_with_structure = model.with_structured_output(ProductReview)

    review = "这个产品太棒了！5星好评。发货速度很快，但价格有点贵。"
    print(f"\n评论: {review}")

    result = model_with_structure.invoke(f"分析这条评论: {review}")
    print(f"评分: {result.rating}")
    print(f"情感: {result.sentiment}")
    print(f"关键点: {result.key_points}")
except Exception as e:
    print(f"错误: {e}")
print("\n")


# ==================================================
# 示例 4: TypedDict 方式
# ==================================================
print("=" * 60)
print("示例 4: TypedDict 方式")
print("=" * 60)

class MovieDict(TypedDict):
    """电影信息"""
    title: Annotated[str, ..., "电影标题"]
    year: Annotated[int, ..., "上映年份"]
    director: Annotated[str, ..., "导演"]
    rating: Annotated[float, ..., "评分 (0-10)"]

print("TypedDict 特点: 仅类型提示, 无运行时验证, 返回字典")

try:
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai")
    model_with_structure = model.with_structured_output(MovieDict)

    result = model_with_structure.invoke("告诉我电影《盗梦空间》的信息")
    print(f"结果类型: {type(result)}")
    print(f"结果: {result}")
    print(f"标题: {result['title']}")
    print(f"年份: {result['year']}")
except Exception as e:
    print(f"错误: {e}")
print("\n")


# ==================================================
# 示例 5: JSON Schema 方式
# ==================================================
print("=" * 60)
print("示例 5: JSON Schema 方式")
print("=" * 60)

json_schema = {
    "title": "Ticket",
    "description": "工单信息",
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["product", "service", "shipping", "billing"],
            "description": "工单类别"
        },
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "严重程度"
        },
        "description": {
            "type": "string",
            "description": "问题描述"
        }
    },
    "required": ["category", "severity", "description"]
}

print("JSON Schema:")
print(json.dumps(json_schema, indent=2, ensure_ascii=False))

try:
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai")
    model_with_structure = model.with_structured_output(
        json_schema,
        method="json_schema"
    )

    ticket_text = "我买的东西收到时已经坏了, 需要立即处理！"
    result = model_with_structure.invoke(f"分类这个工单: {ticket_text}")
    print(f"\n结果: {result}")
except Exception as e:
    print(f"错误: {e}")
print("\n")


# ==================================================
# 示例 6: include_raw 参数
# ==================================================
print("=" * 60)
print("示例 6: include_raw 参数")
print("=" * 60)

class Movie(BaseModel):
    """电影信息"""
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")

try:
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai")
    model_with_structure = model.with_structured_output(
        Movie,
        include_raw=True
    )

    response = model_with_structure.invoke("介绍电影《盗梦空间》")
    print(f"返回的键: {list(response.keys())}")
    print("1. parsed (结构化数据):", response['parsed'])
    print("2. raw (原始AIMessage):", type(response['raw']))
    if response['raw'] and hasattr(response['raw'], 'usage_metadata'):
        print("   Token 使用:", response['raw'].usage_metadata)
    print("3. parsing_error (解析错误):", response['parsing_error'])
except Exception as e:
    print(f"错误: {e}")
print("\n")


# ==================================================
# 示例 7: 嵌套结构
# ==================================================
print("=" * 60)
print("示例 7: 嵌套结构")
print("=" * 60)

class Actor(BaseModel):
    """演员信息"""
    name: str = Field(description="演员姓名")
    role: str = Field(description="扮演角色")

class MovieDetails(BaseModel):
    """电影详细信息"""
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    cast: list[Actor] = Field(description="演员列表")
    genres: list[str] = Field(description="类型")
    budget: float | None = Field(None, description="预算（百万美元）")

print("嵌套Schema结构:")
print(" - MovieDetails 包含 title, year, cast, genres, budget")
print(" - cast 是 list[Actor] 类型，Actor 包含 name 和 role")

try:
    model = init_chat_model("doubao-seed-2.0-lite", model_provider="openai")
    model_with_structure = model.with_structured_output(MovieDetails)

    result = model_with_structure.invoke("详细介绍电影《盗梦空间》")
    print(f"\n标题: {result.title}")
    print(f"年份: {result.year}")
    print(f"类型: {result.genres}")
    print(f"演员:")
    for actor in result.cast:
        print(f"  - {actor.name} 饰演 {actor.role}")
except Exception as e:
    print(f"错误: {e}")
print("\n")


# ==================================================
# 示例 8: 数据提取器工具类
# ==================================================
print("=" * 60)
print("示例 8: 数据提取器工具类")
print("=" * 60)

class DataExtractor:
    """通用数据提取器"""
    def __init__(self, model_name="doubao-seed-2.0-lite", model_provider="openai"):
        try:
            self.model = init_chat_model(model_name, model_provider=model_provider)
        except Exception as e:
            self.model = None
            print(f"警告: 模型初始化失败: {e}")

    def extract(self, schema, text):
        """从文本中提取结构化数据"""
        if not self.model:
            return None
        model_with_structure = self.model.with_structured_output(schema)
        return model_with_structure.invoke(text)

# 定义多个Schema
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

class Company(BaseModel):
    name: str = Field(description="公司名")
    industry: str = Field(description="行业")

try:
    extractor = DataExtractor()
    # 提取人物信息
    text1 = "张三今年 30 岁, 是一名软件工程师"
    person = extractor.extract(Person, text1)
    print(f"人物提取: {person}")

    # 提取公司信息
    text2 = "阿里巴巴是一家电子商务公司"
    company = extractor.extract(Company, text2)
    print(f"公司提取: {company}")
except Exception as e:
    print(f"错误: {e}")
print("\n")


# ==================================================
# 总结
# ==================================================
print("=" * 60)
print("第4课演示代码完成!")
print("=" * 60)
print("\n本课要点:")
print("1. 三种Schema方式 - Pydantic(推荐)、TypedDict、JSON Schema")
print("2. with_structured_output() - 绑定Schema到模型")
print("3. include_raw=True - 同时获取结构化数据和原始信息")
print("4. 嵌套结构 - 支持复杂的嵌套Schema")
print("5. 应用场景 - 数据提取、分类、API格式化")