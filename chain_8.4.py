
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda

# 模型
llm = ChatOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b",
    temperature=0.1
)

# 自定义处理函数1：大写转换
def uppercase(text):
    return text.upper()

# 自定义处理函数2：添加后缀
def add_suffix(text):
    return text + " —— 来自自定义链"

# 构建自定义链
prompt = ChatPromptTemplate.from_template("分析这句话：{text}")
chain = prompt | llm | StrOutputParser() | RunnableLambda(uppercase) | RunnableLambda(add_suffix)

# 运行
print("===== 8.4 自定义链 运行结果 =====")
result = chain.invoke({"text": "RAG技术让大模型更强大"})
print(result)
print("\n✅ 8.4 自定义链 完成！")