import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 连接本地Qwen
llm = ChatOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b",
    temperature=0.3
)

# ==========================
# 8.3.1 call 调用方式
# ==========================
print("="*50)
print("🔹 8.3.1 call 调用")
prompt1 = ChatPromptTemplate.from_messages([
    ("user", "解释：{word}")
])
chain1 = prompt1 | llm | StrOutputParser()
result1 = chain1.invoke({"word": "五虎上将"})
print("结果：", result1)


print("\n" + "="*50)
print("🔹 8.3.2 run 调用")
prompt2 = ChatPromptTemplate.from_template("用成语形容：{scene}")
chain2 = prompt2 | llm | StrOutputParser()
result2 = chain2.invoke({"scene": "刘备打赢胜仗"})
print("结果：", result2)