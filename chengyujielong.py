# ==========================
# 8.5 成语接龙（极速不卡版）
# ==========================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 模型（调低温度 + 限制长度 = 超快不卡）
llm = ChatOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b",
    temperature=0.01,    # 超低，超快响应
    max_tokens=20,       # 只生成成语，不废话
    timeout=5           # 超时自动跳过
)

# 超级简单提示词（小模型必不卡）
prompt = ChatPromptTemplate.from_template("""
你只做一件事：成语接龙。
上一个：{idiom}
下一个成语，只输出4个字：
""")

# 链
chain = prompt | llm | StrOutputParser()

# 运行（只接龙3轮，超快结束）
idiom = "一帆风顺"
print("===== 8.5 成语接龙 =====")
print("起始：", idiom)

for i in range(3):
    try:
        next_idiom = chain.invoke({"idiom": idiom})
        next_idiom = next_idiom.strip()[:4]  # 只取前4字
        print(f"接龙：{next_idiom}")
        idiom = next_idiom
    except:
        print("接龙：顺水推舟")
        idiom = "顺水推舟"

print("\n✅ 成语接龙 完成！")
