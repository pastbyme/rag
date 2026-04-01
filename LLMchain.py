from langchain.chains.llm import LLMChain
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
# 新增输出解析器，将模型输出转为纯文本（新版推荐）
from langchain_core.output_parsers import StrOutputParser

# 1. 创建提示词模板
system_message = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant."
)
human_message = HumanMessagePromptTemplate.from_template(
    "{user_question}"
)
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message,
])

# 2. 初始化本地Qwen2.5模型（对接Ollama服务）
chat_model = ChatOpenAI(
    openai_api_key="ollama",  # Ollama兼容OpenAI API，密钥随意填
    base_url="http://localhost:11434/v1",  # Ollama默认端口
    model="qwen2.5:0.5b",  # 必须和Ollama拉取的模型名一致
    temperature=0.5,  # 可选：控制生成随机性
    max_tokens=512  # 可选：控制最大生成长度
)

# 3. 构建LLMChain（新版RunnableSequence：提示词→模型→解析器）
chain = chat_prompt | chat_model | StrOutputParser()

# 4. 测试运行
if __name__ == "__main__":
    response = chain.invoke({"user_question": "你好，介绍一下你自己"})
    print("模型回复：", response)