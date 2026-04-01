# ========== 强制关闭警告 + 关闭联网检查 + 加速运行 ==========
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

# ===================== 正常代码 =====================
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 1. 模型
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b",
    temperature=0.1,  # 调低temperature，减少幻觉
    max_tokens=512
)

# 2. 加载文档
loader = TextLoader("data/sanguoyanyi.txt", encoding='utf-8')
docs = loader.load()

# 3. 分割（调整粒度，避免关键信息被拆分）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 增大文本块，保留完整语义
    chunk_overlap=30,  # 增大重叠，避免关键信息跨块
    separators=["\n\n", "\n", "。", "！", "？", "，", ""]  # 按中文标点分割，更贴合语义
)
chunks = text_splitter.split_documents(docs)
print(f"文档分割完成：{len(chunks)} 块")
print("正在构建向量库（GPU加速中）... 请稍等\n")

# 4. 嵌入模型 —— GPU加速
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# 5. 向量库（提升检索数量，增加命中概率）
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever(
    search_kwargs={"k": 5}  # 检索5个相关片段，提升命中率
)

# 6. 检索链（优化Prompt，明确要求提取名单）
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
    你是专业的信息提取助手，仅根据提供的资料回答问题，步骤如下：
    1. 从上下文【{context}】中提取与问题直接相关的核心信息；
    2. 仅用简洁的语言直接回答问题，不要复述无关内容；
    3. 如果上下文包含名单类信息，直接列出名单。
    """),
    HumanMessagePromptTemplate.from_template("问题：{question}")
])

qa = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True  # 开启返回源文档，方便排查检索结果
)

# ===================== 运行 & 调试 =====================
print("开始回答...\n")
res = qa.invoke("五虎上将有哪些？")
print("答案：", res["result"])

# 调试：打印检索到的源文档，确认是否命中关键信息
print("\n【检索到的相关文本片段】：")
for i, doc in enumerate(res["source_documents"]):
    print(f"\n第{i+1}段：{doc.page_content[:200]}...")