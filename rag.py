# rag.py（适配通义API text字段返回格式）
import pdfplumber
import os
import numpy as np
from dotenv import load_dotenv
import dashscope

# 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 存储文档片段和向量
doc_chunks = []
doc_embeddings = np.array([])

def cos_sim(a, b):
    """纯Python计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def read_pdf(path):
    """读取PDF文本内容"""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_embedding(texts):
    """调用通义Embedding API"""
    if not texts or len(texts) == 0:
        return []
    
    try:
        response = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v1,
            input=texts,
            api_key=dashscope.api_key
        )
        
        if response.status_code != 200:
            print(f"Embedding API错误：{response.status_code} - {response.message}")
            return []
        if not response.output or "embeddings" not in response.output:
            print(f"Embedding返回格式：{response.output}")
            return []
        
        return [item.get("embedding", []) for item in response.output["embeddings"]]
    
    except Exception as e:
        print(f"Embedding调用异常：{str(e)}")
        return []

def load_knowledge(pdf_path):
    """加载PDF并调用API生成向量"""
    global doc_chunks, doc_embeddings
    
    if not os.path.exists(pdf_path):
        print(f"未找到PDF：{pdf_path}")
        return
    
    text = read_pdf(pdf_path)
    chunks = []
    for i in range(0, len(text), 150):
        chunk = text[i:i+250]
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
    
    if len(chunks) == 0:
        print("PDF无有效文本片段")
        return
    
    embeddings = get_embedding(chunks)
    if len(embeddings) != len(chunks):
        print(f"向量数量不匹配：文本{len(chunks)}段，向量{len(embeddings)}个")
        return
    
    doc_chunks = chunks
    doc_embeddings = np.array(embeddings)
    print(f"✅ 成功加载 {len(chunks)} 个知识片段")

def ask_question(query):
    """问答主逻辑（适配text字段返回格式）"""
    if len(doc_chunks) == 0:
        return "❌ 知识库未加载", []
    
    if not query.strip():
        return "❌ 请输入有效问题", []
    
    # 获取查询向量
    query_emb_list = get_embedding([query.strip()])
    if not query_emb_list or not query_emb_list[0]:
        return "❌ 查询向量生成失败", []
    query_emb = query_emb_list[0]
    
    # 检索相似片段
    similarities = [cos_sim(query_emb, emb) for emb in doc_embeddings]
    top_indices = np.argsort(similarities)[-3:][::-1]
    contexts = [doc_chunks[i] for i in top_indices]
    
    # 调用通义生成回答
    messages = [
        {"role": "system", "content": "仅根据提供的文档内容回答用户问题，不编造信息"},
        {"role": "user", "content": f"参考文档：\n{chr(10).join(contexts)}\n\n问题：{query}\n\n回答："}
    ]
    
    try:
        response = dashscope.Generation.call(
            model="qwen-turbo",
            messages=messages,
            temperature=0.1,
            api_key=dashscope.api_key
        )
        
        if response.status_code != 200:
            return f"❌ API请求失败：{response.message}", contexts
        if not response.output:
            return "❌ API无output返回", contexts
        
        # 优先读取text字段（你的API返回格式）
        if "text" in response.output and response.output["text"].strip():
            answer = response.output["text"].strip()
        # 兼容新版choices格式
        elif "choices" in response.output and response.output["choices"] is not None and len(response.output["choices"]) > 0:
            choice = response.output["choices"][0]
            answer = choice.get("message", {}).get("content", "").strip() or choice.get("text", "").strip()
        else:
            answer = "❌ 未获取到有效回答"
        
        return answer, contexts
    
    except Exception as e:
        print(f"回答调用异常：{str(e)}")
        return f"❌ 调用异常：{str(e)}", contexts

# 兼容main.py的count方法
class FakeCollection:
    def count(self):
        return len(doc_chunks)
collection = FakeCollection()