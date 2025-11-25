# main.py
from fastapi import FastAPI
from rag import load_knowledge, ask_question, collection
import os

app = FastAPI(title="企业知识库问答系统")

# PDF路径
PDF_PATH = "data/knowledge/员工手册.pdf"

# 启动时加载知识库
@app.on_event("startup")
async def startup():
    try:
        if os.path.exists(PDF_PATH):
            load_knowledge(PDF_PATH)
            print("🚀 后端服务启动成功！")
        else:
            print(f"⚠️ 未找到PDF：{PDF_PATH}")
    except Exception as e:
        print(f"启动错误：{e}")

# 健康检查接口
@app.get("/health")
def health_check():
    return {"状态": "正常", "知识片段数": collection.count()}

# 问答接口
@app.get("/ask")
async def get_answer(q: str):
    answer, refs = ask_question(q)
    return {
        "问题": q,
        "回答": answer,
        "引用片段": refs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)