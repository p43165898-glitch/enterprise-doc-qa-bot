# ui.py
import streamlit as st
import requests

# 页面配置
st.set_page_config(page_title="知识库问答机器人", layout="centered")
st.title("🤖 企业知识库问答助手")
st.divider()

# 侧边栏状态显示
with st.sidebar:
    st.subheader("系统状态")
    try:
        resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"✅ 已加载 {data['知识片段数']} 条知识")
        else:
            st.error("❌ 后端服务异常")
    except:
        st.warning("⚠️ 后端未启动，请先运行main.py")

# 问答区域
question = st.text_input("请输入问题（如：年假怎么申请？）", placeholder="例如：加班怎么补偿？")
if st.button("立即提问", type="primary"):
    if question.strip() == "":
        st.warning("请输入有效问题！")
    else:
        with st.spinner("AI思考中..."):
            try:
                resp = requests.get(f"http://127.0.0.1:8000/ask?q={question}", timeout=10)
                data = resp.json()
                st.subheader("回答：")
                st.success(data["回答"])
                with st.expander("查看引用原文"):
                    for i, ref in enumerate(data["引用片段"]):
                        st.info(f"片段 {i+1}：{ref}")
            except Exception as e:
                st.error(f"提问失败：{str(e)}")