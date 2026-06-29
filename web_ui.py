import gradio as gr
from agent_workflow import run_agent
from rag_engine import RAGEngine


def chat_with_agent(user_input):
    """与AI Agent对话"""
    result = run_agent(user_input)
    return result


def rag_search(query):
    """单独测试RAG检索"""
    rag_engine_instance = RAGEngine()
    rag_engine_instance.load_vector_store()
    context = rag_engine_instance.get_context_text(query)
    return context[:2000]  # 限制输出长度


# 创建Gradio界面
with gr.Blocks(title="AI Agent助手") as demo:
    gr.Markdown("# 🤖 AI Agent 智能助手")
    gr.Markdown("基于LangChain + RAG的企业级AI Agent，支持联网搜索和本地知识库检索")

    with gr.Tab("💬 对话模式"):
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="输入你的问题", placeholder="例如：图像分割的常用方法有哪些？")
        clear = gr.Button("清空对话")


        def respond(message, chat_history):
            bot_message = chat_with_agent(message)
            # Gradio 6.x要求字典格式
            return "", chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": bot_message}]


        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot)

    with gr.Tab("📚 RAG知识库检索"):
        query_input = gr.Textbox(label="检索关键词", placeholder="例如：图像分割")
        search_btn = gr.Button("开始检索")
        result_output = gr.Textbox(label="检索结果", lines=15)

        search_btn.click(rag_search, query_input, result_output)

    with gr.Tab("📖 项目说明"):
        gr.Markdown("""
## 项目功能

1. **联网搜索**：查询实时互联网资料
2. **RAG知识库**：基于语义检索查询本地PDF文档
3. **常识问答**：直接回答基础问题

## 使用说明

- 输入问题后，Agent会自动判断需要调用哪种工具
- 如果问题涉及实时资讯，会调用网络搜索
- 如果问题涉及本地文档内容，会调用RAG知识库检索
- 如果是基础常识，会直接回答

## 技术栈

- Python + LangChain
- 通义千问大模型
- Chroma向量数据库
- Gradio Web界面
        """)

if __name__ == "__main__":
    demo.launch(debug=True, theme="soft")