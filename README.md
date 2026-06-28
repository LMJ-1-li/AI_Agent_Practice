# 基于LangChain的场景化AI Agent开发项目
## 项目简介
实现全套大模型Prompt工程、智能Agent工作流、多厂商LLM API封装、开源Agent框架调研，适配企业AI业务落地场景。
## 技术栈
Python、LangChain、DashScope通义千问API、Tavily联网检索、PyPDF文档解析
## 功能模块
1. Prompt工程：角色设定、Few-shot、CoT思维链、JSON格式约束，可复用提示词模板库
2. AI Agent：多轮对话、自动工具调用（PDF知识库+联网搜索）、任务自主拆解
3. LLM统一接口：token截断、异常重试、标准化返回，配套技术文档
4. 开源框架调研：Dify/AutoGPT/OpenManus对比评估，轻量化集成测试
## 运行步骤
1. 安装依赖 pip install xxx
2. 在.env填入通义千问、Tavily密钥
3. 依次运行prompt_demo.py、llm_unified_api.py、agent_workflow.py