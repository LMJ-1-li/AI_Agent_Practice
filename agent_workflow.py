from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain_community.document_loaders import PyPDFLoader
from llm_unified_api import UnifiedLLMClient

# 加载环境变量
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# 1. 联网搜索工具（新版独立包，无弃用警告）
search_tool = TavilySearch(max_results=3)

# 2. PDF读取工具
def read_local_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])
    return full_text

# 3. 封装工具
tools = [
    Tool(
        name="网络搜索",
        func=search_tool.run,
        description="查询实时互联网资料、行业对比信息"
    ),
    Tool(
        name="本地PDF知识库",
        func=read_local_pdf,
        description="读取项目内test.pdf文档内容"
    )
]

# 大模型客户端
llm_client = UnifiedLLMClient()

# 手动实现简易Agent
def run_agent(user_question: str):
    system_prompt = """
你是自研轻量化AI Agent，设计思路参考Dify工业级工具调用标准，同时吸收AutoGPT、OpenManus两款开源智能体的优缺点做优化约束，严格遵守执行规范：
## 一、Dify标准化工具调用核心规则（优先遵循）
1. 前置需求判断：先区分用户问题是否需要外部实时网络数据、本地私有PDF文档，禁止凭空编造无依据信息；
2. 单次单工具串行执行：一次推理周期仅选择一类工具调用，不并行多工具，规避AutoGPT无限制循环调用、Token消耗过高的缺陷；
3. 分步执行流程：判断工具→拉取工具返回上下文→二次整合生成最终回答，工具信息不足可二次调用补充；
4. 固定关键词标记工具，仅输出【网络搜索】/【本地PDF知识库】，不输出额外多余文字。

## 二、轻量化设计约束（参考OpenManus极简架构）
1. 仅保留两类核心工具，删减冗余模块，降低部署与算力成本；
2. 超长文档自动截断，避免上下文窗口超限，保障轻量化运行性能。

## 三、工具匹配判断标准
1. 需要实时资讯、开源AI框架落地对比（Dify/AutoGPT/OpenManus/LangChain/LlamaIndex）、最新行业数据 → 输出【网络搜索】
2. 需要读取本地项目私有调研文档test.pdf → 输出【本地PDF知识库】
3. 仅基础常识、无外部资料依赖，仅凭固有知识即可作答 → 不输出任何工具关键词，直接完整回答用户问题
"""
    judge_result = llm_client.safe_invoke(f"{system_prompt}\n用户问题：{user_question}")["data"]

    # 轻量化Dify式单工具串行调用逻辑，一次只执行一类工具
    if "网络搜索" in judge_result:
        # Dify标准：先拉取外部搜索上下文，再送入模型整合输出
        search_data = search_tool.run(user_question)
        final_prompt = f"""
【联网检索上下文】{search_data}
用户需求：{user_question}
要求：结合真实检索到的资料，条理清晰分析，若涉及Dify、AutoGPT、OpenManus、LangChain、LlamaIndex需对比落地成本、部署可行性、工具调度优劣，输出轻量化落地建议。
"""
        return llm_client.safe_invoke(final_prompt)["data"]
    elif "本地PDF知识库" in judge_result:
        # Dify轻量化私有知识库读取流程，截断超长文本避免超限
        pdf_text = read_local_pdf("test.pdf")
        final_prompt = f"【本地知识库文档片段】{pdf_text[:2000]}\n用户需求：{user_question}\n基于文档私有内容精准作答"
        return llm_client.safe_invoke(final_prompt)["data"]
    else:
        # 无工具需求，直接返回原生回答
        return judge_result

if __name__ == "__main__":
    # 适配本次Dify/AutoGPT/OpenManus调研测试提问
    test_question = "对比Dify、AutoGPT、OpenManus三款开源AI Agent的落地成本、工具调用稳定性与轻量化部署可行性"
    result = run_agent(test_question)
    print("=====Agent最终回答=====")
    print(result)
    # 注释PDF读取测试，若不需要读取可暂时屏蔽避免报错
    # print("\n=====本地PDF前500字符=====")
    # print(read_local_pdf("test.pdf")[:500])