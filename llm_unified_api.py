from dotenv import load_dotenv
import os
import time
import dashscope
from dashscope import Generation
from langchain_text_splitters import TokenTextSplitter

# 加载密钥
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未读取到DASHSCOPE_API_KEY，请检查.env文件")
dashscope.api_key = api_key

class UnifiedLLMClient:
    def __init__(self):
        self.splitter = TokenTextSplitter(chunk_size=2800, chunk_overlap=50)
        self.model_name = "qwen-turbo"

    def safe_invoke(self, prompt: str, retry_times: int = 1) -> dict:
        chunks = self.splitter.split_text(prompt)
        final_prompt = chunks[0] if chunks else prompt

        current_retry = 0
        while current_retry <= retry_times:
            try:
                rsp = Generation.call(
                    model=self.model_name,
                    prompt=final_prompt,
                    result_format="text"
                )
                print("接口原始返回日志：", rsp)
                # 关键修复：status_code=200 才是成功标识
                if rsp.status_code == 200:
                    return {
                        "code": 200,
                        "data": rsp.output.text,
                        "msg": "调用成功"
                    }
                else:
                    raise Exception(f"错误码:{rsp.code}, 错误信息:{rsp.message}")
            except Exception as e:
                current_retry += 1
                time.sleep(1)
                if current_retry > retry_times:
                    return {
                        "code": 500,
                        "data": "",
                        "msg": f"重试{retry_times}次失败，详情：{str(e)}"
                    }

    @property
    def qwen_llm(self):
        from langchain_community.llms import DashScope
        return DashScope(model_name="qwen-turbo", dashscope_api_key=dashscope.api_key)

if __name__ == "__main__":
    client = UnifiedLLMClient()
    res = client.safe_invoke("简单介绍LangChain框架")
    print("最终返回结果：", res)