# 统一大模型API调用文档
## 1. 功能说明
封装通义千问大模型调用能力，预留GPT接口扩展位置，统一处理token长度限制、接口超时、密钥错误等异常。
## 2. 核心类 UnifiedLLMClient
方法：safe_invoke(prompt, retry_times)
参数：
- prompt：输入提示词字符串
- retry_times：接口失败重试次数，默认1次
返回值：字典 {code状态码, data模型输出, msg提示信息}
## 3. 异常处理逻辑
1. 输入文本过长：自动按照token分割截断
2. 接口超时/密钥错误：自动重试，重试失败返回错误信息
## 4. 使用示例
```python
from llm_unified_api import UnifiedLLMClient
client = UnifiedLLMClient()
response = client.safe_invoke("你的提问")
print(response["data"])