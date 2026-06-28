from dotenv import load_dotenv
import os
import dashscope
from dashscope import Generation

# 加载密钥
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 读取模板文件
def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    user_question = "LangChain和LlamaIndex有什么区别？"

    # 1. 基础角色Prompt
    base_prompt = load_prompt("prompt_template/base_prompt.txt")
    full_prompt_1 = base_prompt + "\n问题：" + user_question
    rsp1 = Generation.call(model="qwen-turbo", prompt=full_prompt_1)
    print("=====基础Prompt输出=====")
    print(rsp1.output.text)

    # 2. Few-shot少样本Prompt
    fewshot_prompt = load_prompt("prompt_template/few_shot_prompt.txt")
    full_prompt_2 = fewshot_prompt + "\n问题：" + user_question
    rsp2 = Generation.call(model="qwen-turbo", prompt=full_prompt_2)
    print("\n=====Few-shot输出=====")
    print(rsp2.output.text)

    # 3. CoT思维链+JSON格式Prompt
    cot_prompt = load_prompt("prompt_template/cot_json_prompt.txt")
    full_prompt_3 = cot_prompt + "\n问题：" + user_question
    rsp3 = Generation.call(model="qwen-turbo", prompt=full_prompt_3)
    print("\n=====CoT JSON输出=====")
    print(rsp3.output.text)