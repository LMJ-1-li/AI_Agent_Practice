from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

# 加载环境变量
load_dotenv()
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")


class RAGEngine:
    """
    基于LangChain + Chroma + DashScope的RAG引擎
    实现文档向量化、向量存储、语义检索全流程
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化RAG引擎

        Args:
            persist_directory: 向量数据库持久化目录
        """
        # 文本切分器：递归字符切分，适合长文档
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 每个chunk 500字符
            chunk_overlap=50,  # 重叠50字符保持上下文
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "?", "!", " "]
        )

        # 通义千问Embedding模型
        self.embeddings = DashScopeEmbeddings(
            dashscope_api_key=dashscope_api_key,
            model="text-embedding-v1"
        )

        # Chroma向量数据库
        self.persist_directory = persist_directory
        self.vector_store = None
        self.retriever = None

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本，移除可能导致编码错误的特殊字符

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text:
            return text

        # 移除Unicode代理对（surrogate pairs）
        cleaned = text.encode('utf-8', errors='ignore').decode('utf-8')

        # 移除其他可能的问题字符
        cleaned = ''.join(char for char in cleaned if ord(char) < 0x110000)

        return cleaned.strip()

    def load_and_split_document(self, file_path: str) -> List[Document]:
        """
        加载PDF文档并切分为chunk

        Args:
            file_path: PDF文件路径

        Returns:
            切分后的文档片段列表
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 清理每个文档页面的内容
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)

        split_docs = self.text_splitter.split_documents(documents)
        print(f"文档切分完成，共生成 {len(split_docs)} 个chunk")
        return split_docs

    def build_vector_store(self, documents: List[Document]):
        """
        构建向量数据库

        Args:
            documents: 文档片段列表
        """
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        # 新版Chroma已自动持久化，无需手动调用persist()
        print(f"向量数据库构建完成，存储路径: {self.persist_directory}")

        # 创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",  # 相似度检索
            search_kwargs={"k": 4}  # 返回Top 4相关片段
        )

    def load_vector_store(self):
        """
        从持久化目录加载向量数据库
        """
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            print(f"向量数据库加载成功，存储路径: {self.persist_directory}")
            return True
        return False

    def retrieve_relevant_documents(self, query: str) -> List[Document]:
        """
        根据查询词检索相关文档片段

        Args:
            query: 用户查询词

        Returns:
            相关文档片段列表
        """
        if not self.retriever:
            raise ValueError("检索器未初始化，请先构建或加载向量数据库")

        # 使用新版LangChain推荐的invoke方法
        relevant_docs = self.retriever.invoke(query)
        print(f"检索到 {len(relevant_docs)} 条相关文档片段")
        return relevant_docs

    def get_context_text(self, query: str) -> str:
        """
        获取检索到的上下文文本（拼接为字符串）

        Args:
            query: 用户查询词

        Returns:
            拼接后的上下文文本
        """
        relevant_docs = self.retrieve_relevant_documents(query)
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"【片段{i}】\n{doc.page_content}\n")
        return "\n".join(context_parts)


if __name__ == "__main__":
    # 测试RAG引擎
    rag_engine = RAGEngine()

    # 检查是否已有向量数据库
    if not rag_engine.load_vector_store():
        # 首次运行：构建向量数据库
        print("首次运行，开始构建向量数据库...")
        documents = rag_engine.load_and_split_document("test.pdf")
        rag_engine.build_vector_store(documents)

    # 测试检索
    query = "LangChain和LlamaIndex有什么区别？"
    context = rag_engine.get_context_text(query)
    print("\n=====检索到的上下文=====")
    print(context)