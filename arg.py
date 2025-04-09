import os
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import logging
import glob


# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 加载环境变量 ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请设置 OPENAI_API_KEY 环境变量")

# 设置相对文件开始路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- 全局变量 ---
vector_store = None
qa_chain = None
docs_folder = "资料"  # 资料文件夹路径
vector_store_path = "faiss_index"  # FAISS 索引存储路径

# --- 初始化 Flask 应用 ---
app = Flask(__name__)

# --- 文档处理和 RAG 初始化函数 ---
def initialize_rag_system():
    global vector_store, qa_chain
    logging.info("开始初始化 RAG 系统...")

    # 检查索引是否已存在
    if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
         try:
             logging.info(f"从 '{vector_store_path}' 加载现有的 FAISS 索引...")
             embeddings = OpenAIEmbeddings(api_key=openai_api_key)
             vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
             logging.info("FAISS 索引加载成功。")
         except Exception as e:
             logging.error(f"加载 FAISS 索引失败: {e}. 将重新创建索引。")
             vector_store = None

    # 如果索引不存在或加载失败，则创建新索引
    if vector_store is None:
        logging.info(f"开始处理文档文件夹: {docs_folder}")
        if not os.path.exists(docs_folder):
             logging.error(f"错误：找不到文档文件夹 '{docs_folder}'")
             raise FileNotFoundError(f"错误：找不到文档文件夹 '{docs_folder}'")

        try:
            # 获取所有Word文件
            word_files = glob.glob(os.path.join(docs_folder, "*.docx"))
            if not word_files:
                logging.error(f"在 '{docs_folder}' 文件夹中没有找到Word文件")
                raise FileNotFoundError(f"在 '{docs_folder}' 文件夹中没有找到Word文件")
            
            logging.info(f"找到 {len(word_files)} 个Word文件")
            
            # 加载所有文档
            all_documents = []
            for doc_path in word_files:
                logging.info(f"正在处理文件: {doc_path}")
                loader = Docx2txtLoader(doc_path)
                documents = loader.load()
                if documents:
                    all_documents.extend(documents)
                    logging.info(f"文件 '{doc_path}' 加载成功")
                else:
                    logging.warning(f"文件 '{doc_path}' 没有加载到任何内容")
            
            if not all_documents:
                logging.error("未能从任何Word文档加载内容")
                raise ValueError("未能从任何Word文档加载内容")
            
            logging.info(f"所有文档加载成功，共 {len(all_documents)} 部分")

            # 文本分块
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            texts = text_splitter.split_documents(all_documents)
            if not texts:
                logging.error("文本分割后没有产生任何块")
                raise ValueError("文本分割后没有产生任何块")
            logging.info(f"文本分割成功，共 {len(texts)} 块")

            # 创建Embeddings
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            logging.info("OpenAI Embeddings 初始化成功")

            # 创建向量存储
            logging.info("正在创建 FAISS 向量存储...")
            vector_store = FAISS.from_documents(texts, embeddings)
            logging.info("FAISS 向量存储创建成功")

            # 保存索引
            vector_store.save_local(vector_store_path)
            logging.info(f"FAISS 索引已保存到 '{vector_store_path}'")

        except Exception as e:
            logging.error(f"处理文档和创建向量库时出错: {e}", exc_info=True)
            raise

    llm = ChatOpenAI(model_name="o3-mini", api_key=openai_api_key)
    logging.info("ChatOpenAI LLM 初始化成功 (o3-mini)")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    logging.info("RetrievalQA 链创建成功")
    logging.info("RAG 系统初始化完成")

# --- Flask 路由 ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({'error': '请提供问题'}), 400

        result = qa_chain.invoke({"query": question})
        answer = result.get('result', "抱歉，我无法从文档中找到答案。")
        return jsonify({'answer': answer})

    except Exception as e:
        logging.error(f"处理问题时出错: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    initialize_rag_system()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)