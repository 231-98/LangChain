import streamlit as st
import tiktoken
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback

def main():
    st.set_page_config(
        page_title="Dirchat",
        page_icon="📚"
    )

    st.title("_Private Data :red[QA Chat]_📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금한 것이 있으면 언제든 물어봐 주세요!"}]

    with st.sidebar:
        upload_files = st.file_uploader("📄 파일 업로드", type=['pdf', 'txt', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("🔑 OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("📚 Process")

    if process:
        if not openai_api_key:
            st.info("API 키를 입력해주세요.")
            st.stop()

        files_text = get_text(upload_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("질문을 입력해주세요."):
        if st.session_state.conversation is None:
            st.error("❗ 먼저
