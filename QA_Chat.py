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

    st.title("_Private Data : red[QA Chat]_ 📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금한 것이 있으면 언제든 물어봐 주세요!"}]

    with st.sidebar:
        upload_files = st.file_uploader("Upload your file", type=['pdf', 'txt', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        files_text = get_text(upload_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        if st.session_state.conversation is None:
            st.error("❗ 먼저 파일 업로드 후 'Process' 버튼을 눌러주세요.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                with get_openai_callback() as cb:
                    result = chain({"question": query})
                    st.session_state.chat_history = result['chat_history']

                response = result["answer"]
                source_documents = result["source_documents"]

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

# Helper Functions

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Upload {file_name}")

        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_name)
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(file_name)
        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_name)
        else:
            continue

        documents = loader.load_and_split()
        doc_list.extend(documents)

    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device_map': 'auto'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
