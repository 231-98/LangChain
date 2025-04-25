import streamlit as st
import tiktoken # 텍스트를 여러개의 청크로 나눌때, 토큰 개수를 세기 위한 라이브러리

from loguru import logger # 로그로 기록
from langchain.chains import ConversationalRetrievalChain#
from langchain_community.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader # pdf
from langchain.document_loaders import Docx2txtLoader # txt
from langchain.document_loaders import UnstructuredPowerPointLoader # power point

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings 

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
        page_title = "Dirchat",
        page_icon="books:"
    )
    
    st.title("_Private Data : red[QA Chat]_ : books:") # _ ~ _ → 제목 / :books: → 아이콘
    
    if "conversation" not in st.session_state: # Conversation 변수 설정
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state: # 채팅 히스토리 저장하기 위해 none으로 설정
        st.session_state.conversation = None   
        
    with st.sidebar:
        upload_files = st.file_uploader("Upload your file", type = ['pdf', 'txt', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAPi API Key", key = "chatbot_api_key", type = "password")       
        process = st.button("Process")

    if process :
        if not openai_api_key:
            st.info("Plase add your OpenAI API key to continue.")     
            st.stop()
        files_text = get_text(upload_files) # upload된 파일을 텍스트로 변환
        text_chunks = get_text_chunks(files_text) # 텍스트 파일을 일정한 길이의 작은조각(청크)로 나눔
        vetorestore = get_vectorstore(text_chunks) # 벡터화
        
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key) # 벡터스토어를 가지고 llm이 답변을 할 수있도록 chain 구성
        
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role" : "assistant",
                                        "content" : "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐 주세요!"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    history = StreamlitChatMessageHistory(key = "chat_messages")
        
    # chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role" : "user", "content" : query})
        
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            
            with st.spinner("Thinking...") :
                result = chain({"question" : query})
                with get_openai_callback() as cb :
                    st.session_state.chat_history = result['chat_history']
                response = result["answer"]
                source_documents = result['source_documents']
                
                st.markdown(response)
                with st.expander("참고 문서 확인"): # 참고문서 확인시 접고 펼수 있는 기능(Streamlit Expander)
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)

# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response            })
        
# cl100k_base 기준으로 몇 토큰인지 계산해주는 함수        
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base") # 약 100,000개의 토큰을 기반으로 만들어진 클러스터형 사전(cl = cluster, 100k = 약 10만 토큰)
    tokens = tokenizer.encode(text)
    return len(tokens)

# 업로드한 파일들을 텍스트로 변환
def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name
        # 파일을 쓰기(w) + **이진(binary) 모드(b)**로 염
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Upload {file_name}")
        
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name :
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name :
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
            
        doc_list.extend(documents) # document들의 목록을 반환 (extend 쓰는 이유 → 요소로 추가)
    
    return doc_list
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap = 100,
        length_function = tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


# 텍스트를 embedding 함 (embedding 하는 이유 : LLM이나 검색시스템은 문장을 이해하지 못하여, 수치형으로 바꿔주어야함)
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True})
    
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

# temperature 0인이유 : RAG는 문서를 검색해서 그대로 정답을 생성하는 구조이므로, 창의성 보다는 사실 기반이 중요함
def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', verbose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain

if __name__ == '__main__':
    main()
