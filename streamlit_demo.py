import streamlit as st
from loguru import logger

import pickle
import os
import logging

from langchain.chains import ConversationalRetrievalChain

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory

# from streamlit_chat import message
# from langchain.memory import StreamlitChatMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)


os.environ["LANGCHAIN_PROJECT"] = "EMS Manual Chat Demo"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__f24a6f2714d04a78aebf583287b842c8"

def main():
    st.set_page_config(
        page_title="NkiaChat",
        page_icon=":books:"
    )
    
    st.title("**EMS Manual :red[QA Chat]**")
    
    # conversation을 사용하기 위해 정의
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    # history를 사용하기 위해 정의
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    file_path = '../../data/RAG document/manual_ko.pkl'
    model_id = 'OrionStarAI/Orion-14B-Chat-RAG'
    files_text = get_text(file_path)
    # text_chunks = get_text_chunks(files_text)
    vectorstore = get_vectorstore()
    llm = get_llm(model_id)
    retriever = get_retriever(llm, files_text, vectorstore)
    
    st.session_state.conversation = get_conversation_chain(retriever)
    
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                          "content": "안녕하세요! EMS 매뉴얼에 대해 궁금하신 것이 있으신가요?"}]
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    history = StreamlitChatMessageHistory(key="chat_messages")
    
    # chat Logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            
            chain = st.session_state.conversation
        
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                # with get_openai_callback() as cb:
                    # st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']
                
                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    
                
        # add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
def get_text(file_path):
    with open(file_path, 'rb') as f:
        docs = pickle.load(f)
        
    return docs
        
def get_vectorstore():
    model_names = "BAAI/bge-m3"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_names,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vectorstore = Chroma(
        collection_name="split_parents", embedding_function=embeddings)
    
    return vectorstore

def get_retriever(llm, docs, vectorstore):
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    retriever.add_documents(docs, ids=None)
    
    MultiQuery_template = '''
    You are an AI language model assistant. 
    Your task is to generate **three different versions** of the given user
    question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. 
    Provide these alternative questions separated by newlines. **Answer in Korean**.
    Original question: {question}
    '''
    MultiQuery_prompt = PromptTemplate(
        template=MultiQuery_template,
        input_variables=['question'],
    )
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm,
        prompt=MultiQuery_prompt
    )
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    
    return retriever_from_llm
    
def get_llm(model_id):
    llm = HuggingFacePipeline.from_model_id(
    model_id=model_id, 
    device=0,               # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
    task="text-generation", # 텍스트 생성
    model_kwargs={"temperature": 0.1,
                "do_sample": True, 
                "max_length": 20000},
    )
    
    return llm


def get_conversation_chain(retriever):
    model_name = 'OrionStarAI/Orion-14B-Chat-RAG'

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

if __name__ == '__main__':
    main()