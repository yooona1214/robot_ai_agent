from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import pandas as pd
import numpy as np
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_vector_store_as_retriever(csv_path, str1, str2):
    data = CSVLoader(csv_path)
    data = data.load()
 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)

    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=embedding
    )

    retriever = vectorstore.as_retriever()
    
    retriever.search_kwargs = {'k': 63}

    tool = create_retriever_tool(
        retriever,
        str1,
        str2,
    )

    return tool

def create_vector_store_as_retriever2(csv_path, str1, str2):
    # 1. CSV 파일에서 데이터 로드
    df = pd.read_csv(csv_path)
    data = df.to_dict(orient='records')
    
    # 2. OpenAI 임베딩 생성기 로드
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # 3. 설명을 임베딩으로 변환
    descriptions = list(set([item['Name'] for item in data]))
    # descriptions = list(set([item['Description'] for item in data]))
    duplicates = [description for description in descriptions if descriptions.count(description) > 1]
    print("###########################################중복된 값들:", duplicates)

    # 4. Chroma 벡터 스토어 생성
    vectorstore = Chroma.from_texts(
        texts=descriptions,
        embedding=embedding_model,
        metadatas=data,
    )
    
    # 5. 벡터 스토어를 리트리버로 변환
    retriever = vectorstore.as_retriever(search_type='similarity')
    retriever.search_kwargs = {'k': len(descriptions)}  # 검색할 상위 k개 결과 설정

    # 6. LangChain의 retriever_tool 생성
    tool = create_retriever_tool(
        retriever,
        str1,  # 툴의 이름
        str2   # 툴의 설명
    )

    return tool



