from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import utils

OPENAI_API_KEY = ""

class CreateVectorstore:

    def create_vector_store_as_retriever_lg_voc(data, str1, str2):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = text_splitter.split_documents(data)

        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(
            documents=docs, embedding=embedding, persist_directory="db_lg_voc"
        )

        retriever = vectorstore.as_retriever()

        tool = create_retriever_tool(
            retriever,
            str1,
            str2,
        )

        return tool

    def create_vector_store_as_retriever_lg_manual(data, str1, str2):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = text_splitter.split_documents(data)

        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(
            documents=docs, embedding=embedding, persist_directory="db_lg_manual"
        )

        retriever = vectorstore.as_retriever()

        tool = create_retriever_tool(
            retriever,
            str1,
            str2,
        )

        return tool

    def create_vector_store_as_retriever_bear_manual(data, str1, str2):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = text_splitter.split_documents(data)

        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(
            documents=docs, embedding=embedding, persist_directory="db_bear_manual"
        )

        retriever = vectorstore.as_retriever()

        tool = create_retriever_tool(
            retriever,
            str1,
            str2,
        )

        return tool

    def create_vector_store_as_retriever_error(data, str1, str2):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = text_splitter.split_documents(data)

        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(
            documents=docs, embedding=embedding, persist_directory="db_error"
        )

        retriever = vectorstore.as_retriever()

        tool = create_retriever_tool(
            retriever,
            str1,
            str2,
        )

        return tool


def create_vector_store_as_retriever_w_mode(data, str1, str2):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    docs = utils.filter_complex_metadata(docs)

    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embedding)

    retriever = vectorstore.as_retriever()

    tool = create_retriever_tool(
        retriever,
        str1,
        str2,
    )

    return tool
