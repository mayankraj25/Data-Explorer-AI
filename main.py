from utils.csv_utils import dataframe_to_document
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def load_csv_as_docs(df):
    documents = dataframe_to_document(df)
    return documents

def build_agent(documents, api_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.2)

    return {
        "llm": llm,
        "retriever": retriever
    }
