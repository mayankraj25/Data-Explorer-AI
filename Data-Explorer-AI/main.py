from utils.csv_utils import dataframe_to_document
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS




def load_csv_as_docs(df):
    documents = dataframe_to_document(df)
    return documents


def build_agent(docs):
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.split_documents(docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    embeddings= OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(chunks,embeddings)
    retriever=vectorstore.as_retriever()

    return {"llm": llm, "retriever": retriever}