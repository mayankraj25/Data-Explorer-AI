from langchain_core.documents import Document

def dataframe_to_document(df):
    doc=[]
    for index,row in df.iterrows():
        content="/n".join([f"{col}:{row[col]}" for col in df.columns])
        doc.append(Document(page_content=content))
    return doc