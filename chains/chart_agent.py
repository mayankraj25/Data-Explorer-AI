import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

def query_agent(agent,query,df):
    llm=agent["llm"]
    retriever=agent["retriever"]
    df_summary = df.describe().to_string()
    sample_rows = df.head(5).to_string()
    context="/n/n".join([f"{doc.page_content}" for doc in retriever.get_relevant_documents(query)[:3]])
    full_prompt=f"You are a data analysis agent. Answer the question based on the information provided.\n\nData Preview:{sample_rows}\n\nData Summary:{df_summary}\n\nContext:\n{context}\n\nQuestion: {query}"
    full_prompt+="Just answer the question without any additional information(NO PREAMBLE)"
    answer=llm.invoke(full_prompt).content
    
    chart=None
    if "chart" in query.lower() or "plot" in query.lower():
        try:
            fig,ax=plt.subplots()
            df.plot(ax=ax)
            chart=fig
        except Exception as e:
            pass
    return {"answer": answer, "chart": chart}
            