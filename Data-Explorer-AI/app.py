from main import load_csv_as_docs, build_agent
from chains.chart_agent import query_agent
import pandas as pd
import streamlit as st
from utils.csv_utils import dataframe_to_document
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Data Explorer AI", layout="wide")
st.title("📊 Data Explorer AI")

csv_upload=st.file_uploader("Upload a CSV file", type=["csv"])
if csv_upload is not None:
    df=pd.read_csv(csv_upload)
    st.dataframe(df.head())
    if st.button("Load Data"):
        documents=load_csv_as_docs(df)
        agent=build_agent(documents)
        st.session_state.agent=agent
        st.session_state.df=df
        st.success("Data loaded successfully!")

if "agent" in st.session_state and "df" in st.session_state:
    query=st.text_input("Ask a question about the data:")
    if query:
        st.subheader("Answer:")
        result=query_agent(st.session_state.agent,query,st.session_state.df)
        answer=result["answer"]
        chart=result["chart"]
        st.write(answer)
        if chart:
            st.subheader("Chart:")
            st.pyplot(chart)
