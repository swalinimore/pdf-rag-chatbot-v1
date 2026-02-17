import streamlit as st
from pdf_rag_chatbot import RAGEngine

st.title("📄 RAG Chatbot")

if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 3:
        st.error("You can upload only 3 files.")
    else:
        st.write("Documents processing")
        st.session_state.rag.add_document(uploaded_files)
        st.success("Documents processed")

query = st.chat_input("Ask something")


if query:
    st.chat_message("user").write(query)
    answer = st.session_state.rag.answer(query)
    st.chat_message("assistant").write(answer)