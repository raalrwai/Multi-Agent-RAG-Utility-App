import os
import streamlit as st
from dotenv import load_dotenv

import rag
from manager_agent import run_manager_query  

load_dotenv()

def main():
    st.set_page_config(page_title="Electricity Bills Visual QA", layout="wide")
    st.title("Electricity Bills Visual QA")

    st.sidebar.title("Navigation")
    st.sidebar.info("Upload your electricity bill PDF and ask questions in natural language.")
    st.sidebar.markdown("**Example questions:**")
    st.sidebar.markdown(
        "- What was my total bill in July?\n"
        "- Why is my bill amount so high?\n"
        "- What is the due date for my latest bill?"
    )

    st.markdown("""
    Upload a PDF file containing your electricity bill. 
    This app will process and embed it visually and semantically, then let you query your bills naturally.
    """)

    pdf_upload = st.file_uploader("Upload PDF file", type='pdf')
    if pdf_upload:
        rag.file_to_upsert(pdf_upload)

    user_name = st.text_input("Full Name:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_query = st.chat_input("Ask a question about your electricity bills:")

    if user_query and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.spinner("Searching for answers..."):
            result = run_manager_query(
                user_query=user_query,
                user_name=user_name,
                document_uploaded=bool(pdf_upload)
            )

        if result:
            st.session_state.messages.append({"role": "assistant", "content": result["response"]})
            with st.chat_message("assistant"):
                st.write(result["response"])
                if result.get("explanation"):
                    with st.expander("Explanation", expanded=False):
                        st.write(result["explanation"])

if __name__ == "__main__":
    main()

