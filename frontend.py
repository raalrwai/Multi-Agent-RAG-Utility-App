import os
import tempfile
import zipfile
import hashlib
import time

import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


import rag
import billing_agent as bl_agent


load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = "retrieval-augmented-generation"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)

def main():
    st.set_page_config(page_title="Electricity Bills Visual QA", layout="wide")
    st.title("Electricity Bills Visual QA")

    st.sidebar.title("Navigation")
    st.sidebar.info("Use this app to upload your electricity bill PDF and ask questions about it.")
    st.sidebar.markdown("**Example questions:**")
    st.sidebar.markdown("- What was my total bill in July?\n- Why is my bill amount so high? \n- What is the due date for my latest bill?")

    st.markdown("""
    Upload a PDF file containing your electricity bill.
    This app will process and embed them visually and semantically, then let you query your bills with natural language.
    """)

    jpeg_upload = st.file_uploader("Upload PDF file", type='pdf')
    if jpeg_upload:
        rag.file_to_upsert(jpeg_upload)

    user_name = st.text_input("Full Name:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input for new user query
    user_query = st.chat_input("Ask a question about your electricity bills:")

    if user_query and user_query.strip():
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        with st.spinner("Searching for answers..."):
            response = bl_agent.ask_gpt(user_name, user_query)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)

if __name__ == "__main__":
    main()
