import asyncio
import os
import base64
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
import utility_functions.rag as rag
import utility_functions.log_generator as log_gen
from our_agents.manager_agent import Manager_Agent
from openai import OpenAI
from agents import SQLiteSession  

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "retrieval-augmented-generation"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)
manager = Manager_Agent()

@st.cache_resource
def make_session(name):
    print("SESSION MADE")
    return SQLiteSession(name, 'session_history.sqlite')

saved_stdout = log_gen.start_log()

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False
if "show_company_modal" not in st.session_state:
    st.session_state.show_company_modal = False

def show_pdf_in_modal(pdf_path):
    """Return a base64 iframe for inline PDF rendering."""
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return base64_pdf

# --- Page config ---
st.set_page_config(page_title="Electricity Bills Visual QA", layout="wide")

# --- Main async app ---
async def main():

    # --- Sidebar ---
    with st.sidebar:
        st.header("Electricity Bills Visual QA")

        if st.button("📄 Learn More", key="learn_more_button"):
            st.session_state.show_company_modal = True

        st.markdown("""
        Upload your **electricity bill PDF** and chat with it.  
        The app will extract, embed, and answer your questions naturally.
        """)

        col1, col2 = st.columns([4, 1])
        with col1:
            pdf_upload = st.file_uploader("Upload PDF file", type="pdf")
        with col2:
            if st.button("ℹ️", key="info_button"):
                st.session_state.show_modal = True
                print("Info button clicked, modal shown")

        has_bill = False
        if pdf_upload:
            rag.file_to_upsert(pdf_upload)
            has_bill = True
            st.success("Bill uploaded and processed!")

        user_name = st.text_input("Full Name:")

    if st.session_state.show_modal:
        try:
            pdf_base64 = show_pdf_in_modal("data/sample_bill.pdf")
        except Exception as e:
            pdf_base64 = ""
            print(f"Error loading PDF: {e}")

        modal_container = st.container()
        with modal_container:
            st.markdown("<h3>Sample Document</h3>", unsafe_allow_html=True)

            pdf_col, req_col, close_col = st.columns([0.7, 0.25, 0.05])

            with close_col:
                if st.button("×", key="modal_close"):
                    st.session_state.show_modal = False
                    print("Modal closed via X")
                    st.rerun()  

            with pdf_col:
                st.markdown(
                    f'''
                    <div style="width:100%; height:90vh; overflow:auto; border:1px solid #ccc; border-radius:8px;">
                        <iframe 
                            src="data:application/pdf;base64,{pdf_base64}" 
                            width="100%" 
                            height="100%" 
                            style="border:none;">
                        </iframe>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            with req_col:
                st.markdown("""
                <div style="padding-left:10px;">
                    <h4>Document requirements:</h4>
                    <ul>
                        <li>Must include Full Name</li>
                        <li>Must show Data Period</li>
                        <li>Should be a single page</li>
                        <li>PDF format only</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    if st.session_state.show_company_modal:
        try:
            company_pdf_base64 = show_pdf_in_modal("data/company_overview.pdf")
        except Exception as e:
            company_pdf_base64 = ""
            print(f"Error loading company PDF: {e}")

        company_modal_container = st.container()
        with company_modal_container:
            st.markdown("<h3>About Our Company</h3>", unsafe_allow_html=True)

            pdf_col, close_col = st.columns([0.95, 0.05])

            with close_col:
                if st.button("×", key="company_modal_close"):
                    st.session_state.show_company_modal = False
                    st.rerun()

            with pdf_col:
                st.markdown(
                    f'''
                    <div style="width:100%; height:80vh; overflow:auto; border:1px solid #ccc; border-radius:8px;">
                        <iframe 
                            src="data:application/pdf;base64,{company_pdf_base64}" 
                            width="100%" 
                            height="100%" 
                            style="border:none;">
                        </iframe>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

    st.title("Chat with Your Bill")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("source"):
                    st.caption(f"Response generated by: {message['source']}")

    user_query = st.chat_input("Ask a question or say hi...")

    if user_query and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query})
        session = make_session(user_name or "guest")
        print(f'[{user_name}] ', user_query)

        with st.spinner("Thinking..."):
            try:
                result = await manager.handle_query(
                    user_query=user_query,
                    user_name=user_name,
                    has_bill=has_bill,
                    session=session
                )
            except Exception as e:
                result = {"response": f"Error: {str(e)}", "source": "System"}

        if result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "source": result.get("source")
            })

        st.rerun()

# --- Async event loop fix for Streamlit ---
def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as e:
        if 'There is no current event loop' in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        else:
            raise

if __name__ == "__main__":
    loop = get_or_create_event_loop()
    loop.run_until_complete(main())

