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

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = "retrieval-augmented-generation"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)
# try:
#     index.delete(delete_all=True)
#     print("Pinecone index reset successfully.")
# except Exception as e:
#     print(f"Could not reset index (probably no namespace yet): {e}")



def hash_file(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def vision_embed_file(file_path, multi_modal_model='gpt-4.1-mini', embedding_model='text-embedding-3-small'):
    def create_file(file_path_inner):
        with open(file_path_inner, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="vision",
            )
            return result.id

    file_id = create_file(file_path)
    # st.write(f"Debug: created file_id: {file_id}")

    time.sleep(2)

    response = client.responses.create(
        model=multi_modal_model,
        input=[{
            'role': 'user',
            'content': [
                {'type': 'input_text', 'text': "What's in this image?"},
                {'type': 'input_image', 'file_id': file_id}
            ]
        }]
    )

    caption = response.output_text
    embedding = client.embeddings.create(input=caption, model=embedding_model).data[0].embedding

    return {'image_caption': caption, 'file_id': file_id, 'embedding': embedding}


def file_to_embed(file):        
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "uploaded.pdf")
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    
        image = convert_from_path(file_path)
        jpeg_path = os.path.join(tmp_dir,'uploaded.jpeg')
        image[0].save(jpeg_path, 'JPEG')

        vector_id = hash_file(jpeg_path)

        res = index.fetch(ids=[vector_id])
        if vector_id in res.vectors:
            st.warning(f"File already uploaded")

        embed_result = vision_embed_file(jpeg_path)
        record = {
            'id': vector_id,
            'values': embed_result['embedding'],
            'metadata': {'caption': embed_result['image_caption']}
        }
    return record

def process_zip(uploaded_zip_file):
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip_file.getbuffer())

        pdf_dir = os.path.join(tmp_dir, "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pdf_dir)

        jpeg_dir = os.path.join(tmp_dir, "jpegs")
        os.makedirs(jpeg_dir, exist_ok=True)
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, pdf_file)
                images = convert_from_path(pdf_path)
                jpeg_path = os.path.join(jpeg_dir, pdf_file[:-4] + ".jpeg")
                images[0].save(jpeg_path, 'JPEG')

        records = []
        for jpeg_file in os.listdir(jpeg_dir):
            jpeg_path = os.path.join(jpeg_dir, jpeg_file)
            vector_id = hash_file(jpeg_path)

            res = index.fetch(ids=[vector_id])
            if vector_id in res.vectors:
                # st.warning(f"Skipping duplicate: {jpeg_file}")
                continue

            embed_result = vision_embed_file(jpeg_path)
            records.append({
                'id': vector_id,
                'values': embed_result['embedding'],
                'metadata': {'caption': embed_result['image_caption']}
            })

        df = pd.DataFrame(records)
        return df

# def upsert_to_pinecone(df):
#     vectors = list(df.itertuples(index=False, name=None))
#     res = index.upsert(vectors=vectors)
#     return res


def get_context(user_query, embed_model='text-embedding-3-small', k=5):
    query_embedding = client.embeddings.create(input=user_query, model=embed_model).data[0].embedding
    query_response = index.query(vector=query_embedding, top_k=k, include_metadata=True)

    contexts = [match.metadata.get('caption', '') for match in query_response.matches]
    return contexts, user_query


def augmented_query(user_query, embed_model='text-embedding-3-small', k=5):
    contexts, query = get_context(user_query, embed_model=embed_model, k=k)
    return "\n\n--------------------------\n\n".join(contexts) + "\n\n--------------------------\n\n" + query

def ask_gpt_response(system_prompt, user_prompt, model='gpt-5-chat-latest'):
  response = client.responses.create(
      model=model,
      input=[
          {"role":"developer",
          "content":system_prompt},
          {"role":"user",
           "content":user_prompt}])
  return response.output_text

primer = f"""
You are a knowledgeable assistant specialized in answering questions about electric utility bills. 
You provide accurate and clear explanations based solely on the bill details and information provided above each question. 
If the information is not sufficient to answer the question, respond truthfully with, "I don't know."
"""

def main():
    st.title("Electricity Bills Visual QA")

    st.markdown("""
    Upload a ZIP file containing your electricity bill **PDFs**.
    This app will process and embed them visually and semantically, then let you query your bills with natural language.
    """)

    # uploaded_zip = st.file_uploader("Upload ZIP of PDFs", type="zip")

    # if uploaded_zip:
    #     with st.spinner("Processing and embedding your files..."):
    #         df = process_zip(uploaded_zip)

    #         if not df.empty:
    #             upsert_response = upsert_to_pinecone(df)
    #             st.success(f"Upserted {len(df)} vectors to Pinecone.")
    #             time.sleep(2)

    #         else:
    #             st.info("No new documents to upsert all were duplicates.")

    
    jpeg_upload = st.file_uploader("Upload PDF file", type='pdf')


    if jpeg_upload:
        jpeg_record = file_to_embed(jpeg_upload)
        upsert_response = index.upsert([tuple(jpeg_record.values())])

    user_query = st.text_input("Ask a question about your electricity bills:")
    if user_query:
        with st.spinner("Searching for answers..."):
            response = ask_gpt_response(system_prompt=primer, user_prompt=augmented_query(user_query))
            st.markdown("Results")
            st.write(response)


if __name__ == "__main__":
    main()
