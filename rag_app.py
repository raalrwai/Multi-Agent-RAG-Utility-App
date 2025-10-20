import os
import tempfile
import zipfile
import hashlib

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


def vision_embed_file(file_path, multi_modal_model='gpt-4.1-mini', embedding_model='text-embedding-3-small'):
    def create_file(file_path):
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="vision",
            )
            return result.id

    file_id = create_file(file_path)

    response = client.responses.create(
        model=multi_modal_model,
        input=[{
            'role': 'user',
            'content': [{
                'type': 'input_text',
                'text': "What's in this image?"
            }, {
                'type': 'input_image',
                'file_id': file_id
            }]
        }]
    )

    caption = response.output_text
    embedding_object = client.embeddings.create(input=caption, model=embedding_model)
    vector = embedding_object.data[0].embedding

    return {'image_caption': caption, 'file_id': file_id, 'embedding': vector}


def hash_file(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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

            existing = index.fetch(ids=[vector_id])
            if vector_id in existing.vectors:
                print(f"Skipping duplicate: {jpeg_file}")
                continue

            # 🤖 Generate embedding and caption
            embed_result = vision_embed_file(jpeg_path)

            records.append({
                'id': vector_id,
                'values': embed_result['embedding'],
                'metadata': {'caption': embed_result['image_caption']}
            })

        df = pd.DataFrame(records)
        return df


def upsert_to_pinecone(df):
    vectors = []
    for _, row in df.iterrows():
        vectors.append((row['id'], row['values'], row['metadata']))
    res = index.upsert(vectors=vectors)
    return res


def get_context(user_query, embed_model='text-embedding-3-small', k=5):
    query_embedding = client.embeddings.create(input=user_query, model=embed_model).data[0].embedding
    query_response = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )

    contexts = []
    for match in query_response.matches:
        contexts.append(match.metadata.get('caption', ''))

    return contexts, user_query


def augmented_query(user_query, embed_model='text-embedding-3-small', k=5):
    contexts, query = get_context(user_query, embed_model=embed_model, k=k)
    return "\n\n--------------------------\n\n".join(contexts) + "\n\n--------------------------\n\n" + query


def main():
    st.title("Electricity Bills Visual QA")

    st.markdown("""
    Upload a ZIP file containing your electricity bill PDFs.
    The app will process, embed images, and enable you to query them.
    """)

    uploaded_zip = st.file_uploader("Upload ZIP of PDFs", type="zip")

    if uploaded_zip:
        with st.spinner("Processing files and uploading embeddings..."):
            df = process_zip(uploaded_zip)
            upsert_response = upsert_to_pinecone(df)
            st.success(f"Upserted {upsert_response.get('upserted_count', 0)} vectors to Pinecone!")

    user_query = st.text_input("Ask a question about your electricity bills:")
    if user_query:
        with st.spinner("Searching for answers..."):
            response = augmented_query(user_query)
            st.markdown("### Results:")
            st.write(response)


if __name__ == "__main__":
    main()
