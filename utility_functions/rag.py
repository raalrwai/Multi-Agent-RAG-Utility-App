import os
import tempfile
import zipfile
import hashlib
import time

import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

if not os.getenv("RUNNING_IN_DOCKER"):
    load_dotenv()
# load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = "retrieval-augmented-generation"

print(PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


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


def file_to_upsert(file):        
    with tempfile.TemporaryDirectory() as tmp_dir:

        file_path = os.path.join(tmp_dir, "uploaded.pdf")
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        
        doc = pymupdf.open(file_path)
        pix = doc[0].get_pixmap(dpi=300)        
        png_path = os.path.join(tmp_dir,'uploaded.png')  
        pix.save(png_path)
        doc.close()  
      
        vector_id = hash_file(png_path)

        # image = convert_from_path(file_path)
        # jpeg_path = os.path.join(tmp_dir,'uploaded.jpeg')
        # image[0].save(jpeg_path, 'JPEG')

        # vector_id = hash_file(jpeg_path)

        res = index.fetch(ids=[vector_id])
        if vector_id in res.vectors:
            st.warning(f"File already uploaded")

        embed_result = vision_embed_file(png_path)
        record = {
            'id': vector_id,
            'values': embed_result['embedding'],
            'metadata': {'caption': embed_result['image_caption']}
        }
    
    return index.upsert([tuple(record.values())])



def retrieve_bill_embeddings(user_query, embed_model='text-embedding-3-small', k=5):
    # print('QUERY:  ', user_query)
    query_embedding = client.embeddings.create(input=user_query, model=embed_model).data[0].embedding
    query_response = index.query(vector=query_embedding, top_k=k, include_metadata=True)

    contexts = [match.metadata.get('caption', '') for match in query_response.matches]
    return contexts

# def get_response(system_prompt, user_prompt, model='gpt-5-chat-latest'):
#   response = client.responses.create(
#       model=model,
#       input=[
#           {"role":"developer",
#           "content":system_prompt},
#           {"role":"user",
#            "content":user_prompt}])
#   return response.output_text

def history_to_file(history):
    # Uploads history to pinecone DB
    # maybe, idk if we gonna use this yet
    return None

class RAGAgent:
    def __init__(self):
        pass

    def retrieve(self, query):
        return retrieve_bill_embeddings(query)

    def generate_response(self, system_prompt, user_prompt):
        return get_response(system_prompt, user_prompt)