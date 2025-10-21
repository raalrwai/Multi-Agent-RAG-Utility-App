import os
import sys
import tempfile
import zipfile
import hashlib
import time

import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import json

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

import rag

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = "retrieval-augmented-generation"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)

# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "get_bills",
        "description": "Retrieve bills from a database for a given person.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name that appears on the bills",
                },
            },
            "required": ["name"],
        },
    },       
]
# {
#         "type": "function",
#         "name": "get_response",
#         "description": "Get response for a given query",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "The question being asked.",
#                 },
            
#             },
#             "required": ["query"],
#         },
#     },
def get_bills(name):
    # print('NAME: ', name)
    # print(name['name'])
    return rag.retrieve_bill_embeddings(name['name'])

def get_response():
    return rag



func_dict = {'get_bills': get_bills, 'get_response': get_response}


def ask_gpt(name, query, model='gpt-4.1-mini'):

    input_list = [{"role": "user", "content": 'find bills belonging to ' + name + ' and answer the following question:' + query}]
    instruction = 'You are an assistant who talks with customers to help answer questions ' \
    'about their electric usage. If the question requires information from an electric bill ' \
    'to answer, find bills that belong to "' + name + '". Then if there are no bills with that name'\
    'respond with only "No bills found with that name". The actual question for you to answer is "' +\
     query +'"'
    instruction = 'You are a friendly assistant who will help customer\'s and answer their question' \
    'which is "' + query + '". If this question would not benefit from specific information, answer'
    'it politely, otherwise, retrieve bill information that is under the name "' + name +\
    '". Once you have the bills, make sure that their name is on the bill, otherwise answer with the '
    'prhase "No bills found under that name. Refer to upload." If you do have a bill with their name, '
    'go ahead and answer their original question with information from that bill.'

    # input_list = [{"role": "user", "content": instruction}]

    # 2. Prompt the model with tools defined
    response = client.responses.create(
        model=model,
        tools=tools,
        input=input_list,
    )    
    # Save function call outputs for subsequent requests
    input_list += response.output

    original_stdout = sys.stdout
    # print(response)
    with open('temp.txt', 'w') as f:
        sys.stdout = f
        print(response)
    sys.stdout = original_stdout

    for item in response.output:
        if item.type == "function_call":
            func = func_dict[item.name]
            if item.name == 'get_bills':
                func = get_bills
                # print('ARGUMENTS: ', item.arguments)
                response = func(json.loads(item.arguments))
                
                input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({
                        "bill": response
                        })
                })


    response = client.responses.create(
        model=model,
        instructions= '',
        tools=tools,
        input=input_list,
    )

    # print("Final input:")
    # print(input_list)


    # # 5. The model should be able to give a response!
    # print("Final output:")
    # print(response.model_dump_json(indent=2))
    # print("\n" + response.output_text)

    return response.output_text




if __name__ == '__main__':
    test_dir = os.path.join(os.getcwd(), 'tests/billing_agent/')
    test_questions_df = pd.read_csv(os.path.join(test_dir, 'questions.csv'))

    answers_path = os.path.join(test_dir, 'answers.csv')
    with open(answers_path, 'w') as f:
        for i in range(len(test_questions_df)):
            name = test_questions_df.loc[i]['Name']
            question = test_questions_df.loc[i]['Question']
            answer = '"' + ask_gpt(name,question).replace('\n', ' ').replace('"', "'") +'"'
            content = ','.join([name,question,answer]) + '\n'
            f.write(content)
            break


