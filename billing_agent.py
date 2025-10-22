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


def ask_gpt(name, query, model='gpt-5-chat-latest'):

    instruction = 'If the question requires a bill, then retrieve any bills with the name of ' \
    '"' + name + '". Otherwise, repeat the question exactly which is "' + query + '"'

    instruction = 'Check if the question needs a bill to be answered. The question is "'\
    +query+'". If no, just answer the question. If yes, get bills belonging to "' + name +\
    '" and then answer the question. Do not explain your thought process, just answer the '
    'question. If there are no bills found with the given name, return only the error message'
    'of "No bills found with that name"'
    
    # input_list = [{"role": "user", "content": 'find bills belonging to ' + name + ' and answer the following question:' + query}]

    input_list = [{"role": "user", "content": instruction}]


    response = client.responses.create(
        model=model,
        tools=tools,
        input=input_list,
    )    
    
    input_list += response.output

    # original_stdout = sys.stdout
    # with open('temp.txt', 'w') as f:
    #     sys.stdout = f
    #     print(response)
    # sys.stdout = original_stdout

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


    final_instruction = 'You are a friendly assistant who will help customer\'s and answer their question.'

    
    
    # input_list += [{"role": "developer", "content": 'You are a friendly assistant who will help customer\'s and answer' \
    #                'their question. Perform friendly conversation, but if they ask a question that would require '
    #                'information from a bill, make sure that the bill has their name on it, and then answer with as '
    #                'few words as possible. If it does not have their name, reply with the words "No bill found".'}]
    not_found_msg =  "Furthermore, if " + name + "does not show up in the bill' +\
          print out 'No bills found with that name'."
    # response = client.responses.create(
    #     model=model,
    #     instructions=final_instruction,
    #     tools=tools,
    #     input=input_list,
    # )
    response = client.responses.create(
        model=model,
        instructions= query,
        tools=tools,
        input=input_list,
    )

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