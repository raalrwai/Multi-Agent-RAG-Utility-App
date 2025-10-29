import os

import pandas as pd
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI

from agents import Agent, Runner, function_tool, FunctionTool # type: ignore

import utility_functions.rag as rag
import asyncio
import streamlit



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)



@function_tool
def get_bills(name: str):
    """ Retrieves bills with the given name from a databse
    Args: 
        name: A string containing the name to be searched for
    """
    # print('NAME: ', name)
    return rag.retrieve_bill_embeddings(name)

# @function_tool
# def upload_bills(bill: streamlit.runtime.uploaded_file_manager.UploadedFile):
#     """ Uploads bills to the database
    
#     Args: 
#         bill: a pdf file that is to be uploaded to the database
#     """
#     return rag.file_to_upsert(bill)


def get_agent():
    billing_agent = Agent(
        name="Billing agent",
        model='gpt-4o-mini',
        instructions=(
            "Handle uploading and retrieving utility bills."
            "Call the relevant tools when needed."
        ),
        handoff_description="Specialist agent for retrieving and uploading bills.",
        tools=[get_bills]
    )
    return billing_agent

async def get_info(name, question, session=None):
    query = 'Using the bill belonging to "' + name + '", answer the following question: ' + question
    
    result = await Runner.run(get_agent(), query, session=session)    
    print('[billing] ', result.final_output, end='\n\n')
    return result.final_output

if __name__ == '__main__':

    test_dir = os.path.join(os.getcwd(), 'tests/billing_agent/')
    test_questions_df = pd.read_csv(os.path.join(test_dir, 'questions.csv'))

    answers_path = os.path.join(test_dir, 'answers.csv')
    with open(answers_path, 'w') as f:
        for i in range(len(test_questions_df)):
            name = test_questions_df.loc[i]['Name']
            question = test_questions_df.loc[i]['Question']
            response = asyncio.run(get_info(name,question))
            answer = '"' + response.replace('\n', ' ').replace('"', "'") +'"'
            content = ','.join([name,question,answer]) + '\n'
            f.write(content)
