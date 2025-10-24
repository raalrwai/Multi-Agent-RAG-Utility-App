import os

import pandas as pd
import json

from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool, FunctionTool

import rag
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
    print('NAME: ', name)
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
        instructions=(
            "Handle uploading and retrieving utility bills."
            "Call the relevant tools when needed."
        ),
        handoff_description="Specialist agent for retrieving and uploading bills.",
        tools=[get_bills]
    )
    return billing_agent

async def get_info(name, question):
    query = 'Using the bill belonging to "' + name + '", answer the following question: ' + question
    
    result = await Runner.run(get_agent(), query)
    return result.final_output

if __name__ == '__main__':
    # for tool in get_agent().tools:
    #     if isinstance(tool, FunctionTool):
    #         print(tool.name)
    #         print(tool.description)
    #         print(json.dumps(tool.params_json_schema, indent=2))
    #         print()

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

# class Billing_Agent():
#     def __init__(self):
#         system_instruction = {"role": "system", "content": "You are an agent whose role is to be in charge"
#                         "of managing a database containing bills. This includes uploading and downloading"
#                         "bills, and uploading the chat history once a conversation has been completed. If"
#                         "a requested bill has already been retrieved, return it again instead of downloading"
#                         "it again."}
#         self.input_list = [system_instruction]

#     def make_request(self, query, model='gpt-5-chat-latest'):

#         self.input_list += {"role": "user", "content": query}

#         response = client.responses.create(
#             model=model,
#             tools=tools,
#             input=self.input_list,
#         )    
        
#         self.input_list += response.output

#         # original_stdout = sys.stdout
#         # with open('temp.txt', 'w') as f:
#         #     sys.stdout = f
#         #     print(response)
#         # sys.stdout = original_stdout

#         for item in response.output:
#             if item.type == "function_call":
#                 if item.name in func_dict.keys():
#                     func = func_dict[item.name]                    
#                     response = func(json.loads(item.arguments))
                    
#                     self.input_list.append({
#                             "type": "function_call_output",
#                             "call_id": item.call_id,
#                             "output": json.dumps({
#                             "bill": response
#                             })
#                     })

#         response = client.responses.create(
#             model=model,
#             instructions= query,
#             tools=tools,
#             input=self.input_list,
#         )

#         return response.output_text



# # if __name__ == '__main__':
# #     test_dir = os.path.join(os.getcwd(), 'tests/billing_agent/')
# #     test_questions_df = pd.read_csv(os.path.join(test_dir, 'questions.csv'))

# #     answers_path = os.path.join(test_dir, 'answers.csv')
# #     with open(answers_path, 'w') as f:
# #         for i in range(len(test_questions_df)):
# #             name = test_questions_df.loc[i]['Name']
# #             question = test_questions_df.loc[i]['Question']
# #             answer = '"' + ask_gpt(name,question).replace('\n', ' ').replace('"', "'") +'"'
# #             content = ','.join([name,question,answer]) + '\n'
# #             f.write(content)
                


